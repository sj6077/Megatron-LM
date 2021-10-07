"""Tuning throught for given model and resources"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import math
import os
import sys
import time
from typing import Any, List, Union 

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._C._distributed_c10d import ReduceOp

from pretrain_gpt import model_provider as gpt_model_provider
from pretrain_gpt import forward_step as gpt_forward_step

from megatron import get_args
from megatron import get_timers
from megatron.arguments import parse_args
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.initialize import initialize_megatron
import megatron.mpu as mpu
from megatron.optimizer.optimizer import MegatronOptimizer
from megatron.schedules import backward_step
from megatron.training import cyclic_iter, setup_model_and_optimizer
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.language_model import Embedding

from tuner.comm_helper import CommType, CommHelper

tuner_logger = logging.getLogger('tuner')

NUM_AVERAGE = 100

@dataclass
class Models:
    model_with_pre_process: List[torch.nn.Module]
    model_without_pre_or_post_process: List[torch.nn.Module]
    model_with_post_process: List[torch.nn.Module]

@dataclass
class Optimizers:
    optimizer_with_pre_process: MegatronOptimizer
    optimizer_without_pre_or_post_process: MegatronOptimizer
    optimizer_with_post_process: MegatronOptimizer

@dataclass(frozen=True)
class Task:
    dp: int
    mp: int
    pp: int

@dataclass
class CommTime:
    dp: float = 0.0
    mp: float = 0.0
    pp: float = 0.0

@dataclass
class TimeOrMemory:
    pre_process: float
    single_layer: float
    post_process: float

    def __init__(self, pre_process, single_layer, post_process):
        self.pre_process = max(0, pre_process)
        self.single_layer = max(0, single_layer)
        self.post_process = max(0, post_process)

    def get_total(self, num_layers):
        return self.pre_process + self.single_layer * num_layers + self.post_process

@dataclass
class Log:
    pre_process: list = field(default_factory=list)
    single_layer: list = field(default_factory=list)
    post_process: list = field(default_factory=list)

    def set(self, value, is_pre_process, is_post_process):
        if is_pre_process:
            self.pre_process = value
        elif is_post_process:
            self.post_process = value
        else:
            self.single_layer = value

@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass

def get_single_layer_model(model_provider, args_defaults=None):
    initialize_megatron(args_defaults=args_defaults)

    args = get_args()
    # disable save and load checkpoints
    args.load = None
    args.save = None

    # Get single transformer layer model
    original_num_layers = args.num_layers
    
    optimizers = {}

    args.num_layers = 1
    def model_provider_with_pre_process(pre_process=True, post_process=True):
        return model_provider(pre_process=True, post_process=False)
    model_with_pre_process, optimizer_with_pre_process, _ = setup_model_and_optimizer(
            model_provider_with_pre_process)

    def model_provider_without_pre_or_post_process(pre_process=True, post_process=True):
        return model_provider(pre_process=False, post_process=False)
    model, optimizer, _ = setup_model_and_optimizer(
            model_provider_without_pre_or_post_process)

    def model_provider_with_post_process(pre_process=True, post_process=True):
        model = model_provider(pre_process=False, post_process=True)
        return model
    model_with_post_process, optimizer_with_post_process, _ = setup_model_and_optimizer(
            model_provider_with_post_process)
    unwrapped_model_with_post_process = unwrap_model(
        model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0].language_model
    orig_embedding = unwrap_model(
        model_with_pre_process, (torchDDP, LocalDDP, Float16Module))[0].language_model.embedding
    # set embedding for loss computation
    unwrapped_model_with_post_process.embedding = orig_embedding

    models = Models(model_with_pre_process=model_with_pre_process,
                    model_without_pre_or_post_process=model,
                    model_with_post_process=model_with_post_process)

    optimizers = Optimizers(optimizer_with_pre_process=optimizer_with_pre_process,
                            optimizer_without_pre_or_post_process=optimizer,
                            optimizer_with_post_process=optimizer_with_post_process)
    args.num_layers = original_num_layers
    return models, optimizers

def get_train_dataset(dataset='gpt'):
    if dataset != 'gpt':
        raise NotImplementedError
    args = get_args()
    train_val_test_num_samples = [args.global_batch_size, 0, 0]
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    return train_ds

def get_train_data_iterator(train_data_iterator):
    args = get_args()
    train_dataloader = build_pretraining_data_loader(
            train_data_iterator, 0)
    train_data_iterator = iter(cyclic_iter(train_dataloader))
    return train_data_iterator

def get_forward_step_time(comm_logs, forward_step_func, train_data_iterator,
                          model, input_tensor, compute_loss = False, num_try=NUM_AVERAGE+1):
    assert len(comm_logs) == 0
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)

    for i in range(num_try):
        # keep only the last communication logs
        comm_logs.clear()
        if i == 1 or num_try == 1:
            torch.cuda.synchronize()
            s = time.time()

        output, loss_func = forward_step_func(train_data_iterator, model)
        if compute_loss:
            output = loss_func(output)
            loss, loss_reduced = output
            output = loss
    torch.cuda.synchronize()
    e = time.time()
    return output, (e - s) / max(1, num_try - 1)

def get_backward_step_time(comm_logs, optimizer, input_tensor,
                           output_tensor, output_tensor_grad,
                           num_try=NUM_AVERAGE+1):
    assert len(comm_logs) == 0
    args = get_args()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    for i in range(num_try):
        # keep only the last communication logs
        comm_logs.clear()
        if i == 1 or num_try == 1:
            torch.cuda.synchronize()
            s = time.time()

        # Backward pass.
        if output_tensor_grad is None:
            output_tensor = optimizer.scale_loss(output_tensor)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad,
                                retain_graph=i < num_try - 1)

    torch.cuda.synchronize()
    e = time.time()

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad, (e - s) / max(num_try - 1, 1)

def get_optimizer_time(comm_logs, model: torch.nn.Module, optimizer: MegatronOptimizer):
    assert len(comm_logs) == 0

    for param in model.parameters():
        if param.requires_grad:
            if optimizer.params_have_main_grad:
                param.main_grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()
            else:
                param.grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()

    for i in range(NUM_AVERAGE + 1):
        comm_logs.clear()
        if i == 1:
            torch.cuda.synchronize()
            s = time.time()
        optimizer.step()
    torch.cuda.synchronize()
    e = time.time()

    return (e - s) / NUM_AVERAGE

def do_forward_backward(comm_logs, forward_step_func, models,
                        optimizers, train_data_iterator,
                        pre_process=False, post_process=False,
                        input_tensor_shape=None):
    if pre_process:
        input_tensor = None
        model = models.model_with_pre_process[0]
        optimizer = optimizers.optimizer_with_pre_process
    elif post_process:
        input_tensor = torch.randn(list(input_tensor_shape)).cuda()
        input_tensor.requires_grad = True
        model = models.model_with_post_process[0]
        optimizer = optimizers.optimizer_with_post_process
    else:
        input_tensor = torch.randn(list(input_tensor_shape)).cuda()
        input_tensor.requires_grad = True
        model = models.model_without_pre_or_post_process[0]
        optimizer = optimizers.optimizer_without_pre_or_post_process

    # do forward and backward to get peak memory
    for i in range(2):
        if i == 0:
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            # execute forward and backward to get peak memory usage
            num_try = 1
        else:
            # execute kernel multiple times to get correct kernel time
            num_try = NUM_AVERAGE + 1

        # do forward
        output, forward_time = get_forward_step_time(
                comm_logs,
                forward_step_func,
                train_data_iterator,
                model,
                input_tensor,
                compute_loss=post_process,
                num_try=num_try)
        forward_backward_comm_logs = comm_logs.copy()
        comm_logs.clear()

        activation_shape = output.size()
        activation_size = output.nelement() * output.element_size()

        # do backward
        if post_process:
            output_tensor_grad = None
        else:
            output_tensor_grad = torch.randn(list(activation_shape)).cuda()
            output_tensor_grad.requires_grad = True

        _, backward_time = get_backward_step_time(
                comm_logs,
                optimizer,
                input_tensor,
                output,
                output_tensor_grad,
                num_try=num_try)
        for group, logs in comm_logs.items():
            if group not in forward_backward_comm_logs:
                forward_backward_comm_logs[group] = logs
            else:
                forward_backward_comm_logs[group] += logs
        comm_logs.clear()

        if i == 0:
            peak_memory = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()

    return forward_time, backward_time, activation_shape, activation_size, peak_memory, forward_backward_comm_logs

def get_iter_time_estimation(forward_times, backward_times, optimizer_times,
                             mp_forward_backward_times, mp_opt_times):
    """Get iter time estimation as milliseconds"""
    print("forward_times", forward_times)
    print("backward_times", backward_times)
    print("optimizer_times", optimizer_times)
    print("mp_forward_backward_times", mp_forward_backward_times)
    print("mp_opt_times")

    args = get_args()
    assert args.pipeline_model_parallel_size == args.data_parallel_size == 1
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size

    kernel_forward_time = forward_times.get_total(num_layers) * 1000
    kernel_backward_time = backward_times.get_total(num_layers) * 1000
    optimizer_time = optimizer_times.get_total(num_layers) * 1000
    mp_forward_backward_time = mp_forward_backward_times.get_total(num_layers) * 1000
    mp_opt_time = mp_opt_times.get_total(num_layers) * 1000

    print('kernel forward time', int(kernel_forward_time * num_micro_batches),
          'kernel_backward_time', int(kernel_backward_time * num_micro_batches),
          'optimizer_time', int(optimizer_time),
          'mp_forward_backward_time', int(mp_forward_backward_time * num_micro_batches),
          'mp_opt_time', int(mp_opt_time))

    mb_time = kernel_forward_time + kernel_backward_time + mp_forward_backward_time
    iter_time = mb_time * num_micro_batches + optimizer_time + mp_opt_time
    return iter_time

def tensor_nested_iterator(item):
    if isinstance(item, dict):
        for value in item.values():
            for val in tensor_nested_iterator(value):
                yield val
    elif isinstance(item, list):
        for value in item:
            for val in tensor_nested_iterator(value):
                yield val
    else:
        yield item

def get_unique_param_size(params_list):
    param_size = 0
    param_ids = set()
    for params in params_list:
        for param in params:
            if not isinstance(param, torch.Tensor):
                continue
            if id(param) in param_ids:
                continue
            param_ids.add(id(param))
            param_size += param.nelement() * param.element_size()
    return param_size

def get_param_and_grad_sizes(models: Models, optimizers: Optimizers):
    """Get parameter size for pre_process, single_layer, post_process as bytes"""
    
    with_pre_process_param_size = get_unique_param_size(
        [tensor_nested_iterator(optimizers.optimizer_with_pre_process.state_dict()),  # optimizer states (32 bit)
        optimizers.optimizer_with_pre_process.get_parameters(),  # optimizer params (32 bit)
        models.model_with_pre_process[0].parameters()])  # model params (32 bit or 16bit)

    without_pre_or_post_process_param_size = get_unique_param_size(
        [tensor_nested_iterator(optimizers.optimizer_without_pre_or_post_process.state_dict()),
        optimizers.optimizer_without_pre_or_post_process.get_parameters(),
        models.model_without_pre_or_post_process[0].parameters()])

    # exclude embedding params
    orig_embedding = unwrap_model(
        models.model_with_pre_process, (torchDDP, LocalDDP, Float16Module))[0].language_model.embedding
    unwrapped_model_with_post_process = unwrap_model(
        models.model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0].language_model
    unwrapped_model_with_post_process.embedding = None
    with_post_process_param_size = get_unique_param_size(
        [tensor_nested_iterator(optimizers.optimizer_with_post_process.state_dict()),
        optimizers.optimizer_with_post_process.get_parameters(),
        models.model_with_post_process[0].parameters()])
    unwrapped_model_with_post_process.embedding = orig_embedding

    with_pre_process_param_size -= without_pre_or_post_process_param_size
    with_post_process_param_size -= without_pre_or_post_process_param_size
    param_sizes = TimeOrMemory(with_pre_process_param_size,
                               without_pre_or_post_process_param_size,
                               with_post_process_param_size)

    with_pre_process_grad_size = get_unique_param_size(
            [optimizers.optimizer_with_pre_process.get_parameters()])
    without_pre_or_post_process_grad_size = get_unique_param_size(
            [optimizers.optimizer_without_pre_or_post_process.get_parameters()])
    with_post_process_grad_size = get_unique_param_size(
            [optimizers.optimizer_with_post_process.get_parameters()])
    with_pre_process_grad_size -= without_pre_or_post_process_grad_size
    with_post_process_grad_size -= without_pre_or_post_process_grad_size
    grad_sizes = TimeOrMemory(with_pre_process_grad_size,
                              without_pre_or_post_process_grad_size,
                              with_post_process_grad_size)

    # optimizer grad sizes are always 32bits
    args = get_args()
    if not args.accumulate_allreduce_grads_in_fp32 and args.fp16:
        grad_sizes.pre_process /= 2
        grad_sizes.single_layer /= 2
        grad_sizes.post_process /= 2
    return param_sizes, grad_sizes

def get_required_gpu_memory(param_sizes, grad_sizes, activation_size, peak_memories):
    args = get_args()
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size

    assert args.activations_checkpoint_method == 'uniform'

    param_size = param_sizes.get_total(num_layers) / 1024 / 1024
    peak_memory = max(peak_memories.pre_process, peak_memories.single_layer, peak_memories.post_process) / 1024 / 1024
    checkpoint_size = (activation_size * num_layers // args.activations_checkpoint_num_layers) / 1024 / 1024
    if num_micro_batches > 1 or args.accumulate_allreduce_grads_in_fp32:
        grad_size = grad_sizes.get_total(num_layers) / 1024 / 1024
    else:
        grad_size = max(0, grad_sizes.single_layer - activation_size / args.activations_checkpoint_num_layers) * num_layers / 1024 / 1024
        grad_size += (grad_sizes.pre_process + grad_sizes.post_process) / 1024 / 1024
    print("param_size", param_size, "peak_memory", peak_memory, "checkpoint_size", checkpoint_size, "grad_size", grad_size)
    return param_size + peak_memory + checkpoint_size + grad_size

@dataclass(frozen=True)
class Config:
    micro_batch_size: int
    global_batch_size: int
    dp: int = 1
    mp: int = 1
    pp: int = 1
    parallelism_order: str = 'dp,pp,mp'

class DistributedWrapperContext:
    _DEFAULT_INIT = dist.init_process_group
    _DEFAULT_IS_INIT = dist.is_initialized
    _DEFAULT_NEW_GROUP = dist.new_group
    _DEFAULT_WORLD_SIZE = dist.get_world_size
    _DEFAULT_RANK = dist.get_rank
    _DEFAULT_BARRIER = dist.barrier
    _DEFAULT_ALLREDUCE = dist.all_reduce
    _DEFAULT_BROADCAST = dist.broadcast
    _DUMMY_GROUPS = {}
    _COMM_CALLED = defaultdict(list) # comm_group -> comm_type, tensor_dtype, tensor_shape

    def new_group(*args, **kwargs):
        ranks = args[0]
        assert ranks is not None

        group_name = f"dummy_group{len(DistributedWrapperContext._DUMMY_GROUPS)}"
        DistributedWrapperContext._DUMMY_GROUPS[group_name] = len(ranks)
        return group_name

    def convert_group(group):
        assert group is not None

        if not group.startswith('dummy_group'):
            return group

        if group == mpu.get_model_parallel_group():
            new_group = "model_parallel_group"
        elif group == mpu.get_data_parallel_group():
            new_group = "data_parallel_group"
        elif group == mpu.get_embedding_group():
            new_group = "embedding_group"
        elif group == mpu.get_tensor_model_parallel_group():
            new_group = "tensor_model_parallel_group"
        else:
            assert group == mpu.get_pipeline_model_parallel_group()
            new_group = "pipeline_model_parallel_group"

        return new_group

    def get_world_size(world_size):
        def _get_world_size(group=None):
            if group is None:
                return world_size
            return DistributedWrapperContext._DUMMY_GROUPS[group]
        return _get_world_size

    def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
        comm_group = DistributedWrapperContext.convert_group(group)
        DistributedWrapperContext._COMM_CALLED[comm_group].append((
                CommType.ALLREDUCE, tensor.dtype, list(tensor.size())))

    def broadcast(tensor, src, group=None, async_op=False):
        comm_group = DistributedWrapperContext.convert_group(group)
        DistributedWrapperContext._COMM_CALLED[comm_group].append((
                CommType.BROADCAST, tensor.dtype, list(tensor.size())))

    def dummy_func_with_return(return_value=None):
        def dummy_func(*args, **kwargs):
            return return_value
        return dummy_func

    @staticmethod
    def patch_dist_func(world_size):
        setattr(dist, 'init_process_group', DistributedWrapperContext.dummy_func_with_return())
        setattr(dist, 'is_initialized', DistributedWrapperContext.dummy_func_with_return(True))
        setattr(dist, 'new_group', DistributedWrapperContext.new_group)
        setattr(dist, 'get_world_size', DistributedWrapperContext.get_world_size(world_size))
        setattr(dist, 'get_rank', DistributedWrapperContext.dummy_func_with_return(0))
        setattr(dist, 'barrier', DistributedWrapperContext.dummy_func_with_return())
        setattr(dist, 'all_reduce', DistributedWrapperContext.all_reduce)
        setattr(dist, 'broadcast', DistributedWrapperContext.broadcast)

    @staticmethod
    def unpatch_dist_func():
        setattr(dist, 'init_process_group', DistributedWrapperContext._DEFAULT_INIT)
        setattr(dist, 'is_initialized', DistributedWrapperContext._DEFAULT_IS_INIT)
        setattr(dist, 'new_group', DistributedWrapperContext._DEFAULT_NEW_GROUP)
        setattr(dist, 'get_world_size', DistributedWrapperContext._DEFAULT_WORLD_SIZE)
        setattr(dist, 'get_rank', DistributedWrapperContext._DEFAULT_RANK)
        setattr(dist, 'barrier', DistributedWrapperContext._DEFAULT_BARRIER)
        setattr(dist, 'all_reduce', DistributedWrapperContext._DEFAULT_ALLREDUCE)
        setattr(dist, 'broadcast', DistributedWrapperContext._DEFAULT_BROADCAST)

class Estimator:
    def __init__(self, world_size, num_gpus_per_node, model_name='gpt'):
        self.world_size = world_size
        self.num_gpus_per_node = num_gpus_per_node
        self.model_name = model_name
        assert self.world_size >= self.num_gpus_per_node

        self.forward_times = {}  # mp, mb -> single layer forward time
        self.backward_times = {}
        self.optimizer_times = {}  # mp -> single_layer_optimizer_time
        self.param_sizes = {} # mp -> single layer parameter size including optimizer states
        self.grad_sizes = {}  # mp -> single layer gradient size

        self.activation_size = {}  # (mp, mb) -> activation size
        self.peak_memories = {}  # (mp, mb) -> peak memory

        self.comm_helper = CommHelper()
        self.curr_task_to_rank = {}

        self.mp_forward_backward_comm_logs = defaultdict(Log) # (mp, mb) -> mp comm logs for forward and backward
        self.mp_opt_comm_logs = defaultdict(Log) # mp -> mp comm logs for optimizer
        self.mp_forward_backward_times = {} # (mp, mb, num_node) -> mp_comm_time for forward and backward
        self.mp_opt_times = {} # (mp, mb, num_node) -> mp_comm_time for optimizer

    def __enter__(self):
        DistributedWrapperContext.patch_dist_func(self.world_size)
        self.curr_mp = self.num_gpus_per_node
        if self.model_name == 'gpt':
            # initialize model with mp=world_size, dp=1, pp =1
            sys.argv += ['--tensor-model-parallel-size', str(self.world_size)]
            self.models, self.optimizers = get_single_layer_model(
                    gpt_model_provider,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
            self.train_ds = get_train_dataset(dataset='gpt')
            self.forward_step_func = gpt_forward_step
        else:
            raise NotImplementedError
        return self

    def __exit__(self, type, value, trace_back):
        DistributedWrapperContext.unpatch_dist_func()
        self.comm_helper.terminate()
    
    def _get_models_and_optimizers(self):
        mp = get_args().tensor_model_parallel_size
        if mp == self.curr_mp:
            return self.models, self.optimizers
        
        # TODO(SJ): handle oom for the large model
        if self.model_name == 'gpt':
            del self.models
            del self.optimizers
            torch.cuda.empty_cache()
            os.environ['WORLD_SIZE'] = str(mp)
            get_args().tensor_model_parallel_size = mp
            self.models, self.optimizers = get_single_layer_model(
                    gpt_model_provider,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
            os.environ['WORLD_SIZE'] = self.world_size
            self.curr_mp = mp
            return self.models, self.optimizers
        else:
            raise NotImplementedError

    def _get_optimizer_times(self):
        mp = get_args().tensor_model_parallel_size
        if mp not in self.optimizer_times:
            self._set_optimizer_time_and_memory()
        return self.optimizer_times[mp]

    def _set_optimizer_time_and_memory(self):
        models, optimizers = self._get_models_and_optimizers()

        model = models.model_with_pre_process[0]
        optimizer = optimizers.optimizer_with_pre_process
        pre_process_optimizer_time = get_optimizer_time(
                DistributedWrapperContext._COMM_CALLED,
                model, optimizer)
        pre_process_comm_logs = DistributedWrapperContext._COMM_CALLED.copy()
        self._set_comm_logs(pre_process_comm_logs, is_compute=False, pre_process=True)
        DistributedWrapperContext._COMM_CALLED.clear()

        model = models.model_with_post_process[0]
        optimizer = optimizers.optimizer_with_post_process
        post_process_optimizer_time = get_optimizer_time(
                DistributedWrapperContext._COMM_CALLED,
                model, optimizer)
        single_layer_comm_logs = DistributedWrapperContext._COMM_CALLED.copy()
        self._set_comm_logs(single_layer_comm_logs, is_compute=False)
        DistributedWrapperContext._COMM_CALLED.clear()

        model = models.model_without_pre_or_post_process[0]
        optimizer = optimizers.optimizer_without_pre_or_post_process
        single_layer_optimizer_time = get_optimizer_time(
                DistributedWrapperContext._COMM_CALLED,
                model, optimizer)
        post_process_comm_logs = DistributedWrapperContext._COMM_CALLED.copy()
        self._set_comm_logs(post_process_comm_logs, is_compute=False, post_process=True)
        DistributedWrapperContext._COMM_CALLED.clear()

        post_process_optimizer_time -= single_layer_optimizer_time
        pre_process_optimizer_time -= single_layer_optimizer_time

        args = get_args()
        mp = args.tensor_model_parallel_size
        optimizer_times = TimeOrMemory(pre_process_optimizer_time,
                                       single_layer_optimizer_time,
                                       max(0, post_process_optimizer_time))  # post_process 
        self.optimizer_times[mp] = optimizer_times

        param_sizes, grad_sizes = get_param_and_grad_sizes(models, optimizers)
        self.param_sizes[mp] = param_sizes
        self.grad_sizes[mp] = grad_sizes

    def _set_curr_task_to_rank(self, config):
        assert config.parallelism_order
        parallelism_order = config.parallelism_order.split(',')
        parallel_degree = []
        for parallelism in parallelism_order:
            if parallelism == 'dp':
                parallel_degree.append(config.dp)
            elif parallelism == 'pp':
                parallel_degree.append(config.pp)
            else:
                parallel_degree.append(config.mp)

        dp_order = parallelism_order.index('dp')
        mp_order = parallelism_order.index('mp')
        pp_order = parallelism_order.index('pp')
        
        self.curr_task_to_rank.clear()
        for p_rank_1 in range(parallel_degree[0]):
            for p_rank_2 in range(parallel_degree[1]):
                for p_rank_3 in range(parallel_degree[2]):
                    p_rank_list = [p_rank_1, p_rank_2, p_rank_3]
                    task = Task(dp=p_rank_list[dp_order],
                                mp=p_rank_list[mp_order],
                                pp=p_rank_list[pp_order])
                    self.curr_task_to_rank[task] = len(self.curr_task_to_rank)

    def _get_num_node(self, is_mp, is_dp, is_pp):
        """Num inter-machine communication for a group of communication
           e.g, mp_communication is required for the same dp and pp but different mp.
        """
        machine_ids = set()
        for task, rank in self.curr_task_to_rank.items():
            machine_id = rank // self.num_gpus_per_node
            if is_mp and task.dp == 0 and task.pp == 0:
                machine_ids.add(machine_id)
            elif is_dp and task.mp == 0 and task.pp == 0:
                machine_ids.add(machine_id)
            elif is_pp and task.dp == 0 and task.mp == 0:
                machine_ids.add(machine_id)
        return len(machine_ids)

    def _get_comm_ranks(self, num_req_gpus, num_node):
        """Get comm rank as many as num_req_gpus that over num_node machines"""

        assert num_node <= math.ceil(self.world_size / self.num_gpus_per_node)
        assert num_req_gpus <= num_node * self.num_gpus_per_node

        comm_ranks = []
        for node_id in range(num_node):
            gpus_to_assign = num_req_gpus // num_node
            if node_id < num_req_gpus % num_node:
                gpus_to_assign += 1
            node_start_rank = node_id * self.num_gpus_per_node
            comm_ranks.extend([node_start_rank + gpu_id for gpu_id in range(gpus_to_assign)])
        return comm_ranks

    def _get_comm_times(self, comm_logs, is_mp=False, is_dp=False, is_pp=False):
        """Get communication times for each communication group of a single layer"""
        args = get_args()
        comm_time_per_num_node = defaultdict(CommTime)

        DistributedWrapperContext.unpatch_dist_func()
        if is_mp:
            min_num_node = math.ceil(args.tensor_model_parallel_size / self.num_gpus_per_node)
            max_num_node = min(args.tensor_model_parallel_size,
                                math.ceil(self.world_size / self.num_gpus_per_node))
            parallel_degree = args.tensor_model_parallel_size
        elif is_dp:
            min_num_node = math.ceil(args.data_parallel_size / self.num_gpus_per_node)
            max_num_node = min(args.data_parallel_size,
                                math.ceil(self.world_size / self.num_gpus_per_node))
            parallel_degree = args.data_parallel_size
        elif is_pp:
            min_num_node = 1
            max_num_node = min(2, math.ceil(self.world_size / self.num_gpus_per_node))
            parallel_degree = min(2, args.pipeline_model_parallel_size)
        else:
            raise NotImplementedError(f"{comm_group} is not supported")

        for num_node in range(min_num_node, max_num_node + 1):
            if is_pp and num_node not in [0, 2]:
                continue
            comm_ranks = self._get_comm_ranks(parallel_degree, num_node)
            for comm_type, tensor_dtype, tensor_shape in comm_logs:
                single_comm_time = self.comm_helper.request_comm(
                        comm_type, comm_ranks, tensor_dtype, tensor_shape)
                if is_mp:
                    comm_time_per_num_node[num_node].mp += single_comm_time
                elif is_dp:
                    comm_time_per_num_node[num_node].dp += single_comm_time
                elif is_pp:
                    comm_time_per_num_node[num_node].pp += single_comm_time
                else:
                    raise NotImplementedError
        DistributedWrapperContext.patch_dist_func(self.world_size)
        return comm_time_per_num_node

    def _get_mp_forward_backward_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size
        num_node = self._get_num_node(is_mp=True, is_dp=False, is_pp=False)
        key = (mp, mb, num_node)
        if key in self.mp_forward_backward_times:
            return self.mp_forward_backward_times[key]

        if (mp, mb) not in self.mp_forward_backward_comm_logs:
            self._set_forward_backward_time_and_memory()

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].pre_process
        pre_process_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].single_layer
        single_layer_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].post_process
        post_process_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)

        for num_node in single_layer_comm_time_per_num_node:
            pre_process_comm_time = pre_process_comm_time_per_num_node[num_node]
            single_layer_comm_time = single_layer_comm_time_per_num_node[num_node]
            post_process_comm_time = post_process_comm_time_per_num_node[num_node]

            self.mp_forward_backward_times[(mp, mb, num_node)] = TimeOrMemory(
                    max(0, pre_process_comm_time.mp - single_layer_comm_time.mp),
                    single_layer_comm_time.mp,
                    max(0, post_process_comm_time.mp - single_layer_comm_time.mp))
        return self.mp_forward_backward_times[key]

    def _get_mp_opt_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        num_node = self._get_num_node(is_mp=True, is_dp=False, is_pp=False)
        key = (mp, num_node)
        if key in self.mp_opt_times:
            return self.mp_opt_times[key]

        if mp not in self.mp_opt_comm_logs:
            # execute optimizers are required to get optimizer communication logs
            self._set_optimizer_time_and_memory()

        comm_logs = self.mp_opt_comm_logs[mp].pre_process
        pre_process_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)
        
        comm_logs = self.mp_opt_comm_logs[mp].single_layer
        single_layer_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_opt_comm_logs[mp].post_process
        post_process_comm_time_per_num_node = self._get_comm_times(comm_logs, is_mp=True)

        for num_node in single_layer_comm_time_per_num_node:
            pre_process_comm_time = pre_process_comm_time_per_num_node[num_node]
            single_layer_comm_time = single_layer_comm_time_per_num_node[num_node]
            post_process_comm_time = post_process_comm_time_per_num_node[num_node]

            self.mp_opt_times[(mp, num_node)] = TimeOrMemory(
                    pre_process_comm_time.mp - single_layer_comm_time.mp,
                    single_layer_comm_time.mp,
                    post_process_comm_time.mp - single_layer_comm_time.mp)

        return self.mp_opt_times[key]

    def _set_comm_logs(self, comm_logs, is_compute, pre_process=False, post_process=False):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        for comm_group, comms_to_measure in comm_logs.items():
            is_mp = comm_group in ['tensor_model_parallel_group', 'model_parallel_group']
            is_dp = comm_group == 'data_parallel_group'
            is_pp = comm_group == 'pipeline_model_parallel_group'
            if is_mp:
                if is_compute:
                    self.mp_forward_backward_comm_logs[(mp, mb)].set(comms_to_measure, pre_process, post_process)
                else:
                    self.mp_opt_comm_logs[mp].set(comms_to_measure, pre_process, post_process)
            else:
                tuner_logger.info(f"{comm_group} is not suppported for now")

    def _set_forward_backward_time_and_memory(self):
        """Set forward and backward times, activation size, and peak memory. """
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size
        key = (mp, mb)

        global_batch_size = args.global_batch_size
        args.global_batch_size = 1
        train_data_iterator = get_train_data_iterator(self.train_ds)
        models, optimizers = self._get_models_and_optimizers()

        # prevent ddp comm
        context_handler = dummy_handler
        if isinstance(models.model_with_pre_process, torchDDP):
            context_handler = models.model_with_pre_process.no_sync

        activation_shape = None
        activation_size = None

        # Ignore small data loader/pmp communication
        comm_logs = DistributedWrapperContext._COMM_CALLED
        for comm_group, comms_to_measure in comm_logs.items():
            for _, _, tensor_shape in comms_to_measure:
                assert tensor_shape == [1]
        DistributedWrapperContext._COMM_CALLED.clear()

        with context_handler():
            pre_process_forward_time, pre_process_backward_time, \
                    activation_shape, _, pre_process_peak_memory, pre_process_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, pre_process=True)
            self._set_comm_logs(pre_process_comm_logs, is_compute=True, pre_process=True)

            single_layer_forward_time, single_layer_backward_time, \
                    activation_shape, activation_size, single_layer_peak_memory, \
                    single_layer_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, input_tensor_shape=activation_shape)
            self._set_comm_logs(single_layer_comm_logs, is_compute=True)

            post_process_forward_time, post_process_backward_time, \
                    _, _, post_process_peak_memory, post_process_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, post_process=True,
                                    input_tensor_shape=activation_shape)
            self._set_comm_logs(post_process_comm_logs, is_compute=True, post_process=True)

        pre_process_forward_time -= single_layer_forward_time
        post_process_forward_time -= single_layer_forward_time
        self.forward_times[key] = TimeOrMemory(pre_process_forward_time,
                                               single_layer_forward_time,
                                               post_process_forward_time)

        pre_process_backward_time -= single_layer_backward_time
        post_process_backward_time -= single_layer_backward_time
        self.backward_times[key] = TimeOrMemory(pre_process_backward_time,
                                                single_layer_backward_time,
                                                post_process_backward_time)

        self.peak_memories[key] = TimeOrMemory(pre_process_peak_memory,
                                               single_layer_peak_memory,
                                               post_process_peak_memory)

        self.activation_size[key] = activation_size
        args.global_batch_size = global_batch_size

    def _get_compute_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.forward_times:
            self._set_forward_backward_time_and_memory()
        return self.forward_times[key], self.backward_times[key]

    def _get_param_and_grad_sizes(self):
        mp = get_args().tensor_model_parallel_size
        if mp not in self.param_sizes:
            self._set_optimizer_time_and_memory()
        return self.param_sizes[mp], self.grad_sizes[mp]

    def _get_activation_size(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.activation_size:
            self._set_forward_backward_time_and_memory()
        return self.activation_size[key]

    def _get_peak_memories(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.peak_memories:
            self._set_forward_backward_time_and_memory()

        return self.peak_memories[key]

    def get_iter_time(self, config):

        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        self._set_curr_task_to_rank(config)

        try:
            forward_times, backward_times = self._get_compute_times()
            optimizer_times = self._get_optimizer_times()
            mp_forward_backward_times = self._get_mp_forward_backward_times()
            mp_opt_times = self._get_mp_opt_times()

            iter_time = get_iter_time_estimation(
                    forward_times, backward_times, optimizer_times,
                    mp_forward_backward_times, mp_opt_times)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                tuner_logger.info(f"OOM for {config}")
                return 0
            raise e
        return iter_time

    def get_max_gpu_memory(self, config):

        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        self._set_curr_task_to_rank(config)

        try:
            param_sizes, grad_sizes = self._get_param_and_grad_sizes()
            activation_sizes = self._get_activation_size()
            peak_memories = self._get_peak_memories()

            print("param_sizes", param_sizes)
            print("grad_sizes", grad_sizes)
            print("activation_sizes", activation_sizes)
            print("peak_memories", peak_memories)

            req_gpu_memory = get_required_gpu_memory(param_sizes, grad_sizes, activation_sizes, peak_memories)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                tuner_logger.info(f"OOM for {config}")
                return 0
            raise e
        return req_gpu_memory
