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
torch_cuda_synchronize = torch.cuda.synchronize

NUM_AVERAGE = 20

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

    def get_total(self, num_layers):
        return self.pre_process + self.single_layer * num_layers + self.post_process

    def get_max(self, num_layers):
        return max(self.pre_process, self.post_process) + self.single_layer * num_layers

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

    args = get_args()
    # disable save and load checkpoints
    args.load = None
    args.save = None

    # Get single transformer layer model
    optimizers = {}
    assert args.num_layers == args.pipeline_model_parallel_size == 1
    def model_provider_without_pre_or_post_process(pre_process=True, post_process=True):
        return model_provider(pre_process=False, post_process=False)
    model, optimizer, _ = setup_model_and_optimizer(
            model_provider_without_pre_or_post_process)
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))[0]
    named_params = {}
    for n, p in unwrapped_model.named_parameters():
        named_params[n] = p

    def model_provider_with_pre_process(pre_process=True, post_process=True):
        new_model = model_provider(pre_process=True, post_process=False)
        for n, p in new_model.named_parameters():
            if n in named_params:
                print("pre, same module", n)
                p.data = named_params[n].data
            else:
                print("pre, diff module", n)
        return new_model
    model_with_pre_process, optimizer_with_pre_process, _ = setup_model_and_optimizer(
            model_provider_with_pre_process)
    unwrapped_model_with_pre_process = unwrap_model(
            model_with_pre_process, (torchDDP, LocalDDP, Float16Module))[0]

    def model_provider_with_post_process(pre_process=True, post_process=True):
        new_model = model_provider(pre_process=False, post_process=True)
        for n, p in new_model.named_parameters():
            if n in named_params:
                print("post, same module", n)
                p.data = named_params[n].data
            else:
                print("post, diff module", n)
        return new_model
    model_with_post_process, optimizer_with_post_process, _ = setup_model_and_optimizer(
            model_provider_with_post_process)
    # embedding set after optimizer is created so that the embedding is excluded from the optimizer
    unwrapped_model_with_post_process = unwrap_model(
            model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0]
    unwrapped_model_with_post_process.language_model.embedding = \
            unwrapped_model_with_pre_process.language_model.embedding

    models = Models(model_with_pre_process=model_with_pre_process,
                    model_without_pre_or_post_process=model,
                    model_with_post_process=model_with_post_process)

    optimizers = Optimizers(optimizer_with_pre_process=optimizer_with_pre_process,
                            optimizer_without_pre_or_post_process=optimizer,
                            optimizer_with_post_process=optimizer_with_post_process)
    embedding = unwrapped_model_with_pre_process.language_model.embedding
    return models, optimizers, embedding.word_embeddings.weight.size()

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
                          model, input_tensor, compute_loss = False):
    assert len(comm_logs) == 0
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)

    # keep only the last communication logs
    comm_logs.clear()

    output, loss_func = forward_step_func(train_data_iterator, model)
    if compute_loss:
        output = loss_func(output)
        loss, loss_reduced = output
        output = loss

    return output

def get_backward_step_time(comm_logs, optimizer, input_tensor,
                           output_tensor, output_tensor_grad):
    assert len(comm_logs) == 0
    args = get_args()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    comm_logs.clear()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad

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
            torch_cuda_synchronize()
            s = time.time()
        optimizer.step()
    torch_cuda_synchronize()
    e = time.time()
    opt_time = (e - s) / NUM_AVERAGE
    return opt_time

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
    for i in range(NUM_AVERAGE + 1):
        if i == 0:
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        # do forward
        output = get_forward_step_time(
                comm_logs,
                forward_step_func,
                train_data_iterator,
                model,
                input_tensor,
                compute_loss=post_process)
        if i == 0:
            forward_backward_comm_logs = comm_logs.copy()

            activation_shape = output.size()
            activation_size = output.nelement() * output.element_size()

            # do backward
            if post_process:
                output_tensor_grad = None
            else:
                output_tensor_grad = torch.randn(list(activation_shape)).cuda()
                output_tensor_grad.requires_grad = True
        comm_logs.clear()

        get_backward_step_time(
                comm_logs,
                optimizer,
                input_tensor,
                output,
                output_tensor_grad)

        if i == 0:
            for group, logs in comm_logs.items():
                if group not in forward_backward_comm_logs:
                    forward_backward_comm_logs[group] = logs
                else:
                    forward_backward_comm_logs[group] += logs

            torch_cuda_synchronize()
            peak_memory = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
            s = time.time()

        comm_logs.clear()

    torch_cuda_synchronize()
    e = time.time()
    forward_backward_time = (e - s) / NUM_AVERAGE
    return forward_backward_time, activation_shape, activation_size, peak_memory, forward_backward_comm_logs

def get_iter_time_estimation(forward_backward_times,
                             optimizer_times,
                             mp_forward_backward_times,
                             pp_warmup_comm_time_per_stage,
                             pp_steady_comm_time_per_stage,
                             pp_embedding_sync_time):
    """Get iter time estimation as milliseconds"""
    print("forward_backward_times", forward_backward_times)
    print("optimizer_times", optimizer_times)
    print("mp_forward_backward_times", mp_forward_backward_times)
    print("pp_warmup_comm_time_per_stage", pp_warmup_comm_time_per_stage)
    print("pp_steady_comm_time_per_stage", pp_steady_comm_time_per_stage)
    print("pp_embedding_sync_time", pp_embedding_sync_time)

    args = get_args()
    assert args.data_parallel_size == 1
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size

    warmup_kernel_forward_backward_time = forward_backward_times.get_total(num_layers) * 1000
    warmup_mp_forward_backward_time = mp_forward_backward_times.get_total(num_layers) * 1000
    warmup_pp_forward_backward_time = 0
    num_layers_per_stage = num_layers / args.pipeline_model_parallel_size
    for stage in range(args.pipeline_model_parallel_size):
        warmup_pp_forward_backward_time += \
                pp_warmup_comm_time_per_stage[stage]
    warmup_pp_forward_backward_time *= 1000
    warmup_time = warmup_kernel_forward_backward_time + \
            warmup_mp_forward_backward_time + \
            warmup_pp_forward_backward_time
    print("warmup_kernel_forward_backward_time", warmup_kernel_forward_backward_time,
          "warmup_mp_forward_backward_time", warmup_mp_forward_backward_time,
          "warmup_pp_forward_backward_time", warmup_pp_forward_backward_time,
          "warmup_time", warmup_time)

    if args.pipeline_model_parallel_size == 1:
        stage_time = warmup_time
        stage_optimizer_time = optimizer_times.get_total(num_layers) * 1000
    else:
        stage_time = 0
        for stage in range(args.pipeline_model_parallel_size):
            curr_stage_time = 0
            if stage == 0:
                curr_stage_time += forward_backward_times.pre_process
                curr_stage_time += mp_forward_backward_times.pre_process
            elif stage == args.pipeline_model_parallel_size - 1:
                curr_stage_time += forward_backward_times.post_process
                curr_stage_time += mp_forward_backward_times.post_process

            curr_stage_time += forward_backward_times.single_layer * num_layers_per_stage
            curr_stage_time += mp_forward_backward_times.single_layer * num_layers_per_stage
            curr_stage_time += pp_steady_comm_time_per_stage[stage]

            stage_time = max(stage_time, curr_stage_time)
            print(f"stage time of {stage}: {curr_stage_time}")
        stage_time *= 1000

        # the first or last stage finishes parameter update at last
        stage_optimizer_time = max(optimizer_times.pre_process, optimizer_times.post_process)
        stage_optimizer_time += optimizer_times.single_layer * num_layers_per_stage
        stage_optimizer_time *= 1000

    pp_embedding_sync_time *= 1000
    iter_time = warmup_time + stage_time * (num_micro_batches - 1)
    iter_time += stage_optimizer_time + pp_embedding_sync_time
    print(f"warmup_time - {warmup_time}, stage_time - {stage_time * (num_micro_batches - 1)}")
    print(f"optimizer_time - {stage_optimizer_time}, embedding_sync_time - {pp_embedding_sync_time}")
    print(f"iter_time - {iter_time}")
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

    # exclude embedding params from post_process
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
    pp = args.pipeline_model_parallel_size
    num_layers_per_stage = num_layers / pp
    assert args.activations_checkpoint_method == 'uniform'
    
    peak_memory = max(peak_memories.pre_process, peak_memories.single_layer, peak_memories.post_process) / 1024 / 1024
    checkpoint_size = (activation_size * (num_layers_per_stage // args.activations_checkpoint_num_layers - 1)) / 1024 / 1024
    if pp == 1:
        param_size = param_sizes.get_total(num_layers) / 1024 / 1024
    else:
        param_size = param_sizes.get_max(num_layers_per_stage) / 1024 / 1024

    if num_micro_batches > 1 or args.accumulate_allreduce_grads_in_fp32:
        if pp == 1:
            grad_size = grad_sizes.get_total(num_layers) / 1024 / 1024
        else:
            grad_size = grad_sizes.get_max(num_layers_per_stage) / 1024 / 1024
    else:
        grad_size = max(0, grad_sizes.single_layer - activation_size / args.activations_checkpoint_num_layers)
        grad_size *= num_layers_per_stage / 1024 / 1024
        if pp == 1:
            grad_size += (grad_sizes.pre_process + grad_sizes.post_process) / 1024 / 1024
        else:
            grad_size += max(grad_sizes.pre_process, grad_sizes.post_process) / 1024 / 1024 
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
    CURR_CONFIG = None
   
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
            new_group = DistributedWrapperContext.convert_group(group)
            curr_config = DistributedWrapperContext.CURR_CONFIG
            if curr_config is None:
                return DistributedWrapperContext._DUMMY_GROUPS[group]

            if new_group ==  "model_parallel_group":
                return curr_config.pp * curr_config.mp
            elif new_group == "data_parallel_group":
                return curr_config.dp
            elif new_group == "embedding_group":
                return 0 if curr_config.pp == 1 else 2
            elif new_group == "tensor_model_parallel_group":
                return curr_config.mp
            else:
                assert new_group == "pipeline_model_parallel_group"
                return curr_config.pp
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
        setattr(torch.cuda, 'synchronize', DistributedWrapperContext.dummy_func_with_return())

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
        setattr(torch.cuda, 'synchronize', torch_cuda_synchronize)

class Estimator:
    def __init__(self, world_size, num_gpus_per_node, model_name='gpt'):
        self.world_size = world_size
        self.num_gpus_per_node = num_gpus_per_node
        self.model_name = model_name
        assert self.world_size >= self.num_gpus_per_node

        self.forward_backward_times = {}  # mp, mb -> single layer forward time
        self.optimizer_times = {}  # mp -> single_layer_optimizer_time
        self.param_sizes = {} # mp -> single layer parameter size including optimizer states
        self.grad_sizes = {}  # mp -> single layer gradient size

        self.activation_size = {}  # (mp, mb) -> activation size
        self.peak_memories = {}  # (mp, mb) -> peak memory

        self.comm_helper = CommHelper()
        self.curr_task_to_rank = {}

        self.mp_forward_backward_comm_logs = defaultdict(Log) # (mp, mb) -> mp comm logs for forward and backward
        #self.mp_forward_backward_times = {} # (mp, mb, num_node) -> mp_comm_time for forward and backward

        self.curr_mp = None
        self.models = None
        self.optimizers = None
        self.embedding_shape = None

    def __enter__(self):
        DistributedWrapperContext.patch_dist_func(self.world_size)
        #self.curr_mp = self.num_gpus_per_node
        if self.model_name == 'gpt':
            args_defaults = args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
            initialize_megatron(args_defaults=args_defaults)
            self.train_ds = get_train_dataset(dataset='gpt')
            self.forward_step_func = gpt_forward_step
        else:
            raise NotImplementedError
        return self

    def __exit__(self, type, value, trace_back):
        DistributedWrapperContext.unpatch_dist_func()
        self.comm_helper.terminate()
    
    def _get_models_and_optimizers(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        if mp == self.curr_mp:
            return self.models, self.optimizers
        dp = args.data_parallel_size
        
        # TODO(SJ): handle oom for the large model
        if self.model_name == 'gpt':
            if self.curr_mp:
                del self.models
                del self.optimizers
            torch.cuda.empty_cache()
            os.environ['WORLD_SIZE'] = str(mp * dp)

            # set pp1 and num_layers1 to get single layer model
            original_pp = args.pipeline_model_parallel_size
            original_layers = args.num_layers
            original_config = DistributedWrapperContext.CURR_CONFIG
            assert original_pp == original_config.pp
            new_config = Config(global_batch_size=original_config.global_batch_size,
                                micro_batch_size=original_config.micro_batch_size,
                                pp=1,
                                mp=original_config.mp,
                                dp=original_config.dp,
                                parallelism_order=original_config.parallelism_order)
            DistributedWrapperContext.CURR_CONFIG = new_config
            args.num_layers = 1
            args.pipeline_model_parallel_size = 1

            self.models, self.optimizers, self.embedding_shape = \
                    get_single_layer_model(
                    gpt_model_provider,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

            args.pipeline_model_parallel_size = original_pp
            args.num_layers = original_layers
            DistributedWrapperContext.CURR_CONFIG = original_config
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
        for comms_to_measure in DistributedWrapperContext._COMM_CALLED.values():
            for _, _, tensor_shape in comms_to_measure:
                assert tensor_shape == [1]
        DistributedWrapperContext._COMM_CALLED.clear()

        model = models.model_with_post_process[0]
        optimizer = optimizers.optimizer_with_post_process
        post_process_optimizer_time = get_optimizer_time(
                DistributedWrapperContext._COMM_CALLED,
                model, optimizer)
        for comms_to_measure in DistributedWrapperContext._COMM_CALLED.values():
            for _, _, tensor_shape in comms_to_measure:
                assert tensor_shape == [1]
        DistributedWrapperContext._COMM_CALLED.clear()

        model = models.model_without_pre_or_post_process[0]
        optimizer = optimizers.optimizer_without_pre_or_post_process
        single_layer_optimizer_time = get_optimizer_time(
                DistributedWrapperContext._COMM_CALLED,
                model, optimizer)
        for comms_to_measure in DistributedWrapperContext._COMM_CALLED.values():
            for _, _, tensor_shape in comms_to_measure:
                assert tensor_shape == [1]
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
        DistributedWrapperContext.CURR_CONFIG = config
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

    def _get_comm_ranks(self, is_mp=False, is_dp=False):
        """Get comm rank as many as num_req_gpus that over num_node machines"""
        args = get_args()
        mp = args.tensor_model_parallel_size
        dp = args.data_parallel_size
        comm_ranks_per_task = {}
        for task, rank in self.curr_task_to_rank.items():
            if is_mp:
                key = (task.dp, task.pp)
                parallelism_degree = mp
                parallelism_rank = task.mp
            elif is_dp:
                key = (task.mp, task.pp)
                parallelism_degree = dp
                parallelism_rank = task.dp
            if key not in comm_ranks_per_task:
                comm_ranks_per_task[key] = [-1] * parallelism_degree
                
            comm_ranks_per_task[key][parallelism_rank] = rank

        comm_ranks = {}
        for task, rank in self.curr_task_to_rank.items():
            if is_mp:
                comm_ranks[rank] = comm_ranks_per_task[(task.dp, task.pp)]
            elif is_dp:
                comm_ranks[rank] = comm_ranks_per_task[(task.mp, task.pp)]
        return comm_ranks

    def _get_comm_times(self, comm_logs, is_mp=False, is_dp=False):
        """Get communication times for each communication group of a single layer"""
        args = get_args()

        DistributedWrapperContext.unpatch_dist_func()
        comm_ranks = self._get_comm_ranks(is_mp, is_dp)
        comm_times = CommTime()

        for comm_type, tensor_dtype, tensor_shape in comm_logs:
            # the same time for the collective communication
            single_comm_time_per_rank = self.comm_helper.request_comm(
                    comm_type, comm_ranks, tensor_dtype, tensor_shape)
            single_comm_time = 0
            for rank in comm_ranks:
                single_comm_time += single_comm_time_per_rank[rank]
            single_comm_time /= len(comm_ranks)
            if is_mp:
                comm_times.mp += single_comm_time
            elif is_dp:
                comm_times.dp += single_comm_time
            else:
                raise NotImplementedError
        DistributedWrapperContext.patch_dist_func(self.world_size)
        return comm_times

    def _get_mp_forward_backward_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        if (mp, mb) not in self.mp_forward_backward_comm_logs:
            self._set_forward_backward_time_and_memory()

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].pre_process
        pre_process_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].single_layer
        single_layer_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_forward_backward_comm_logs[(mp, mb)].post_process
        post_process_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        mp_forward_backward_times = TimeOrMemory(
                max(0, pre_process_comm_time.mp - single_layer_comm_time.mp),
                single_layer_comm_time.mp,
                max(0, post_process_comm_time.mp - single_layer_comm_time.mp))
        return mp_forward_backward_times

    def _get_pp_comm_time_per_stage(self):
        DistributedWrapperContext.unpatch_dist_func()
        args = get_args()
        pp = args.pipeline_model_parallel_size

        pp_warmup_comm_time_per_stage = {0:0}
        pp_steady_comm_time_per_stage = {0:0}
        pp_embedding_sync_time = 0
        if pp == 1:
            return pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, \
                    pp_embedding_sync_time

        stage_ranks = {} # (mp, dp) -> list of ranks
        for mp in range(args.tensor_model_parallel_size):
            for dp in range(args.data_parallel_size):
                stage_ranks[(mp, dp)] = [-1] * pp
        for task, rank in self.curr_task_to_rank.items():
            stage_ranks[(task.mp, task.dp)][task.pp] = rank

        assert pp % 2 == 0
        all_send_recv_ranks = {}
        send_recv_except_first_and_last_ranks = {}
        for task, rank in self.curr_task_to_rank.items():
            key = (task.mp, task.dp)
            stage = task.pp
            if task.pp % 2 == 0:
                all_send_recv_ranks[rank] = [rank, stage_ranks[key][task.pp + 1]]
                if stage != 0:
                    send_recv_except_first_and_last_ranks[rank] = \
                            [stage_ranks[key][task.pp - 1], rank]
            else:
                all_send_recv_ranks[rank] = [stage_ranks[key][task.pp - 1], rank]
                if stage != pp - 1:
                    send_recv_except_first_and_last_ranks[rank] = \
                            [rank, stage_ranks[key][task.pp + 1]]

        tensor_dtype = torch.float16 if args.fp16 else torch.float32
        tensor_shape = [args.seq_length, args.micro_batch_size, args.hidden_size]
        for stage in range(pp):
            stage_rank = stage_ranks[(0, 0)][stage]

            # warmup_forward: recv_forward
            if stage == 0:
                recv_forward = 0
            else:
                recv_forward_rank_to_comm_ranks = {}
                for (mp, dp), stage_rank_list in stage_ranks.items():
                    src_rank = stage_rank_list[stage - 1]
                    dst_rank = stage_rank_list[stage]
                    recv_forward_rank_to_comm_ranks[src_rank] = [src_rank, dst_rank]
                    recv_forward_rank_to_comm_ranks[dst_rank] = [src_rank, dst_rank]
                recv_forward = self.comm_helper.request_comm(
                        CommType.SEND_OR_RECV, recv_forward_rank_to_comm_ranks,
                        tensor_dtype, tensor_shape)[stage_rank]
            
            # warmup_backward: recv_backward
            if stage == pp - 1:
                recv_backward = 0
            else:
                recv_backward_rank_to_comm_ranks = {}
                for (mp, dp), stage_rank_list in stage_ranks.items():
                    src_rank = stage_rank_list[stage + 1]
                    dst_rank = stage_rank_list[stage]
                    recv_backward_rank_to_comm_ranks[src_rank] = [src_rank, dst_rank]
                    recv_backward_rank_to_comm_ranks[dst_rank] = [src_rank, dst_rank]
                recv_backward = self.comm_helper.request_comm(
                        CommType.SEND_OR_RECV, recv_backward_rank_to_comm_ranks,
                        tensor_dtype, tensor_shape)[stage_rank]
            pp_warmup_comm_time_per_stage[stage] = recv_forward + recv_backward

            # steady: all_send_recv_ranks + send_recv_except_first_and_last_ranks
            all_send_recv_time = self.comm_helper.request_comm(
                    CommType.SEND_AND_RECV,
                    all_send_recv_ranks,
                    tensor_dtype, tensor_shape)[stage_rank]
            if stage != 0 and stage != pp - 1:
                send_recv_except_first_and_last_time = \
                        self.comm_helper.request_comm(
                                CommType.SEND_AND_RECV,
                                send_recv_except_first_and_last_ranks,
                                tensor_dtype, tensor_shape)[stage_rank]
            else:
                send_recv_except_first_and_last_time = 0

            pp_steady_comm_time_per_stage[stage] = \
                    all_send_recv_time + send_recv_except_first_and_last_time

        if pp > 1:
            sync_embedding_rank_to_comm_ranks = {}
            for (mp, dp), stage_rank_list in stage_ranks.items():
                sync_embedding_rank_to_comm_ranks[stage_rank_list[0]] = \
                        [stage_rank_list[0], stage_rank_list[-1]]
                sync_embedding_rank_to_comm_ranks[stage_rank_list[-1]] = \
                        [stage_rank_list[0], stage_rank_list[-1]]
            embedding_shape = self.embedding_shape
            pp_embedding_sync_time = self.comm_helper.request_comm(
                    CommType.ALLREDUCE,
                    sync_embedding_rank_to_comm_ranks,
                    torch.float16 if args.fp16 else torch.float32,
                    embedding_shape)[stage_ranks[(0,0)][0]]

        DistributedWrapperContext.patch_dist_func(self.world_size) 
        return pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, pp_embedding_sync_time

    def _set_comm_logs(self, comm_logs, pre_process=False, post_process=False):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        for comm_group, comms_to_measure in comm_logs.items():
            assert comm_group != 'model_parallel_group', (comms_to_measure)
            is_mp = comm_group == 'tensor_model_parallel_group'
            is_dp = comm_group == 'data_parallel_group'
            is_pp = comm_group == 'pipeline_model_parallel_group'
            if is_mp:
                self.mp_forward_backward_comm_logs[(mp, mb)].set(comms_to_measure, pre_process, post_process)
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
            pre_process_forward_backward_time, \
                    activation_shape, _, pre_process_peak_memory, pre_process_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, pre_process=True)
            self._set_comm_logs(pre_process_comm_logs, pre_process=True)

            single_layer_forward_backward_time, \
                    activation_shape, activation_size, single_layer_peak_memory, \
                    single_layer_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, input_tensor_shape=activation_shape)
            self._set_comm_logs(single_layer_comm_logs)

            post_process_forward_backward_time, \
                    _, _, post_process_peak_memory, post_process_comm_logs = \
                do_forward_backward(DistributedWrapperContext._COMM_CALLED,
                                    self.forward_step_func, models, optimizers,
                                    train_data_iterator, post_process=True,
                                    input_tensor_shape=activation_shape)
            self._set_comm_logs(post_process_comm_logs, post_process=True)

        pre_process_forward_backward_time -= single_layer_forward_backward_time
        post_process_forward_backward_time -= single_layer_forward_backward_time
        self.forward_backward_times[key] = TimeOrMemory(pre_process_forward_backward_time,
                                               single_layer_forward_backward_time,
                                               post_process_forward_backward_time)

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
        if key not in self.forward_backward_times:
            self._set_forward_backward_time_and_memory()
        return self.forward_backward_times[key]

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
            forward_backward_times = self._get_compute_times()
            optimizer_times = self._get_optimizer_times()
            mp_forward_backward_times = self._get_mp_forward_backward_times()
            pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, \
                    pp_embedding_sync_time = self._get_pp_comm_time_per_stage()
            iter_time = get_iter_time_estimation(
                    forward_backward_times, optimizer_times,
                    mp_forward_backward_times, pp_warmup_comm_time_per_stage,
                    pp_steady_comm_time_per_stage, pp_embedding_sync_time)
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

            req_gpu_memory = get_required_gpu_memory(param_sizes, grad_sizes, activation_sizes, peak_memories)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                tuner_logger.info(f"OOM for {config}")
                return 0
            raise e
        return req_gpu_memory
