"""Tuning throught for given model and resources"""
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, List

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

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

@dataclass
class TimeOrMemory:
    pre_process: float
    single_layer: float
    post_process: float

def dummy_func_with_return(return_value=None):
    def dummy_func(*args, **kwargs):
        return return_value
    return dummy_func

def dummy_func_with_first_input():
    def dummy_func(*args, **kwargs):
        return args[0]
    return dummy_func

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

@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass

def get_forward_step_time(forward_step_func, train_data_iterator, model, input_tensor, compute_loss = False):
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    torch.cuda.synchronize()
    s = time.time()
    output, loss_func = forward_step_func(train_data_iterator, model)
    if compute_loss:
        output = loss_func(output)
        loss, loss_reduced = output
        output = loss
    torch.cuda.synchronize()
    e = time.time()
    return output, e - s

def get_backward_step_time(optimizer, input_tensor, output_tensor, output_tensor_grad):
    args = get_args()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    torch.cuda.synchronize()
    s = time.time()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    torch.cuda.synchronize()
    e = time.time()

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad, e - s

def get_optimizer_time(model: torch.nn.Module, optimizer: MegatronOptimizer):
    for param in model.parameters():
        if param.requires_grad:
            if optimizer.params_have_main_grad:
                param.main_grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()
            else:
                param.grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()
    torch.cuda.synchronize()
    s = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    e = time.time()
    return e - s

def do_forward_backward(forward_step_func, models, optimizers, train_data_iterator,
                        pre_process=False, post_process=False, input_tensor_shape=None):
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

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    # do forward
    output, forward_time = get_forward_step_time(
            forward_step_func,
            train_data_iterator,
            model,
            input_tensor,
            compute_loss=post_process)

    activation_shape = output.size()
    activation_size = output.nelement() * output.element_size()

    # do backward
    if post_process:
        output_tensor_grad = None
    else:
        output_tensor_grad = torch.randn(list(activation_shape)).cuda()
        output_tensor_grad.requires_grad = True

    _, backward_time = get_backward_step_time(
            optimizer,
            input_tensor,
            output,
            output_tensor_grad)
    peak_memory = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    #print(torch.cuda.memory_summary())

    return forward_time, backward_time, activation_shape, activation_size, peak_memory

def get_model_stat(forward_step_func, models, optimizers, train_data_iterator):
    """Get kernel times, activation_size, and peak memory for single layer"""
    args = get_args()
    global_batch_size = args.global_batch_size
    args.global_batch_size = 1

    # prevent ddp comm
    context_handler = dummy_handler
    if isinstance(models.model_with_pre_process, torchDDP):
        context_handler = models.model_with_pre_process.no_sync

    activation_shape = None
    activation_size = None
    with context_handler():
        for _ in range(2):
            pre_process_forward_time, pre_process_backward_time, \
                    activation_shape, activation_size, pre_process_peak_memory = \
                do_forward_backward(forward_step_func, models, optimizers, train_data_iterator,
                        pre_process=True, post_process=False, input_tensor_shape=None)

            single_layer_forward_time, single_layer_backward_time, \
                    activation_shape, activation_size, single_layer_peak_memory = \
                do_forward_backward(forward_step_func, models, optimizers, train_data_iterator,
                        pre_process=False, post_process=False, input_tensor_shape=activation_shape)

            post_process_forward_time, post_process_backward_time, \
                    _, _, post_process_peak_memory = \
                do_forward_backward(forward_step_func, models, optimizers, train_data_iterator,
                        pre_process=False, post_process=True, input_tensor_shape=activation_shape)

    pre_process_forward_time -= single_layer_forward_time
    post_process_forward_time -= single_layer_forward_time
    forward_times = TimeOrMemory(pre_process_forward_time,
                                 single_layer_forward_time,
                                 post_process_forward_time)

    pre_process_backward_time -= single_layer_backward_time
    post_process_backward_time -= single_layer_backward_time
    backward_times = TimeOrMemory(pre_process_backward_time,
                                  single_layer_backward_time,
                                  post_process_backward_time)

    peak_memories = TimeOrMemory(pre_process_peak_memory,
                                 single_layer_peak_memory,
                                 post_process_peak_memory)

    args.global_batch_size = global_batch_size
    return forward_times, backward_times, activation_size, peak_memories

def get_optimizer_times(models, optimizers):

    model = models.model_with_pre_process[0]
    optimizer = optimizers.optimizer_with_pre_process
    pre_process_optimizer_time = get_optimizer_time(model, optimizer)

    model = models.model_with_post_process[0]
    optimizer = optimizers.optimizer_with_post_process
    post_process_optimizer_time = get_optimizer_time(model, optimizer)

    model = models.model_without_pre_or_post_process[0]
    optimizer = optimizers.optimizer_without_pre_or_post_process
    single_layer_optimizer_time = get_optimizer_time(model, optimizer)

    post_process_optimizer_time -= single_layer_optimizer_time
    pre_process_optimizer_time -= single_layer_optimizer_time

    optimizer_times = TimeOrMemory(pre_process_optimizer_time,
                                   single_layer_optimizer_time,
                                   max(0, post_process_optimizer_time))  # post_process 
    return optimizer_times

def get_iter_time_estimation(forward_times, backward_times, optimizer_times):
    """Get iter time estimation as milliseconds"""

    args = get_args()
    #TODO(SJ): consider parallelism
    assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == args.data_parallel_size == 1
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size
    kernel_forward_time = forward_times.pre_process
    kernel_forward_time += forward_times.single_layer * num_layers
    kernel_forward_time += forward_times.post_process

    kernel_backward_time = backward_times.pre_process
    kernel_backward_time += backward_times.single_layer * num_layers
    kernel_backward_time += backward_times.post_process

    optimizer_time = optimizer_times.pre_process
    optimizer_time += optimizer_times.single_layer * num_layers
    optimizer_time += optimizer_times.post_process

    print('kernel forward time', int(kernel_forward_time * num_micro_batches * 1000),
          'kernel_backward_time', int(kernel_backward_time * num_micro_batches * 1000),
          'optimizer_time', int(optimizer_time * 1000))
    return ((kernel_forward_time + kernel_backward_time) * num_micro_batches + optimizer_time) * 1000

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
        #print(params)
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
    assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == args.data_parallel_size == 1

    param_size = (param_sizes.pre_process + param_sizes.single_layer * num_layers + param_sizes.post_process) / 1024 / 1024
    peak_memory = max(peak_memories.pre_process, peak_memories.single_layer, peak_memories.post_process) / 1024 / 1024
    checkpoint_size = (activation_size * num_layers // args.activations_checkpoint_num_layers) / 1024 / 1024
    if num_micro_batches > 1 or args.accumulate_allreduce_grads_in_fp32:
        grad_size = (grad_sizes.pre_process + grad_sizes.single_layer * num_layers + grad_sizes.post_process) / 1024 / 1024
    else:
        grad_size = max(0, grad_sizes.single_layer - activation_size / args.activations_checkpoint_num_layers)
        grad_size *= num_layers / 1024 / 1024
    print("param_size", param_size, "peak_memory", peak_memory, "checkpoint_size", checkpoint_size, "grad_size", grad_size)
    return param_size + peak_memory + checkpoint_size + grad_size

@dataclass
class Config:
    micro_batch_size: int
    global_batch_size: int
    dp = 1
    mp = 1
    pp = 1
    parallelism_order=None

class DistributedWrapperContext:
    _DEFAULT_INIT = dist.init_process_group
    _DEFAULT_IS_INIT = dist.is_initialized
    _DEFAULT_NEW_GROUP = dist.new_group
    _DEFAULT_WORLD_SIZE = dist.get_world_size
    _DEFAULT_RANK = dist.get_rank
    _DEFAULT_BARRIER = dist.barrier
    _DEFAULT_ALLREDUCE = dist.all_reduce
    _DEFAULT_BROADCAST = dist.broadcast

    @staticmethod
    def patch_dist_func(world_size):
        setattr(dist, 'init_process_group', dummy_func_with_return())
        setattr(dist, 'is_initialized', dummy_func_with_return(True))
        setattr(dist, 'new_group', dummy_func_with_return("dummy_group"))
        setattr(dist, 'get_world_size', dummy_func_with_return(world_size))
        setattr(dist, 'get_rank', dummy_func_with_return(0))
        setattr(dist, 'barrier', dummy_func_with_return())
        setattr(dist, 'all_reduce', dummy_func_with_first_input())
        setattr(dist, 'broadcast', dummy_func_with_first_input())

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
    def __init__(self, world_size, model='gpt'):
        self.world_size = world_size
        self.model = model

        self.forward_times = {}  # mp, mb -> single layer forward time
        self.backward_times = {}
        self.optimizer_times = {}  # mp -> single_layer_optimizer_time
        self.param_sizes = {} # mp -> single layer parameter size including optimizer states
        self.grad_sizes = {}  # mp -> single layer gradient size

        self.activation_size = {}  # (mp, mb) -> activation size
        self.peak_memories = {}  # (mp, mb) -> peak memory

    def __enter__(self):
        DistributedWrapperContext.patch_dist_func(self.world_size)
        if self.model == 'gpt':
            self.models, self.optimizers = get_single_layer_model(
                    gpt_model_provider,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
            self.train_ds = get_train_dataset(dataset='gpt')
            self.forward_step_func = gpt_forward_step
        else:
            raise NotImplementedError
        return self

    def __exit__(self, type, value, traceback):
        DistributedWrapperContext.unpatch_dist_func()

    def _get_optimizer_times(self):
        mp = get_args().tensor_model_parallel_size
        if mp in self.optimizer_times:
            return self.optimizer_times[mp]

        optimizer_times = get_optimizer_times(self.models, self.optimizers)
        self.optimizer_times[mp] = optimizer_times
        return optimizer_times

    def _set_model_stat(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size
        key = (mp, mb)

        train_data_iterator = get_train_data_iterator(self.train_ds)
        forward_times, backward_times, activation_size, peak_memories = \
                get_model_stat(self.forward_step_func,
                               self.models,
                               self.optimizers,
                               train_data_iterator)
        self.forward_times[key] = forward_times
        self.backward_times[key] = backward_times
        self.activation_size[key] = activation_size
        self.peak_memories[key] = peak_memories

    def _get_compute_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.forward_times:
            self._set_model_stat()
        return self.forward_times[key], self.backward_times[key]

    def _get_param_and_grad_sizes(self):
        mp = get_args().tensor_model_parallel_size
        if mp in self.param_sizes:
            return self.param_sizes[mp], self.grad_sizes[mp]

        # optimizer must be called at least once to get optimizer states
        self._get_optimizer_times()
        
        param_sizes, grad_sizes = get_param_and_grad_sizes(self.models, self.optimizers)
        self.param_sizes[mp] = param_sizes
        self.grad_sizes[mp] = grad_sizes
        return param_sizes, grad_sizes

    def _get_activation_size(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.activation_size:
            self._set_model_stat()
        return self.activation_size[key]

    def _get_peak_memories(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        key = (mp, mb)
        if key not in self.peak_memories:
            self._set_model_stat()

        return self.peak_memories[key]

    def get_iter_time(self, config):
        print("get iter time for", config)

        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        forward_times, backward_times = self._get_compute_times()
        optimizer_times = self._get_optimizer_times()

        iter_time = get_iter_time_estimation(forward_times, backward_times, optimizer_times)
        return iter_time

    def get_max_gpu_memory(self, config):
        print("get max gpu memory for", config)

        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        param_sizes, grad_sizes = self._get_param_and_grad_sizes()
        activation_sizes = self._get_activation_size()
        peak_memories = self._get_peak_memories()

        req_gpu_memory = get_required_gpu_memory(param_sizes, grad_sizes, activation_sizes, peak_memories)
        return req_gpu_memory
