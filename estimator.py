"""Tuning throught for given model and resources"""
from contextlib import contextmanager
from dataclasses import dataclass
import time

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
from megatron.schedules import backward_step
from megatron.training import cyclic_iter, setup_model_and_optimizer
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.language_model import Embedding

def get_single_layer_model(model_provider, args_defaults=None):
    initialize_megatron(args_defaults=args_defaults)

    args = get_args()
    # disable save and load checkpoints
    args.load = None
    args.save = None

    # Get single transformer layer model
    original_num_layers = args.num_layers
    
    models = {}
    optimizers = {}

    args.num_layers = 1
    def model_provider_with_pre_process(pre_process=True, post_process=True):
        return model_provider(pre_process=True, post_process=False)
    model_with_pre_process, optimizer_with_pre_process, _ = setup_model_and_optimizer(
            model_provider_with_pre_process)
    models['model_with_pre_process'] = model_with_pre_process
    optimizers['optimizer_with_pre_process'] = optimizer_with_pre_process

    def model_provider_without_pre_or_post_process(pre_process=True, post_process=True):
        return model_provider(pre_process=False, post_process=False)
    model, optimizer, _ = setup_model_and_optimizer(
            model_provider_without_pre_or_post_process)
    models['model_without_pre_or_post_process'] = model
    optimizers['optimizer_without_pre_or_post_process'] = optimizer

    def model_provider_with_post_process(pre_process=True, post_process=True):
        return model_provider(pre_process=False, post_process=True)
    model_with_post_process, optimizer_with_post_process, _ = setup_model_and_optimizer(
            model_provider_with_post_process)
    unwrapped_model_with_post_process = unwrap_model(
        model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0].language_model
    orig_embedding = unwrap_model(
        model_with_pre_process, (torchDDP, LocalDDP, Float16Module))[0].language_model.embedding
    # set embedding for loss computation
    unwrapped_model_with_post_process.embedding = orig_embedding
    models['model_with_post_process'] = model_with_post_process
    optimizers['optimizer_with_post_process'] = optimizer_with_post_process

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

def get_optimizer_time(optimizer):
    torch.cuda.synchronize()
    s = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    e = time.time()
    return e - s

def get_single_layer_iter_time(forward_step_func, models, optimizers, train_data_iterator):
    args = get_args()

    # prevent ddp comm
    context_handler = dummy_handler
    if isinstance(models['model_with_pre_process'], torchDDP):
        context_handler = models['model_with_pre_process'].no_sync

    input_tensors = {}
    output_tensors = {}
    output_tensor_grads = {}

    input_tensors['model_with_pre_process'] = None
    output_tensor_grads['model_with_post_process'] = None
    with context_handler():
        for _ in range(2):
            # do forward
            output, pre_process_forward_time = get_forward_step_time(
                    forward_step_func,
                    train_data_iterator,
                    models['model_with_pre_process'][0],
                    input_tensors['model_with_pre_process'])
            output_tensors['model_with_pre_process'] = output
            input_tensors['model_without_pre_or_post_process'] = output.clone().detach()
            input_tensors['model_without_pre_or_post_process'].requires_grad = True

            output, single_layer_forward_time = get_forward_step_time(
                    forward_step_func,
                    train_data_iterator,
                    models['model_without_pre_or_post_process'][0],
                    input_tensors['model_without_pre_or_post_process'])
            output_tensors['model_without_pre_or_post_process'] = output
            input_tensors['model_with_post_process'] = output.clone().detach()
            input_tensors['model_with_post_process'].requires_grad = True

            output, post_process_forward_time = get_forward_step_time(
                    forward_step_func,
                    train_data_iterator,
                    models['model_with_post_process'][0],
                    input_tensors['model_with_post_process'],
                    compute_loss=True)
            output_tensors['model_with_post_process'] = output

            optimizer = optimizers['optimizer_with_post_process']
            input_tensor_grad, post_process_backward_time = get_backward_step_time(
                    optimizer,
                    input_tensors['model_with_post_process'],
                    output_tensors['model_with_post_process'],
                    output_tensor_grads['model_with_post_process'])
            post_process_optimizer_time = get_optimizer_time(optimizer)

            output_tensor_grads['model_without_pre_or_post_process'] = \
                    torch.randn(list(input_tensor_grad.size())).cuda()
            optimizer = optimizers['optimizer_without_pre_or_post_process']
            input_tensor_grad, single_layer_backward_time = get_backward_step_time(
                    optimizer,
                    input_tensors['model_without_pre_or_post_process'],
                    output_tensors['model_without_pre_or_post_process'],
                    output_tensor_grads['model_without_pre_or_post_process'])
            single_layer_optimizer_time = get_optimizer_time(optimizer)

            output_tensor_grads['model_with_pre_process'] = \
                    torch.randn(list(input_tensor_grad.size())).cuda()
            optimizer = optimizers['optimizer_with_pre_process']
            input_tensor_grad, pre_process_backward_time = get_backward_step_time(
                    optimizer,
                    input_tensors['model_with_pre_process'],
                    output_tensors['model_with_pre_process'],
                    output_tensor_grads['model_with_pre_process'])
            pre_process_optimizer_time = get_optimizer_time(optimizer)

    pre_process_forward_time -= single_layer_forward_time
    post_process_forward_time -= single_layer_forward_time
    pre_process_backward_time -= single_layer_backward_time
    post_process_backward_time -= single_layer_backward_time

    post_process_optimizer_time -= single_layer_optimizer_time
    pre_process_optimizer_time -= single_layer_optimizer_time

    comp_times = {}
    comp_times['pre_process_forward_time'] = pre_process_forward_time
    comp_times['pre_process_backward_time'] = pre_process_backward_time
    comp_times['single_layer_forward_time'] = single_layer_forward_time
    comp_times['single_layer_backward_time'] = single_layer_backward_time
    comp_times['post_process_forward_time'] = post_process_forward_time
    comp_times['post_process_backward_time'] = post_process_backward_time

    optimizer_times = {}
    optimizer_times['pre_process_optimizer_time'] = pre_process_optimizer_time
    optimizer_times['single_layer_optimizer_time'] = single_layer_optimizer_time
    optimizer_times['post_process_optimizer_time'] = post_process_optimizer_time
    return comp_times, optimizer_times

def get_iter_time_estimation(comp_times, optimizer_times):
    """Get iter time estimation as milliseconds"""

    args = get_args()
    #TODO(SJ): consider parallelism
    assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == args.data_parallel_size == 1
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size
    kernel_forward_time = comp_times['pre_process_forward_time']
    kernel_forward_time += comp_times['single_layer_forward_time'] * num_layers
    kernel_forward_time += comp_times['post_process_forward_time']

    kernel_backward_time = comp_times['pre_process_backward_time']
    kernel_backward_time += comp_times['single_layer_backward_time'] * num_layers
    kernel_backward_time += comp_times['post_process_backward_time']

    optimizer_time = optimizer_times['pre_process_optimizer_time']
    optimizer_time += optimizer_times['single_layer_optimizer_time'] * num_layers
    optimizer_time += optimizer_times['post_process_optimizer_time']

    print('kernel forward time', int(kernel_forward_time * num_micro_batches * 1000),
          'kernel_backward_time', int(kernel_backward_time * num_micro_batches * 1000),
          'optimizer_time', int(optimizer_time * 1000))
    return ((kernel_forward_time + kernel_backward_time) * num_micro_batches + optimizer_time) * 1000

@dataclass
class Config:
    micro_batch_size: int
    global_batch_size: int
    dp = 1
    mp = 1
    pp = 1
    parallelism_order=None

def dummy_func_with_return(return_value=None):
    def dummy_func(*args, **kwargs):
        return return_value
    return dummy_func

def dummy_func_with_first_input():
    def dummy_func(*args, **kwargs):
        return args[0]
    return dummy_func

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

    def __enter__(self):
        DistributedWrapperContext.patch_dist_func(self.world_size)
        if self.model == 'gpt':
            self.models, self.optimizers = get_single_layer_model(
                    gpt_model_provider,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
            self.train_ds = get_train_dataset(dataset='gpt')
        else:
            raise NotImplementedError
        return self

    def __exit__(self, type, value, traceback):
        DistributedWrapperContext.unpatch_dist_func()

    def get_iter_time(self, config):
        print("get iter time for", config)
        args = get_args()

        # apply config to arguments
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        train_data_iterator = get_train_data_iterator(self.train_ds)
        comp_times, optimizer_times = get_single_layer_iter_time(
                gpt_forward_step, self.models, self.optimizers, train_data_iterator)
        iter_time = get_iter_time_estimation(comp_times, optimizer_times)
        return iter_time

if __name__ == "__main__":
    print(sys.argv)
