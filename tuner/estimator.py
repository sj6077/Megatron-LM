""" Tuning throught for given model and resources """
# pylint: disable=too-many-lines, logging-fstring-interpolation
from collections import defaultdict
from dataclasses import dataclass, field
import logging
import multiprocessing
import os
import json
import time
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._C._distributed_c10d import ReduceOp

from pretrain_gpt import model_provider as gpt_model_provider
from pretrain_gpt import forward_step as gpt_forward_step

from megatron import get_args
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.initialize import initialize_megatron
from megatron.optimizer.optimizer import MegatronOptimizer
from megatron.training import cyclic_iter, setup_model_and_optimizer
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from tuner.comm_helper import CommType, CommHelper

tuner_logger = logging.getLogger('tuner')
torch_cuda_synchronize = torch.cuda.synchronize

NUM_AVERAGE = 20

@dataclass
class Models:
    """Single layer models including pre or post process"""
    model_with_pre_process: List[torch.nn.Module]
    model_without_pre_or_post_process: List[torch.nn.Module]
    model_with_post_process: List[torch.nn.Module]

@dataclass
class Optimizers:
    """Single layer optimizers including pre or post process"""
    optimizer_with_pre_process: MegatronOptimizer
    optimizer_without_pre_or_post_process: MegatronOptimizer
    optimizer_with_post_process: MegatronOptimizer

# pylint: disable=invalid-name
@dataclass(frozen=True)
class Task:
    """Task identifier for 3D parallelism"""
    dp: int
    mp: int
    pp: int

@dataclass
class CommTime:
    """Communication time for each parallelism"""
    dp: float = 0.0
    mp: float = 0.0
    pp: float = 0.0

@dataclass
class TimeOrMemory:
    """Time or memory for single layer including pre or post process"""
    pre_process: float
    single_layer: float
    post_process: float

    def get_total(self, num_layers):
        """Get total time or memory including both pre and post process"""
        return self.pre_process + self.single_layer * num_layers + self.post_process

    def get_max(self, num_layers):
        """Get total time or memory including max value between pre or post process"""
        return max(self.pre_process, self.post_process) + self.single_layer * num_layers

@dataclass
class Log:
    """Communication logs"""
    pre_process: list = field(default_factory=list)
    single_layer: list = field(default_factory=list)
    post_process: list = field(default_factory=list)

    def set(self, value, is_pre_process, is_post_process):
        """Set the log value"""
        if is_pre_process:
            self.pre_process = value
        elif is_post_process:
            self.post_process = value
        else:
            self.single_layer = value

@dataclass(frozen=True)
class Config:
    """Configuration for parallelization method"""
    micro_batch_size: int
    global_batch_size: int
    dp: int = 1
    mp: int = 1
    pp: int = 1

class DistributedWrapperContext:
    """Mock torch distributed APIs"""
    _DEFAULT_INIT = dist.init_process_group
    _DEFAULT_IS_INIT = dist.is_initialized
    _DEFAULT_NEW_GROUP = dist.new_group
    _DEFAULT_WORLD_SIZE = dist.get_world_size
    _DEFAULT_RANK = dist.get_rank
    _DEFAULT_BARRIER = dist.barrier
    _DEFAULT_ALLREDUCE = dist.all_reduce
    _DEFAULT_BROADCAST = dist.broadcast
    _DUMMY_GROUPS = {}
    _BEFORE_MODEL_START = defaultdict(list) # comm_group -> comm_type, tensor_dtype, tensor_shape
    _AFTER_MODEL_END = defaultdict(list) # comm_group -> comm_type, tensor_dtype, tensor_shape
    _WITHIN_MODEL = defaultdict(list) # comm_group -> comm_type, tensor_dtype, tensor_shape
    _START_RECORD_COMM = False
    IS_MODEL_STARTED = False
    IS_MODEL_ENDED = False
    CURR_CONFIG = None
    START_TIME = None

    @staticmethod
    def record_comm_logs(after_forward_backward=False):
        """Record communication logs for torch.distributed"""
        DistributedWrapperContext._START_RECORD_COMM = True
        if not after_forward_backward:
            DistributedWrapperContext.IS_MODEL_STARTED = False
            DistributedWrapperContext.IS_MODEL_ENDED = False
        else:
            DistributedWrapperContext.IS_MODEL_STARTED = True
            DistributedWrapperContext.IS_MODEL_ENDED = True

    @staticmethod
    def new_group(*args, **kwargs): # pylint: disable=unused-argument
        """Create dummy group"""
        ranks = args[0]
        assert ranks is not None

        group_name = f"dummy_group{len(DistributedWrapperContext._DUMMY_GROUPS)}"
        DistributedWrapperContext._DUMMY_GROUPS[group_name] = len(ranks)
        return group_name

    @staticmethod
    def convert_group(group):
        """Convert dummy group to the parallelization group"""
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

    @staticmethod
    def get_world_size(world_size):
        """Get world size according to the current configuration"""
        def _get_world_size(group=None):
            if group is None:
                return world_size
            new_group = DistributedWrapperContext.convert_group(group)
            curr_config = DistributedWrapperContext.CURR_CONFIG
            if curr_config is None:
                return DistributedWrapperContext._DUMMY_GROUPS[group]

            if new_group == "model_parallel_group":
                group_world_size = curr_config.pp * curr_config.mp
            elif new_group == "data_parallel_group":
                group_world_size = curr_config.dp
            elif new_group == "embedding_group":
                group_world_size = 0 if curr_config.pp == 1 else 2
            elif new_group == "tensor_model_parallel_group":
                group_world_size = curr_config.mp
            else:
                assert new_group == "pipeline_model_parallel_group"
                group_world_size = curr_config.pp
            return group_world_size
        return _get_world_size

    @staticmethod
    def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False): # pylint: disable=unused-argument
        """Mock all_reduce"""
        if not DistributedWrapperContext._START_RECORD_COMM:
            return

        if not DistributedWrapperContext.IS_MODEL_STARTED:
            comm_dict = DistributedWrapperContext._BEFORE_MODEL_START
        elif DistributedWrapperContext.IS_MODEL_ENDED:
            comm_dict = DistributedWrapperContext._AFTER_MODEL_END
        else:
            comm_dict = DistributedWrapperContext._WITHIN_MODEL

        comm_group = DistributedWrapperContext.convert_group(group)
        comm_dict[comm_group].append((
            CommType.ALLREDUCE, tensor.dtype, list(tensor.size())))

    @staticmethod
    def broadcast(tensor, src, group=None, async_op=False):  # pylint: disable=unused-argument
        """Mock broadcast"""
        if not DistributedWrapperContext._START_RECORD_COMM:
            return

        if not DistributedWrapperContext.IS_MODEL_STARTED:
            comm_dict = DistributedWrapperContext._BEFORE_MODEL_START
        elif DistributedWrapperContext.IS_MODEL_ENDED:
            comm_dict = DistributedWrapperContext._AFTER_MODEL_END
        else:
            comm_dict = DistributedWrapperContext._WITHIN_MODEL
        comm_group = DistributedWrapperContext.convert_group(group)
        comm_dict[comm_group].append((
            CommType.BROADCAST, tensor.dtype, list(tensor.size())))

    @staticmethod
    def get_comm_logs(pre_process=False, post_process=False, after_forward_backward=False):
        """Get collected communication logs"""
        assert DistributedWrapperContext._START_RECORD_COMM
        tuner_logger.debug(f"curr_config: {DistributedWrapperContext.CURR_CONFIG}")
        tuner_logger.debug(f"before model start comm logs: {DistributedWrapperContext._BEFORE_MODEL_START}")  # pylint: disable=line-too-long
        tuner_logger.debug(f"within model comm logs: {DistributedWrapperContext._WITHIN_MODEL}")
        tuner_logger.debug(f"after model end comm logs: {DistributedWrapperContext._AFTER_MODEL_END}")  # pylint: disable=line-too-long

        comm_logs = {}
        if after_forward_backward or post_process:
            comm_logs = DistributedWrapperContext._AFTER_MODEL_END.copy()
        elif pre_process:
            comm_logs = DistributedWrapperContext._BEFORE_MODEL_START.copy()
        for comm_group, comm_log_list in DistributedWrapperContext._WITHIN_MODEL.items():
            if comm_group not in comm_logs:
                comm_logs[comm_group] = comm_log_list.copy()
            else:
                comm_logs[comm_group] += comm_log_list
        DistributedWrapperContext.reset_comm_logs()
        return comm_logs

    @staticmethod
    def reset_comm_logs():  # pylint: disable=no-method-argument
        """Clear all collected communication logs"""
        DistributedWrapperContext._BEFORE_MODEL_START.clear()
        DistributedWrapperContext._AFTER_MODEL_END.clear()
        DistributedWrapperContext._WITHIN_MODEL.clear()

        DistributedWrapperContext._START_RECORD_COMM = False

    @staticmethod
    def dummy_func_with_return(return_value=None):
        """dummy function that has a fixed return value"""
        def dummy_func(*args, **kwargs):  # pylint:disable=unused-argument
            return return_value
        return dummy_func

    @staticmethod
    def patch_dist_func(world_size):
        """Mock torch.distributed APIs"""
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
        """Revert torch.distributed APIs to orignal functions"""
        setattr(dist, 'init_process_group', DistributedWrapperContext._DEFAULT_INIT)
        setattr(dist, 'is_initialized', DistributedWrapperContext._DEFAULT_IS_INIT)
        setattr(dist, 'new_group', DistributedWrapperContext._DEFAULT_NEW_GROUP)
        setattr(dist, 'get_world_size', DistributedWrapperContext._DEFAULT_WORLD_SIZE)
        setattr(dist, 'get_rank', DistributedWrapperContext._DEFAULT_RANK)
        setattr(dist, 'barrier', DistributedWrapperContext._DEFAULT_BARRIER)
        setattr(dist, 'all_reduce', DistributedWrapperContext._DEFAULT_ALLREDUCE)
        setattr(dist, 'broadcast', DistributedWrapperContext._DEFAULT_BROADCAST)
        setattr(torch.cuda, 'synchronize', torch_cuda_synchronize)

def forward_hook(module, forward_input):  # pylint: disable=unused-argument
    """Set model forward is started"""
    DistributedWrapperContext.IS_MODEL_STARTED = True
    DistributedWrapperContext.START_TIME = time.time()

def backward_hook(module, grad_input, grad_output):  # pylint: disable=unused-argument
    """Set model backward is ended"""
    DistributedWrapperContext.IS_MODEL_ENDED = True

def get_post_process_device(num_gpus_per_node):
    """Get device for single layer model including post process"""
    if num_gpus_per_node > 2:
        return 2
    return 0

def get_single_layer_device(num_gpus_per_node):
    """Get device for single layer model"""
    if num_gpus_per_node == 1:
        return 0
    return 1

# pylint: disable=too-many-locals
def get_single_layer_model(model_provider, num_gpus_per_node):
    """Create models and optimizers for pre+single_layer, single_layer, and post+single_layer"""
    tuner_logger.debug(f"get_single_layer_model for num_gpus_per_node: {num_gpus_per_node}")

    args = get_args()

    pre_process_device = 0
    torch.cuda.set_device(pre_process_device)
    def model_provider_with_pre_process(pre_process=True, post_process=True):  # pylint: disable=unused-argument
        new_model = model_provider(pre_process=True, post_process=False)
        return new_model
    model_with_pre_process, optimizer_with_pre_process, _ = setup_model_and_optimizer(
        model_provider_with_pre_process)
    model_with_pre_process[0].register_forward_pre_hook(forward_hook)
    model_with_pre_process[0].register_backward_hook(backward_hook)
    unwrapped_model_with_pre_process = unwrap_model(
        model_with_pre_process, (torchDDP, LocalDDP, Float16Module))[0]
    embedding = unwrapped_model_with_pre_process.language_model.embedding

    single_layer_device = get_single_layer_device(num_gpus_per_node)
    torch.cuda.set_device(single_layer_device)
    # Get single transformer layer model
    assert args.num_layers == args.pipeline_model_parallel_size == 1
    def model_provider_without_pre_or_post_process(pre_process=True, post_process=True):  # pylint: disable=unused-argument
        return model_provider(pre_process=False, post_process=False)
    model, optimizer, _ = setup_model_and_optimizer(
        model_provider_without_pre_or_post_process)
    model[0].register_forward_pre_hook(forward_hook)
    model[0].register_backward_hook(backward_hook)

    post_process_device = get_post_process_device(num_gpus_per_node)
    torch.cuda.set_device(post_process_device)
    if pre_process_device != post_process_device:
        tmp_pre_process_model = model_provider(pre_process=True, post_process=False)
        embedding = tmp_pre_process_model.language_model.embedding
        del tmp_pre_process_model
    def model_provider_with_post_process(pre_process=True, post_process=True): # pylint: disable=unused-argument
        new_model = model_provider(pre_process=False, post_process=True)
        return new_model
    model_with_post_process, optimizer_with_post_process, _ = setup_model_and_optimizer(
        model_provider_with_post_process)
    model_with_post_process[0].register_forward_pre_hook(forward_hook)
    model_with_post_process[0].register_backward_hook(backward_hook)
    # embedding set after optimizer is created so that the embedding is excluded from the optimizer
    unwrapped_model_with_post_process = unwrap_model(
        model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0]
    unwrapped_model_with_post_process.language_model.embedding = \
            embedding

    models = Models(model_with_pre_process=model_with_pre_process,
                    model_without_pre_or_post_process=model,
                    model_with_post_process=model_with_post_process)

    optimizers = Optimizers(optimizer_with_pre_process=optimizer_with_pre_process,
                            optimizer_without_pre_or_post_process=optimizer,
                            optimizer_with_post_process=optimizer_with_post_process)
    embedding = unwrapped_model_with_pre_process.language_model.embedding
    return models, optimizers, embedding.word_embeddings.weight.size()

def get_train_dataset(dataset='gpt'):
    """Create dataset for training"""
    if dataset != 'gpt':
        raise NotImplementedError
    args = get_args()
    train_val_test_num_samples = [args.global_batch_size * (NUM_AVERAGE + 1), 0, 0]
    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    return train_ds

def get_train_data_iterator(train_data_iterator):
    """Create train data iterator for the current micro batch size"""
    train_dataloader = build_pretraining_data_loader(
        train_data_iterator, 0)
    train_data_iterator = iter(cyclic_iter(train_dataloader))
    return train_data_iterator

def get_forward_step_time(forward_step_func, train_data_iterator,
                          model, compute_loss=False):
    """Get single layer forward step time"""
    s = time.time()
    output, loss_func = forward_step_func(train_data_iterator, model)
    if compute_loss:
        output = loss_func(output)
        loss, _ = output
        output = loss

    return output, DistributedWrapperContext.START_TIME - s

def get_backward_step_time(optimizer, input_tensor,
                           output_tensor, output_tensor_grad):
    """Get single layer backward step time"""

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad

def get_optimizer_time(model: torch.nn.Module, optimizer: MegatronOptimizer):
    """Get single layer optimizer time"""
    tuner_logger.debug("get_optimizer_time")
    for param in model.parameters():
        if param.requires_grad:
            if optimizer.params_have_main_grad:
                param.main_grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()
            else:
                param.grad = torch.randn(list(param.size()), dtype=param.dtype).cuda()

    for i in range(NUM_AVERAGE + 1):
        if i == 1:
            torch_cuda_synchronize()
            s = time.time()
        optimizer.step()
    torch_cuda_synchronize()
    e = time.time()

    # do allreduce gradients
    DistributedWrapperContext.record_comm_logs(after_forward_backward=True)
    tuner_logger.debug("start allreduce gradients")
    model.allreduce_gradients()
    torch_cuda_synchronize()
    tuner_logger.debug("finish allreduce gradients")
    dp_comm_logs = DistributedWrapperContext.get_comm_logs(after_forward_backward=True)
    DistributedWrapperContext.reset_comm_logs()
    optimizer.zero_grad()
    opt_time = (e - s) / NUM_AVERAGE
    tuner_logger.debug(f"finish get_optimizer_time: {opt_time}, {dp_comm_logs}")
    return opt_time, dp_comm_logs

def do_forward_backward(num_gpus_per_node, forward_step_func, models,  # pylint: disable=too-many-arguments
                        optimizers, train_data_iterator,
                        pre_process=False, post_process=False,
                        input_tensor_shape=None):
    """Execute forward and backward computation for a single layer"""
    tuner_logger.debug(f"run model start pre - {pre_process}, post - {post_process}")
    if pre_process:
        torch.cuda.set_device(0)
        input_tensor = None
        model = models.model_with_pre_process[0]
        optimizer = optimizers.optimizer_with_pre_process
    elif post_process:
        torch.cuda.set_device(get_post_process_device(num_gpus_per_node))
        input_tensor = torch.randn(list(input_tensor_shape)).cuda()
        input_tensor.requires_grad = True
        model = models.model_with_post_process[0]
        optimizer = optimizers.optimizer_with_post_process
    else:
        torch.cuda.set_device(get_single_layer_device(num_gpus_per_node))
        input_tensor = torch.randn(list(input_tensor_shape)).cuda()
        input_tensor.requires_grad = True
        model = models.model_without_pre_or_post_process[0]
        optimizer = optimizers.optimizer_without_pre_or_post_process


    # do forward and backward to get peak memory
    for i in range(NUM_AVERAGE + 1):
        if i == 0:
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            DistributedWrapperContext.record_comm_logs()

            unwrapped_model = unwrap_model(
                model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)

        # do forward
        output, get_batch_time = get_forward_step_time(
            forward_step_func,
            train_data_iterator,
            model,
            compute_loss=post_process)
        if i == 0:
            activation_shape = output.size()
            activation_size = output.nelement() * output.element_size()

            # do backward
            if post_process:
                output_tensor_grad = None
            else:
                output_tensor_grad = torch.randn(list(activation_shape)).cuda()
                output_tensor_grad.requires_grad = True

        get_backward_step_time(
            optimizer,
            input_tensor,
            output,
            output_tensor_grad)

        if i == 0:
            forward_backward_comm_logs = DistributedWrapperContext.get_comm_logs(
                pre_process, post_process)

            torch_cuda_synchronize()
            peak_memory = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
            s = time.time()

    torch_cuda_synchronize()
    e = time.time()
    if not pre_process:
        e -= get_batch_time
    forward_backward_time = (e - s) / NUM_AVERAGE
    return forward_backward_time, activation_shape, activation_size, \
            peak_memory, forward_backward_comm_logs

def get_iter_time_estimation(forward_backward_times,  # pylint: disable=too-many-arguments, too-many-statements
                             optimizer_times,
                             mp_comm_times,
                             pp_warmup_comm_time_per_stage,
                             pp_steady_comm_time_per_stage,
                             pp_embedding_sync_time,
                             dp_comm_times):
    """Get iter time estimation as milliseconds"""
    tuner_logger.debug(f"forward_backward_times: {forward_backward_times}")
    tuner_logger.debug(f"optimizer_times: {optimizer_times}")
    tuner_logger.debug(f"mp_comm_times: {mp_comm_times}")
    tuner_logger.debug(f"pp_warmup_comm_time_per_stage: {pp_warmup_comm_time_per_stage}")
    tuner_logger.debug(f"pp_steady_comm_time_per_stage: {pp_steady_comm_time_per_stage}")
    tuner_logger.debug(f"pp_embedding_sync_time: {pp_embedding_sync_time}")
    tuner_logger.debug(f"dp_comm_times: {dp_comm_times}")

    args = get_args()
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size

    warmup_kernel_forward_backward_time = forward_backward_times.get_total(num_layers) * 1000
    warmup_mp_comm_time = mp_comm_times.get_total(num_layers) * 1000
    warmup_pp_time = 0
    num_layers_per_stage = num_layers / args.pipeline_model_parallel_size
    for stage in range(args.pipeline_model_parallel_size):
        warmup_pp_time += \
                pp_warmup_comm_time_per_stage[stage]
    warmup_pp_time *= 1000
    warmup_time = warmup_kernel_forward_backward_time + \
            warmup_mp_comm_time + \
            warmup_pp_time
    log = f"warmup_kernel_forward_backward_time: {warmup_kernel_forward_backward_time}, " \
          f"warmup_mp_comm_time: {warmup_mp_comm_time}, " \
          f"warmup_pp_comm_time: {warmup_pp_time}, " \
          f"warmup_time: {warmup_time}"
    tuner_logger.debug(log)

    if args.pipeline_model_parallel_size == 1:
        stage_time = warmup_time
        mp_stage_time = warmup_mp_comm_time
        pp_stage_time = warmup_pp_time
        stage_optimizer_time = optimizer_times.get_total(num_layers) * 1000
        dp_comm_time = dp_comm_times.get_total(num_layers) * 1000
    else:
        mp_stage_time = 0
        pp_stage_time = 0
        stage_time = 0
        for stage in range(args.pipeline_model_parallel_size):
            curr_stage_time = 0
            curr_mp_stage_time = 0
            curr_pp_stage_time = 0
            if stage == 0:
                curr_stage_time += forward_backward_times.pre_process
                curr_stage_time += mp_comm_times.pre_process
                curr_mp_stage_time += mp_comm_times.pre_process
            elif stage == args.pipeline_model_parallel_size - 1:
                curr_stage_time += forward_backward_times.post_process
                curr_stage_time += mp_comm_times.post_process
                curr_mp_stage_time += mp_comm_times.post_process

            curr_stage_time += forward_backward_times.single_layer * num_layers_per_stage
            curr_stage_time += mp_comm_times.single_layer * num_layers_per_stage
            curr_mp_stage_time += mp_comm_times.single_layer * num_layers_per_stage
            curr_stage_time += pp_steady_comm_time_per_stage[stage]
            curr_pp_stage_time += pp_steady_comm_time_per_stage[stage]

            stage_time = max(stage_time, curr_stage_time)
            mp_stage_time = max(mp_stage_time, curr_mp_stage_time)
            pp_stage_time = max(pp_stage_time, curr_pp_stage_time)
        stage_time *= 1000
        mp_stage_time *= 1000
        pp_stage_time *= 1000

        # the first or last stage finishes parameter update at last
        stage_optimizer_time = optimizer_times.get_max(num_layers_per_stage) * 1000
        dp_comm_time = dp_comm_times.get_max(num_layers_per_stage) * 1000

    pp_embedding_sync_time *= 1000
    iter_time = warmup_time + stage_time * (num_micro_batches - 1)
    iter_time += stage_optimizer_time + pp_embedding_sync_time + dp_comm_time
    mp_time = warmup_mp_comm_time + mp_stage_time * (num_micro_batches - 1)
    pp_time = warmup_pp_time + pp_stage_time * (num_micro_batches - 1)
    dp_time = dp_comm_time

    tuner_logger.info(f"iter_time - {iter_time}")
    log = f"mp_time - {mp_time}, pp_time - {pp_time}, dp_time - {dp_time}, " \
          f"pp_embedding_sync_time - {pp_embedding_sync_time}," \
          f"optimizer_time - {stage_optimizer_time}"
    tuner_logger.info(log)
    return iter_time, mp_time, pp_time, dp_time

def tensor_nested_iterator(item):
    """Iterate module's state_dict to find all leaf tensors"""
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
    """Get the sum of all the unique parameter sizes as bytes"""
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
        [tensor_nested_iterator(
            optimizers.optimizer_with_pre_process.state_dict()),  # optimizer states (32 bit)
         optimizers.optimizer_with_pre_process.get_parameters(),  # optimizer params (32 bit)
         models.model_with_pre_process[0].parameters()])  # model params (32 bit or 16bit)

    without_pre_or_post_process_param_size = get_unique_param_size(
        [tensor_nested_iterator(
            optimizers.optimizer_without_pre_or_post_process.state_dict()),
         optimizers.optimizer_without_pre_or_post_process.get_parameters(),
         models.model_without_pre_or_post_process[0].parameters()])

    # exclude embedding params from post_process
    unwrapped_model_with_post_process = unwrap_model(
        models.model_with_post_process, (torchDDP, LocalDDP, Float16Module))[0].language_model
    orig_embedding = unwrapped_model_with_post_process.embedding
    unwrapped_model_with_post_process.embedding = None
    with_post_process_param_size = get_unique_param_size(
        [tensor_nested_iterator(
            optimizers.optimizer_with_post_process.state_dict()),
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
    """Get required gpu memory to execute a model"""
    args = get_args()
    num_layers = args.num_layers
    num_micro_batches = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
    pp = args.pipeline_model_parallel_size
    num_layers_per_stage = num_layers / pp
    assert args.activations_checkpoint_method == 'uniform'
    peak_memory = max(peak_memories.pre_process,
                      peak_memories.single_layer,
                      peak_memories.post_process)
    peak_memory /= 1024 * 1024
    num_checkpoint_layers = num_layers_per_stage / args.activations_checkpoint_num_layers
    checkpoint_size = (activation_size * (num_checkpoint_layers - 1)) / 1024 / 1024
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
        checkpoint_size_per_single_layer = activation_size / args.activations_checkpoint_num_layers
        grad_size = max(0, grad_sizes.single_layer - checkpoint_size_per_single_layer)
        grad_size *= num_layers_per_stage / 1024 / 1024
        if pp == 1:
            grad_size += (grad_sizes.pre_process + grad_sizes.post_process) / 1024 / 1024
        else:
            grad_size += max(grad_sizes.pre_process, grad_sizes.post_process) / 1024 / 1024
    log = f"param_size - {param_size}, peak_memory - {peak_memory}, " \
          f"checkpoint_size - {checkpoint_size}, grad_size - {grad_size}"
    tuner_logger.info(log)
    return param_size + peak_memory + checkpoint_size + grad_size


def get_p2p_ranks_from_prev_stage(stage, stage_ranks):
    """Get p2p communication ranks for the given stage when
       the stage receives data from the previous stage.
    """
    p2p_ranks_from_prev_stage = {}
    for stage_rank_list in stage_ranks.values():
        src_rank = stage_rank_list[stage - 1]
        dst_rank = stage_rank_list[stage]
        p2p_ranks_from_prev_stage[src_rank] = [src_rank, dst_rank]
        p2p_ranks_from_prev_stage[dst_rank] = [src_rank, dst_rank]
    return p2p_ranks_from_prev_stage

def get_p2p_ranks_from_next_stage(stage, stage_ranks):
    """Get p2p communication ranks for the given stage when
       the stage receives data from the next stage.
    """
    p2p_ranks_from_next_stage = {}
    for stage_rank_list in stage_ranks.values():
        src_rank = stage_rank_list[stage + 1]
        dst_rank = stage_rank_list[stage]
        p2p_ranks_from_next_stage[src_rank] = [src_rank, dst_rank]
        p2p_ranks_from_next_stage[dst_rank] = [src_rank, dst_rank]
    return p2p_ranks_from_next_stage

def get_p2p_ranks_for_embedding_sync(stage_ranks):
    """P2P ranks for embedding sync for each (mp, dp)"""
    p2p_ranks_for_embedding_sync = {}
    for stage_rank_list in stage_ranks.values():
        p2p_ranks_for_embedding_sync[stage_rank_list[0]] = \
                [stage_rank_list[0], stage_rank_list[-1]]
        p2p_ranks_for_embedding_sync[stage_rank_list[-1]] = \
                [stage_rank_list[0], stage_rank_list[-1]]
    return p2p_ranks_for_embedding_sync

def get_p2p_ranks_for_all(stage_ranks, task_to_rank):
    """Get communication ranks when all gpus communicate with prev or next"""
    all_send_recv_ranks = {}
    for task, rank in task_to_rank.items():
        key = (task.mp, task.dp)
        if task.pp % 2 == 0:
            all_send_recv_ranks[rank] = [rank, stage_ranks[key][task.pp + 1]]
        else:
            all_send_recv_ranks[rank] = [stage_ranks[key][task.pp - 1], rank]
    return all_send_recv_ranks

def get_p2p_ranks_except_first_and_last_stage(stage_ranks, task_to_rank, pp):
    """Get communication ranks when all gpus communicate with prev or next
       except the first and the last stage"""

    send_recv_except_first_and_last_ranks = {}
    for task, rank in task_to_rank.items():
        key = (task.mp, task.dp)
        stage = task.pp
        if task.pp % 2 == 0:
            if stage != 0:
                send_recv_except_first_and_last_ranks[rank] = \
                        [stage_ranks[key][task.pp - 1], rank]
        else:
            if stage != pp - 1:
                send_recv_except_first_and_last_ranks[rank] = \
                        [rank, stage_ranks[key][task.pp + 1]]
    return send_recv_except_first_and_last_ranks

def run_comm_helper(env, event, req_queue, resp_queue):
    """Run communication helper for each GPU"""
    rank = int(env['RANK'])
    os.environ = env
    comm_helper = CommHelper(event, req_queue, resp_queue)
    if rank == 0:
        comm_helper.run_msg_handler_for_rank0()
    else:
        comm_helper.run_msg_handler()

class Estimator:  # pylint: disable=too-many-instance-attributes
    """Estimator for a model's iteration time and gpu memory requirements
       for different parallelization methods as represented as configs.
    """
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

        self.curr_task_to_rank = {}

        self.mp_comm_logs = defaultdict(Log) # (mp, mb) -> mp comm logs for forward and backward
        self.dp_comm_logs = defaultdict(Log) # mp -> dp comm logs

        self.curr_mp = None
        self.models = None
        self.optimizers = None
        self.embedding_shape = None
        self.comm_helper_procs = []
        self.event = None
        self.req_queue = None
        self.resp_queue = None
        self.init_comm_helper_procs()
        self.comm_cache = {}

        DistributedWrapperContext.patch_dist_func(1)
        if self.model_name == 'gpt':
            # initialize as single gpu at the first
            os.environ['WORLD_SIZE'] = '1'
            args_defaults = args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
            initialize_megatron(args_defaults=args_defaults)
            self.train_ds = get_train_dataset(dataset='gpt')
            self.forward_step_func = gpt_forward_step
            os.environ['WORLD_SIZE'] = str(self.world_size)
        else:
            raise NotImplementedError
        DistributedWrapperContext.patch_dist_func(self.world_size)
        self.curr_config = None

    def init_comm_helper_procs(self):
        """Initialize communication helper processes for this machine.
           Non-master process(node_rank!=0) just executes communication helper until
           the job is terminated.
           Master process(node_rank==0) reset communication helper processes
        """
        tuner_logger.debug("init_comm_helper_procs")
        ctx = multiprocessing.get_context('spawn')
        node_rank = int(os.environ['NODE_RANK'])
        while True:
            self.terminate()
            self.comm_helper_procs.clear()

            self.event = ctx.Event()
            self.req_queue = ctx.Queue() if node_rank == 0 else None
            self.resp_queue = ctx.Queue() if node_rank == 0 else None
            rank = node_rank * self.num_gpus_per_node
            for local_rank in range(self.num_gpus_per_node):
                env = os.environ.copy()
                env['WORLD_SIZE'] = self.world_size
                env['RANK'] = str(rank)
                env['LOCAL_RANK'] = str(local_rank)
                proc = ctx.Process(
                    target=run_comm_helper,
                    args=(env, self.event, self.req_queue, self.resp_queue))
                self.comm_helper_procs.append(proc)
                proc.start()
                rank += 1

            if node_rank == 0:
                # wait until initialization
                self.resp_queue.get()
                return
            for proc in self.comm_helper_procs:
                proc.join()

    def terminate(self):
        """Terminate estimator and communication helper processes"""
        assert self.comm_helper_procs
        if self.req_queue:
            self.req_queue.put((CommType.END, None, None, None))
        for proc in self.comm_helper_procs:
            proc.join()

    def request_comm(self, comm_type, comm_ranks, tensor_dtype, tensor_shape):
        """Get communication time by requesting communication to communication helpers"""
        assert os.environ['NODE_RANK'] == '0'
        comm_rank_key = json.dumps(comm_ranks, sort_keys=True)
        key = (comm_type, comm_rank_key, tensor_dtype, tuple(tensor_shape))
        if key in self.comm_cache:
            return self.comm_cache[key]

        while True:
            self.req_queue.put((comm_type, comm_ranks, tensor_dtype, tensor_shape))

            while True:
                try:
                    comm_time = self.resp_queue.get(False)
                    self.comm_cache[key] = comm_time
                    return comm_time
                except Exception:  # pylint: disable=broad-except
                    if self.event.is_set():
                        # assume all communication errors as oom error
                        raise RuntimeError("out of memory")  # pylint: disable=raise-missing-from
                    time.sleep(1)

    def _get_models_and_optimizers(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        if mp == self.curr_mp:
            return self.models, self.optimizers
        dp = args.data_parallel_size

        if self.model_name != 'gpt':
            raise NotImplementedError

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
                            dp=original_config.dp)
        DistributedWrapperContext.CURR_CONFIG = new_config
        args.num_layers = 1
        args.pipeline_model_parallel_size = 1

        self.models, self.optimizers, self.embedding_shape = \
                get_single_layer_model(gpt_model_provider,
                                       self.num_gpus_per_node)

        args.pipeline_model_parallel_size = original_pp
        args.num_layers = original_layers
        DistributedWrapperContext.CURR_CONFIG = original_config
        self.curr_mp = mp
        return self.models, self.optimizers

    def _get_optimizer_times(self):
        mp = get_args().tensor_model_parallel_size
        if mp not in self.optimizer_times:
            self._set_optimizer_time_and_memory()
        return self.optimizer_times[mp]

    def _set_optimizer_time_and_memory(self):
        models, optimizers = self._get_models_and_optimizers()

        model = models.model_with_pre_process[0]
        optimizer = optimizers.optimizer_with_pre_process
        torch.cuda.set_device(0)
        pre_process_optimizer_time, pre_process_dp_comm_logs = \
                get_optimizer_time(model, optimizer)
        self._set_dp_comm_logs(pre_process_dp_comm_logs, pre_process=True, post_process=False)

        model = models.model_with_post_process[0]
        optimizer = optimizers.optimizer_with_post_process
        torch.cuda.set_device(get_post_process_device(self.num_gpus_per_node))
        post_process_optimizer_time, post_process_dp_comm_logs = \
                get_optimizer_time(model, optimizer)
        self._set_dp_comm_logs(post_process_dp_comm_logs, pre_process=False, post_process=True)

        model = models.model_without_pre_or_post_process[0]
        optimizer = optimizers.optimizer_without_pre_or_post_process
        torch.cuda.set_device(get_single_layer_device(self.num_gpus_per_node))
        single_layer_optimizer_time, single_layer_comm_logs = \
                get_optimizer_time(model, optimizer)
        self._set_dp_comm_logs(single_layer_comm_logs, pre_process=False, post_process=False)

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

    def _set_curr_config(self, config):
        if config == self.curr_config:
            return

        DistributedWrapperContext.CURR_CONFIG = config
        self.curr_config = config

        # initilaize communication helper processes to prevent oom error
        # due to many communication groups(too much of nccl buffer size)
        self.init_comm_helper_procs()

    def _set_curr_task_to_rank(self, mp_comm_size, pp_comm_size, dp_comm_size):
        comm_size = {'mp': mp_comm_size, 'pp': pp_comm_size, 'dp': dp_comm_size}

        parallelism_order = [k for k, v in sorted(comm_size.items(), key=lambda item: -item[1])]
        tuner_logger.info(f"curr parallelism order: {parallelism_order}")

        config = self.curr_config
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

        #DistributedWrapperContext.unpatch_dist_func()
        comm_ranks = self._get_comm_ranks(is_mp, is_dp)
        comm_times = CommTime()

        for comm_type, tensor_dtype, tensor_shape in comm_logs:
            # the same time for the collective communication
            single_comm_time_per_rank = self.request_comm(
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
        #DistributedWrapperContext.patch_dist_func(self.world_size)
        return comm_times

    def _get_mp_comm_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size

        if (mp, mb) not in self.mp_comm_logs:
            self._set_forward_backward_time_and_memory()

        comm_logs = self.mp_comm_logs[(mp, mb)].pre_process
        pre_process_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_comm_logs[(mp, mb)].single_layer
        single_layer_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        comm_logs = self.mp_comm_logs[(mp, mb)].post_process
        post_process_comm_time = self._get_comm_times(comm_logs, is_mp=True)

        mp_comm_times = TimeOrMemory(
            max(0, pre_process_comm_time.mp - single_layer_comm_time.mp),
            single_layer_comm_time.mp,
            max(0, post_process_comm_time.mp - single_layer_comm_time.mp))
        return mp_comm_times

    def _get_dp_comm_times(self):
        args = get_args()
        mp = args.tensor_model_parallel_size
        # dp communication is dependent on the parameter size
        if mp not in self.dp_comm_logs:
            self._set_optimizer_time_and_memory()

        comm_logs = self.dp_comm_logs[mp].pre_process
        pre_process_comm_time = self._get_comm_times(comm_logs, is_dp=True)

        comm_logs = self.dp_comm_logs[mp].single_layer
        single_layer_comm_time = self._get_comm_times(comm_logs, is_dp=True)

        comm_logs = self.dp_comm_logs[mp].post_process
        post_process_comm_time = self._get_comm_times(comm_logs, is_dp=True)

        dp_comm_times = TimeOrMemory(
            max(0, pre_process_comm_time.dp - single_layer_comm_time.dp),
            single_layer_comm_time.dp,
            max(0, post_process_comm_time.dp - single_layer_comm_time.dp))
        return dp_comm_times

    def _get_comm_size(self):
        """Get each parallelism communication size for a single GPU"""
        args = get_args()
        mp = args.tensor_model_parallel_size
        dp = args.data_parallel_size
        pp = args.pipeline_model_parallel_size
        mb = args.micro_batch_size
        num_micro_batches = args.global_batch_size // dp // mb

        mp_size, pp_size, dp_size = 0, 0, 0
        mp_comm_logs = self.mp_comm_logs[(mp, mb)].single_layer
        for comm_type, tensor_dtype, tensor_shape in mp_comm_logs:
            assert comm_type in [CommType.BROADCAST, CommType.ALLREDUCE]
            elem_size = 2 if tensor_dtype in [torch.int16, torch.float16] else 4
            num_elems = 1
            for size in tensor_shape:
                num_elems *= size
            mp_size += elem_size * num_elems
        mp_size *= num_micro_batches

        if pp > 1:
            pp_elem_size = 2 if args.fp16 else 4
            pp_num_elems = args.seq_length * args.micro_batch_size * args.hidden_size
            pp_size = pp_elem_size * pp_num_elems * num_micro_batches * 2 # forward/backward

        return mp_size, pp_size, dp_size

    def _get_stage_ranks(self):
        """Get stage ranks in order for each (mp, dp) """
        args = get_args()

        stage_ranks = {} # (mp, dp) -> list of ranks
        pp = args.pipeline_model_parallel_size
        for mp in range(args.tensor_model_parallel_size):
            for dp in range(args.data_parallel_size):
                stage_ranks[(mp, dp)] = [-1] * pp
        for task, rank in self.curr_task_to_rank.items():
            stage_ranks[(task.mp, task.dp)][task.pp] = rank
        return stage_ranks

    def _get_pp_comm_time_per_stage(self):  # pylint: disable=too-many-branches
        args = get_args()
        pp = args.pipeline_model_parallel_size

        pp_warmup_comm_time_per_stage = {0:0}
        pp_steady_comm_time_per_stage = {0:0}
        pp_embedding_sync_time = 0
        if pp == 1:
            return pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, \
                    pp_embedding_sync_time

        stage_ranks = self._get_stage_ranks()

        assert pp % 2 == 0
        all_send_recv_ranks = get_p2p_ranks_for_all(stage_ranks, self.curr_task_to_rank)
        send_recv_except_first_and_last_ranks = \
            get_p2p_ranks_except_first_and_last_stage(stage_ranks, self.curr_task_to_rank, pp)

        tensor_dtype = torch.float16 if args.fp16 else torch.float32
        tensor_shape = [args.seq_length, args.micro_batch_size, args.hidden_size]
        for stage in range(pp):
            stage_rank = stage_ranks[(0, 0)][stage]

            # warmup_forward: recv_forward
            if stage == 0:
                recv_forward = 0
            else:
                p2p_ranks_from_prev_stage = get_p2p_ranks_from_prev_stage(
                    stage, stage_ranks)
                recv_forward = self.request_comm(
                    CommType.SEND_OR_RECV, p2p_ranks_from_prev_stage,
                    tensor_dtype, tensor_shape)[stage_rank]

            # warmup_backward: recv_backward
            if stage == pp - 1:
                recv_backward = 0
            else:
                p2p_ranks_from_next_stage = get_p2p_ranks_from_next_stage(stage, stage_ranks)
                recv_backward = self.request_comm(
                    CommType.SEND_OR_RECV, p2p_ranks_from_next_stage,
                    tensor_dtype, tensor_shape)[stage_rank]
            pp_warmup_comm_time_per_stage[stage] = recv_forward + recv_backward

            # steady: all_send_recv_ranks + send_recv_except_first_and_last_ranks
            all_send_recv_time = self.request_comm(
                CommType.SEND_AND_RECV,
                all_send_recv_ranks,
                tensor_dtype, tensor_shape)[stage_rank]

            if stage not in (0, pp - 1):
                send_recv_except_first_and_last_time = self.request_comm(
                    CommType.SEND_AND_RECV,
                    send_recv_except_first_and_last_ranks,
                    tensor_dtype, tensor_shape)[stage_rank]
            else:
                send_recv_except_first_and_last_time = 0

            pp_steady_comm_time_per_stage[stage] = \
                    all_send_recv_time + send_recv_except_first_and_last_time

        if pp > 1:
            p2p_ranks_for_embedding_sync = get_p2p_ranks_for_embedding_sync(stage_ranks)
            embedding_shape = self.embedding_shape
            pp_embedding_sync_time = self.request_comm(
                CommType.ALLREDUCE,
                p2p_ranks_for_embedding_sync,
                torch.float16 if args.fp16 else torch.float32,
                embedding_shape)[stage_ranks[(0,0)][0]]

        return pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, pp_embedding_sync_time

    def _set_mp_comm_logs(self, comm_logs, pre_process=False, post_process=False):
        args = get_args()
        mp = args.tensor_model_parallel_size
        mb = args.micro_batch_size
        for comm_group, comms_to_measure in comm_logs.items():
            is_mp = comm_group == 'tensor_model_parallel_group'
            if not is_mp:
                tuner_logger.info(f"{comms_to_measure} of {comm_group} is ignored")
                continue
            self.mp_comm_logs[(mp, mb)].set(comms_to_measure, pre_process, post_process)

    def _set_dp_comm_logs(self, comm_logs, pre_process=False, post_process=False):
        args = get_args()
        mp = args.tensor_model_parallel_size
        for comm_group, comms_to_measure in comm_logs.items():
            is_dp = comm_group == 'data_parallel_group'
            if not is_dp:
                tuner_logger.info(f"{comms_to_measure} of {comm_group} is ignored")
                continue
            self.dp_comm_logs[mp].set(comms_to_measure, pre_process, post_process)

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

        assert not isinstance(models.model_with_pre_process, torchDDP)

        activation_shape = None
        activation_size = None

        pre_process_forward_backward_time, \
                activation_shape, _, pre_process_peak_memory, pre_process_comm_logs = \
            do_forward_backward(self.num_gpus_per_node,
                                self.forward_step_func, models, optimizers,
                                train_data_iterator, pre_process=True)
        self._set_mp_comm_logs(pre_process_comm_logs, pre_process=True)

        single_layer_forward_backward_time, \
                activation_shape, activation_size, single_layer_peak_memory, \
                single_layer_comm_logs = \
            do_forward_backward(self.num_gpus_per_node,
                                self.forward_step_func, models, optimizers,
                                train_data_iterator, input_tensor_shape=activation_shape)
        self._set_mp_comm_logs(single_layer_comm_logs)

        post_process_forward_backward_time, \
                _, _, post_process_peak_memory, post_process_comm_logs = \
            do_forward_backward(self.num_gpus_per_node,
                                self.forward_step_func, models, optimizers,
                                train_data_iterator, post_process=True,
                                input_tensor_shape=activation_shape)
        self._set_mp_comm_logs(post_process_comm_logs, post_process=True)

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
        """Get estimated iteration time for given config"""

        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        self._set_curr_config(config)

        try:
            forward_backward_times = self._get_compute_times()
            optimizer_times = self._get_optimizer_times()

            mp_comm_size, pp_comm_size, dp_comm_size = self._get_comm_size()
            self._set_curr_task_to_rank(mp_comm_size, pp_comm_size, dp_comm_size)
            mp_comm_times = self._get_mp_comm_times()
            pp_warmup_comm_time_per_stage, pp_steady_comm_time_per_stage, \
                    pp_embedding_sync_time = self._get_pp_comm_time_per_stage()
            dp_comm_times = self._get_dp_comm_times()

            iter_time, mp_time, pp_time, dp_time = get_iter_time_estimation(
                    forward_backward_times, optimizer_times,
                    mp_comm_times, pp_warmup_comm_time_per_stage,
                    pp_steady_comm_time_per_stage, pp_embedding_sync_time,
                    dp_comm_times)
        except RuntimeError as e:
            if "out of memory" in str(e):
                tuner_logger.info(f"OOM for {config}")
                return 0, 0, 0, 0
            raise e

        return iter_time, mp_time, pp_time, dp_time

    def get_max_gpu_memory(self, config):
        """Get required per gpu memory for given config"""
        # apply config to arguments
        args = get_args()
        args.tensor_model_parallel_size = config.mp
        args.pipeline_model_parallel_size = config.pp
        args.data_parallel_size = config.dp
        args.micro_batch_size = config.micro_batch_size
        args.global_batch_size = config.global_batch_size

        self._set_curr_config(config)

        try:
            param_sizes, grad_sizes = self._get_param_and_grad_sizes()
            activation_sizes = self._get_activation_size()
            peak_memories = self._get_peak_memories()

            req_gpu_memory = get_required_gpu_memory(
                param_sizes, grad_sizes, activation_sizes, peak_memories)
        except RuntimeError as e:
            if "out of memory" in str(e):
                tuner_logger.info(f"OOM for {config}")
                return 0
            raise e

        return req_gpu_memory
