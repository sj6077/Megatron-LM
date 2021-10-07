"""Receive commuinication request from estimator and send back the communication time"""
from enum import IntEnum
import os
import time
from typing import List

import torch
from torch._C._distributed_c10d import ReduceOp

NUM_AVERAGE = 100

supported_dtypes = [torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]

class CommType(IntEnum):
    END = 0
    BROADCAST = 1
    ALLREDUCE = 2

def encode_comm_type(comm_type: CommType):
    return torch.IntTensor([int(comm_type)])

def decode_comm_type(tensor: torch.Tensor):
    comm_type_val = tensor.item()
    return CommType(comm_type_val)

def encode_comm_ranks(world_size:int, comm_ranks: List[int]):
    """Return an IntTensor with a length of world_size + 1.
       Its contents are num_comm_ranks, comm_ranks, -1 for the rest"""

    val = [-1] * (world_size + 1)
    num_ranks = len(comm_ranks)
    assert num_ranks <= world_size
    val[0] = num_ranks
    for i, comm_rank in enumerate(comm_ranks):
        val[i+1] = comm_rank
    return torch.IntTensor(val)

def decode_comm_ranks(tensor: torch.Tensor):
    val = tensor.tolist()
    num_ranks = val[0]
    comm_ranks = [val[i+1] for i in range(num_ranks)]
    return comm_ranks

def encode_tensor_dtype_and_shape(dtype: torch.Tensor.dtype, tensor_shape: List[int]):
    dtype_id = supported_dtypes.index(dtype)
    assert len(tensor_shape) < 9
    val = [0] * 10
    val[0] = dtype_id
    val[1] = len(tensor_shape)
    for i, size in enumerate(tensor_shape):
        val[i+2] = size
    return torch.IntTensor(val)

def decode_tensor_dtype_and_shape(tensor: torch.Tensor):
    val = tensor.tolist()
    dtype_id = val[0]
    dtype = supported_dtypes[dtype_id]
    dim = val[1]
    return dtype, [val[i+2] for i in range(dim)]

class CommHelper:
    """A helper class that can estimate nccl communication time."""

    def __init__(self):
        assert 'MASTER_ADDR' in os.environ
        assert 'MASTER_PORT' in os.environ

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        
        torch.distributed.init_process_group(
                backend='gloo',
                world_size=world_size,
                rank=rank)
        self.world_size = world_size
        self.my_rank = rank

        self.comm_type_tensor = torch.IntTensor([0])
        self.comm_ranks_tensor = torch.IntTensor([0] * (world_size + 1))
        self.tensor_shape_tensor = torch.IntTensor([0] * 10)

        self.comm_group = {}
        self.comm_cache = {}

    def request_comm(self, comm_type, comm_ranks, tensor_dtype, tensor_shape):
        comm_ranks.sort()
        key = (comm_type, tuple(comm_ranks), tensor_dtype, tuple(tensor_shape))
        if key in self.comm_cache:
            return self.comm_cache[key]

        assert self.my_rank == 0

        encoded_comm_type = encode_comm_type(comm_type)
        encoded_comm_ranks = encode_comm_ranks(self.world_size, comm_ranks)
        encoded_tensor_shape = encode_tensor_dtype_and_shape(
                tensor_dtype, tensor_shape)

        torch.distributed.broadcast(encoded_comm_type, src=0)
        torch.distributed.broadcast(encoded_comm_ranks, src=0)
        torch.distributed.broadcast(encoded_tensor_shape, src=0)

        comm_time = self._handle_comm(
                comm_type, comm_ranks, tensor_dtype, tensor_shape)
        self.comm_cache[key] = comm_time
        return comm_time

    def terminate(self):
        encoded_comm_type = encode_comm_type(CommType.END)
        torch.distributed.broadcast(encoded_comm_type, src=0)
        torch.distributed.barrier()

    def run_msg_handler(self):
        while True:
            torch.distributed.broadcast(self.comm_type_tensor, src=0)
            comm_type = decode_comm_type(self.comm_type_tensor)
            if comm_type == CommType.END:
                break
            torch.distributed.broadcast(self.comm_ranks_tensor, src=0)
            comm_ranks = decode_comm_ranks(self.comm_ranks_tensor)
            torch.distributed.broadcast(self.tensor_shape_tensor, src=0)
            tensor_dtype, tensor_shape = decode_tensor_dtype_and_shape(self.tensor_shape_tensor)

            self._handle_comm(comm_type, comm_ranks, tensor_dtype, tensor_shape)
        torch.distributed.barrier()

    def _get_comm_group(self, comm_ranks):
        key = tuple(comm_ranks)
        if key not in self.comm_group:
            new_group = torch.distributed.new_group(ranks=comm_ranks, backend='nccl')
            self.comm_group[key] = new_group
        return self.comm_group[key]

    def _handle_comm(self, comm_type, comm_ranks, tensor_dtype, tensor_shape):
        """Execute given communication and send the communication time to rank0"""
        comm_time_tensor = torch.FloatTensor([0.0])
        if self.my_rank not in comm_ranks:
            torch.distributed.barrier()
            torch.distributed.all_reduce(comm_time_tensor, op=ReduceOp.SUM)
            comm_time = comm_time_tensor.item() / len(comm_ranks)
            return comm_time

        group = self._get_comm_group(comm_ranks)
        if tensor_dtype in [torch.int16, torch.int32, torch.int64]:
            comm_tensor = torch.randint(low=0, high=100, size=tuple(tensor_shape)).cuda()
        else:
            comm_tensor = torch.randn(tensor_shape, dtype=tensor_dtype).cuda()
        comm_tensor = comm_tensor.cuda()

        torch.distributed.barrier()
        for i in range(NUM_AVERAGE + 1):
            if i == 1:
                torch.cuda.synchronize()
                start = time.time()

            if comm_type == CommType.BROADCAST:
                torch.distributed.broadcast(comm_tensor, src=comm_ranks[0], group=group)
            elif comm_type == CommType.ALLREDUCE:
                torch.distributed.all_reduce(comm_tensor, group=group)
            else:
                raise NotImplementedError
        
        torch.cuda.synchronize()
        comm_time = (time.time() - start) / NUM_AVERAGE
        comm_time_tensor.data[0] = comm_time

        torch.distributed.all_reduce(comm_time_tensor, op=ReduceOp.SUM)
        comm_time = comm_time_tensor.item() / len(comm_ranks)
        return comm_time
