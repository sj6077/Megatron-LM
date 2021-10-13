"""Receive commuinication request from estimator and send back the communication time"""
from collections import defaultdict
from enum import IntEnum
import json
import os
import time
from typing import Dict, List

import torch
from torch._C._distributed_c10d import ReduceOp

NUM_AVERAGE = 5

supported_dtypes = [torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]

class CommType(IntEnum):
    END = 0
    BROADCAST = 1
    ALLREDUCE = 2
    SEND_OR_RECV = 3
    SEND_AND_RECV = 4

def encode_comm_type(comm_type: CommType):
    return torch.IntTensor([int(comm_type)])

def decode_comm_type(tensor: torch.Tensor):
    comm_type_val = tensor.item()
    return CommType(comm_type_val)

def encode_comm_ranks(world_size:int, rank_to_comm_ranks: Dict[int, List[int]]):
    """Return an IntTensor with a length of world_size + 1.
       Its contents are num_comm_ranks, comm_ranks, -1 for the rest"""

    val = [[]] * world_size
    for rank in range(world_size):
        if rank in rank_to_comm_ranks:
            comm_rank_size = len(rank_to_comm_ranks[rank])
            val[rank] = rank_to_comm_ranks[rank].copy()
            if comm_rank_size < world_size:
                padding = [-1] * (world_size - comm_rank_size)
                val[rank] += padding
        else:
            val[rank] = [-1] * world_size
        assert len(val[rank]) == world_size
    return torch.IntTensor(val)

def decode_comm_ranks(world_size: int, tensor: torch.Tensor):
    rank_to_comm_ranks = {}
    val = tensor.tolist()
    for rank in range(world_size):
        comm_ranks = [r for r in val[rank] if r != -1]
        if comm_ranks:
            rank_to_comm_ranks[rank] = comm_ranks
    return rank_to_comm_ranks

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
        self.local_rank = local_rank
        
        torch.distributed.init_process_group(
                backend='gloo',
                world_size=world_size,
                rank=rank)
        print("initialize communication helper", rank, world_size)
        self.world_size = world_size
        self.my_rank = rank

        self.comm_type_tensor = torch.IntTensor([0])
        self.comm_ranks_tensor = torch.IntTensor([[-1] * world_size] * world_size)
        self.tensor_shape_tensor = torch.IntTensor([0] * 10)

        self.comm_group = {}
        self.comm_cache = {}

    def request_comm(self, comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape):
        comm_rank_key = json.dumps(rank_to_comm_ranks, sort_keys=True)
        key = (comm_type, comm_rank_key, tensor_dtype, tuple(tensor_shape))
        if key in self.comm_cache:
            return self.comm_cache[key]

        assert self.my_rank == 0

        encoded_comm_type = encode_comm_type(comm_type)
        encoded_comm_ranks = encode_comm_ranks(self.world_size, rank_to_comm_ranks)
        encoded_tensor_shape = encode_tensor_dtype_and_shape(
                tensor_dtype, tensor_shape)

        torch.distributed.broadcast(encoded_comm_type, src=0)
        torch.distributed.broadcast(encoded_comm_ranks, src=0)
        torch.distributed.broadcast(encoded_tensor_shape, src=0)

        comm_time_per_rank = self._handle_comm(
                comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape)
        self.comm_cache[key] = comm_time_per_rank
        return comm_time_per_rank

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
            rank_to_comm_ranks = decode_comm_ranks(self.world_size, self.comm_ranks_tensor)
            torch.distributed.broadcast(self.tensor_shape_tensor, src=0)
            tensor_dtype, tensor_shape = decode_tensor_dtype_and_shape(self.tensor_shape_tensor)

            self._handle_comm(comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape)
        torch.distributed.barrier()

    def _get_comm_group(self, comm_ranks):
        key = tuple(comm_ranks)
        if key not in self.comm_group:
            new_group = torch.distributed.new_group(ranks=comm_ranks, backend='nccl')
            self.comm_group[key] = new_group
        return self.comm_group[key]

    def _handle_comm(self, comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape):
        torch.cuda.set_device(self.local_rank)

        """Execute given communication and send the communication time to rank0"""
        print(self.my_rank, comm_type, tensor_dtype, tensor_shape, rank_to_comm_ranks)
        if self.my_rank == 0:
            tensor_list = [torch.FloatTensor([0.0])] * self.world_size
        else:
            tensor_list = None

        comm_time_tensor = torch.FloatTensor([0.0])
        if tensor_dtype in [torch.int16, torch.int32, torch.int64]:
            comm_tensor = torch.randint(low=0, high=100, size=tuple(tensor_shape)).cuda()
        else:
            comm_tensor = torch.randn(tensor_shape, dtype=tensor_dtype).cuda()
        if comm_type == CommType.SEND_AND_RECV:
            comm_tensor2 = comm_tensor.clone()

        groups = {}
        for rank in range(self.world_size):
            if rank in rank_to_comm_ranks:
                # group must be initialized across all ranks
                group = self._get_comm_group(rank_to_comm_ranks[rank])
                groups[rank] = group

        torch.distributed.barrier()
        torch.cuda.synchronize()
        if self.my_rank in rank_to_comm_ranks:
            comm_ranks = rank_to_comm_ranks[self.my_rank]
            group = groups[self.my_rank]
            if len(comm_ranks) > 1:
                comm_times = []
                for i in range(NUM_AVERAGE + 1):
                    if comm_type == CommType.BROADCAST:
                        torch.distributed.broadcast(comm_tensor, src=comm_ranks[0], group=group)
                    elif comm_type == CommType.ALLREDUCE:
                        torch.distributed.all_reduce(comm_tensor, group=group)
                    elif comm_type == CommType.SEND_OR_RECV:
                        if self.my_rank == comm_ranks[0]:
                            torch.distributed.send(comm_tensor, dst=comm_ranks[1], group=group)
                        else:
                            torch.distributed.recv(comm_tensor, src=comm_ranks[0], group=group)
                    elif comm_type == CommType.SEND_AND_RECV:
                        if self.my_rank == comm_ranks[0]:
                            other_rank = comm_ranks[1]
                        else:
                            other_rank = comm_ranks[0]
                        send_op = torch.distributed.P2POp(torch.distributed.isend,
                                                          comm_tensor,
                                                          other_rank,
                                                          group=group)
                        recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                                          comm_tensor2,
                                                          other_rank,
                                                          group=group)
                        reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
                        for req in reqs:
                            req.wait()
                    else:
                        raise NotImplementedError

                    s = time.time()
                    torch.cuda.synchronize()
                    if i > 0:
                        comm_times.append(time.time() - s)
                comm_time = sum(comm_times) / len(comm_times)
                comm_time_tensor.data[0] = comm_time

        torch.distributed.gather(comm_time_tensor, tensor_list, dst=0)
        if self.my_rank != 0:
            return None
        comm_time_per_rank = defaultdict(int)
        for rank in range(self.world_size):
            if rank in rank_to_comm_ranks:
                comm_time_per_rank[rank] = tensor_list[rank].item()
        return comm_time_per_rank
