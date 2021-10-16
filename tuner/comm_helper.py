"""Receive commuinication request from estimator and send back the communication time"""
# pylint: disable=logging-fstring-interpolation
from collections import defaultdict
from functools import wraps
from enum import IntEnum
import errno
import logging
import os
import signal
import time
from typing import Dict, List

import torch

NUM_AVERAGE = 5
supported_dtypes = [torch.int16, torch.int32, torch.int64, torch.float16, torch.float32]
tuner_logger = logging.getLogger('tuner')

class CommType(IntEnum):
    """Enum for communication type"""
    END = 0
    BROADCAST = 1
    ALLREDUCE = 2
    SEND_OR_RECV = 3
    SEND_AND_RECV = 4

def encode_comm_type(comm_type: CommType):
    """Encode communication type as torch.Tensor"""
    return torch.IntTensor([int(comm_type)])

def decode_comm_type(tensor: torch.Tensor):
    """Decode torch.Tensor to get communication type"""
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
    """Decode torch.Tensor to get communication ranks"""
    rank_to_comm_ranks = {}
    val = tensor.tolist()
    for rank in range(world_size):
        comm_ranks = [r for r in val[rank] if r != -1]
        if comm_ranks:
            rank_to_comm_ranks[rank] = comm_ranks
    return rank_to_comm_ranks

def encode_tensor_dtype_and_shape(dtype: torch.Tensor.dtype, tensor_shape: List[int]):
    """Encode tensor dtype and shape as torch.Tensor"""
    dtype_id = supported_dtypes.index(dtype)
    assert len(tensor_shape) < 9
    val = [0] * 10
    val[0] = dtype_id
    val[1] = len(tensor_shape)
    for i, size in enumerate(tensor_shape):
        val[i+2] = size
    return torch.IntTensor(val)

def decode_tensor_dtype_and_shape(tensor: torch.Tensor):
    """Decode torch.Tensor to get tensor dtype and shape"""
    val = tensor.tolist()
    dtype_id = val[0]
    dtype = supported_dtypes[dtype_id]
    dim = val[1]
    return dtype, [val[i+2] for i in range(dim)]

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Decorator to raise timeout error"""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator

class CommHelper:  # pylint: disable=too-many-instance-attributes
    """A helper class that can estimate nccl communication time."""

    def __init__(self, event, req_queue=None, resp_queue=None):
        assert 'MASTER_ADDR' in os.environ
        assert 'MASTER_PORT' in os.environ

        self.event = event
        self.req_queue = req_queue
        self.resp_queue = resp_queue

        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        self.local_rank = local_rank
        torch.cuda.set_device(self.local_rank)

        torch.distributed.init_process_group(
                backend='gloo',
                world_size=world_size,
                rank=rank)
        torch.distributed.barrier()
        tuner_logger.info(f"initialize communication helper for rank{rank}")
        self.world_size = world_size
        self.my_rank = rank

        self.comm_type_tensor = torch.IntTensor([0])
        self.comm_ranks_tensor = torch.IntTensor([[-1] * world_size] * world_size)
        self.tensor_shape_tensor = torch.IntTensor([0] * 10)

        self.comm_group = {}
        self.comm_time_tensor = torch.FloatTensor([0.0])
        if self.my_rank == 0:
            self.tensor_list = [torch.FloatTensor([0.0])] * self.world_size
        else:
            self.tensor_list = None

        if self.my_rank == 0:
            self.resp_queue.put(0)

    @timeout(30)
    def _send_msg_to_all_ranks(self, comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape):
        assert self.my_rank == 0

        encoded_comm_type = encode_comm_type(comm_type)
        if comm_type != CommType.END:
            encoded_comm_ranks = encode_comm_ranks(self.world_size, rank_to_comm_ranks)
            encoded_tensor_shape = encode_tensor_dtype_and_shape(
                    tensor_dtype, tensor_shape)

        torch.distributed.broadcast(encoded_comm_type, src=0)
        if comm_type == CommType.END:
            return
        torch.distributed.broadcast(encoded_comm_ranks, src=0)
        torch.distributed.broadcast(encoded_tensor_shape, src=0)

    def run_msg_handler_for_rank0(self):
        """Run message handler for master(rank0)
           It receives communication requests using multiprocessing queue
           and broadcasts the request to all other ranks.
        """
        tuner_logger.debug("run_msg_handler_for_rank0")
        while not self.event.is_set():
            tuner_logger.debug("wait request queue's input")
            comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape = \
                    self.req_queue.get()
            try:
                tuner_logger.debug(f"received req: {comm_type}, {tensor_dtype}, {tensor_shape}")
                self._send_msg_to_all_ranks(
                    comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape)
                tuner_logger.debug("send_msg_to_all_ranks")
                if comm_type == CommType.END:
                    break

                comm_time_per_rank = self._handle_comm(
                        comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape)
                if comm_time_per_rank is None:
                    self.event.set()
                    break
                tuner_logger.debug(f"response queue put: {comm_time_per_rank}")
                self.resp_queue.put(comm_time_per_rank)
            except Exception as error:
                self.event.set()
                raise error

    def run_msg_handler(self):
        """Run message handler for non-rank0 GPUs."""
        while not self.event.is_set():
            try:
                torch.distributed.broadcast(self.comm_type_tensor, src=0)
                comm_type = decode_comm_type(self.comm_type_tensor)
                if comm_type == CommType.END:
                    break
                torch.distributed.broadcast(self.comm_ranks_tensor, src=0)
                rank_to_comm_ranks = decode_comm_ranks(self.world_size, self.comm_ranks_tensor)
                torch.distributed.broadcast(self.tensor_shape_tensor, src=0)
                tensor_dtype, tensor_shape = decode_tensor_dtype_and_shape(self.tensor_shape_tensor)
                tuner_logger.debug(f"receive msg for {comm_type}, {tensor_dtype}, {tensor_shape}")
                self._handle_comm(comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape)
            except Exception as error:
                self.event.set()
                raise error

    def _get_comm_group(self, comm_ranks):
        key = tuple(comm_ranks)
        if key not in self.comm_group:
            new_group = torch.distributed.new_group(ranks=comm_ranks, backend='nccl')
            self.comm_group[key] = new_group
        return self.comm_group[key]

    def _get_comm_time(self, comm_type, comm_tensor, comm_ranks, group):
        """Get communication time by actual execution"""
        if comm_type == CommType.SEND_AND_RECV:
            comm_tensor2 = comm_tensor.clone()

        comm_times = []
        for _ in range(NUM_AVERAGE + 1):
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

            start = time.time()
            torch.cuda.synchronize()
            comm_times.append(time.time() - start)
        del comm_times[0]
        comm_time = sum(comm_times) / len(comm_times)
        return comm_time

    @timeout(60)
    def _handle_comm(self, comm_type, rank_to_comm_ranks, tensor_dtype, tensor_shape):
        """Execute given communication and send the communication time to rank0"""

        torch.distributed.barrier()
        torch.cuda.synchronize()

        groups = {}
        for rank in range(self.world_size):
            if rank in rank_to_comm_ranks:
                # group must be initialized across all ranks
                group = self._get_comm_group(rank_to_comm_ranks[rank])
                groups[rank] = group

        if self.my_rank in rank_to_comm_ranks:
            if tensor_dtype in [torch.int16, torch.int32, torch.int64]:
                comm_tensor = torch.randint(low=0, high=100, size=tuple(tensor_shape)).cuda()
            else:
                comm_tensor = torch.randn(tensor_shape, dtype=tensor_dtype).cuda()

            comm_ranks = rank_to_comm_ranks[self.my_rank]
            group = groups[self.my_rank]
            if len(comm_ranks) > 1:
                comm_time = self._get_comm_time(comm_type, comm_tensor, comm_ranks, group)
                self.comm_time_tensor.data[0] = comm_time

        torch.distributed.gather(self.comm_time_tensor, self.tensor_list, dst=0)
        if self.my_rank != 0:
            return None
        comm_time_per_rank = defaultdict(int)
        for rank in range(self.world_size):
            if rank in rank_to_comm_ranks:
                comm_time_per_rank[rank] = self.tensor_list[rank].item()
        return comm_time_per_rank
