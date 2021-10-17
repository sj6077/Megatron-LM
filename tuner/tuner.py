"""Tuner to find optimal parallelization methods"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
import time

import torch

from tuner.estimator import Config, Estimator # pylint: disable=no-name-in-module

tuner_logger = logging.getLogger('tuner')

def get_explore_target(min_value=1, max_value=float("inf"), reverse=False):
    """Get exploration targets from min_value to max_value while increasing the value twice"""
    res = []
    value = min_value
    while value <= max_value:
        res.append(value)
        value *= 2
    if reverse:
        res.reverse()
    return res

class Tuner:  # pylint: disable=too-few-public-methods
    """Find optimal parallelization methods for given model and resources"""
    def __init__(self, world_size, num_gpus_per_node, global_batch_size,
                 result_filename):
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.result_filename = result_filename
        self.estimator = Estimator(world_size, num_gpus_per_node)
        self.device_memory = torch.cuda.get_device_properties(0).total_memory

    def tune(self):  # pylint: disable=too-many-locals
        """Explore all configurations and record it to the file"""
        node_rank = int(os.environ['NODE_RANK'])
        if node_rank != 0:
            self.estimator.init_comm_helper_procs()
            return

        oom_configs = []
        all_result = {}

        increase_tensor_model_parallel_size = True
        best_iter_time = float("inf")
        start = time.time()
        for tensor_model_parallel_size in get_explore_target(max_value=self.world_size):
            if not increase_tensor_model_parallel_size:
                continue

            increase_micro_batch_size = True
            for micro_batch_size in get_explore_target(max_value=self.global_batch_size):
                if not increase_micro_batch_size:
                    continue

                decrease_pipeline_model_parallel_size = True
                for pipeline_model_parallel_size in get_explore_target(
                        max_value=self.world_size//tensor_model_parallel_size, reverse=True):
                    data_parallel_size = self.world_size // \
                            tensor_model_parallel_size // \
                            pipeline_model_parallel_size
                    if not decrease_pipeline_model_parallel_size or \
                            self.global_batch_size % (data_parallel_size * micro_batch_size) != 0:
                        continue

                    config = Config(micro_batch_size=micro_batch_size,
                                    global_batch_size=self.global_batch_size,
                                    dp=data_parallel_size,
                                    mp=tensor_model_parallel_size,
                                    pp=pipeline_model_parallel_size)

                    is_oom = False
                    gpu_memory = self.estimator.get_max_gpu_memory(config)
                    is_oom = gpu_memory > self.device_memory
                    if not is_oom:
                        iter_time, tensor_model_parallel_time, \
                                pipeline_model_parallel_time, data_parallel_time = \
                                self.estimator.get_iter_time(config)
                        is_oom |= iter_time == 0

                    decrease_pipeline_model_parallel_size = not is_oom
                    increase_micro_batch_size = not is_oom
                    if is_oom:
                        res_str = f"{config}: OOM"
                        oom_configs.append(config)
                        tuner_logger.info(res_str)
                        continue

                    res_str = f"{config}: memory - {gpu_memory} MB, iter time - {iter_time} ms, " \
                              f"mp time - {tensor_model_parallel_time} ms, " \
                              f"pp time - {pipeline_model_parallel_time} ms, " \
                              f"dp time - {data_parallel_time} ms"
                    tuner_logger.info(res_str)

                    all_result[config] = (gpu_memory, iter_time, tensor_model_parallel_time,
                                          pipeline_model_parallel_time, data_parallel_time)
                    best_iter_time = min(iter_time, best_iter_time)
                    increase_tensor_model_parallel_size = len(all_result) == 0 or \
                            tensor_model_parallel_time < best_iter_time * 0.5
                    decrease_pipeline_model_parallel_size = len(all_result) == 0 or \
                            data_parallel_time < best_iter_time * 0.5
                    tuner_logger.debug(f"increase mp: {increase_tensor_model_parallel_size}")
                    tuner_logger.debug(f"decrease pp: {decrease_pipeline_model_parallel_size}")
        end = time.time()

        sorted_result = sorted(all_result.items(), key=lambda item: item[1][1])
        with open(self.result_filename, 'w', encoding='utf-8') as log_file:
            log_file.write(f"tuning-time {(int(end - start))} sec\n")
            for config, (gpu_memory, iter_time, tensor_model_parallel_time, \
                    pipeline_model_parallel_time, data_parallel_time) in sorted_result:
                log_file.write(f"{config}: memory - {gpu_memory} MB, iter time - {iter_time} ms, ")
                log_file.write(f"mp time - {tensor_model_parallel_time} ms, ")
                log_file.write(f"pp time - {pipeline_model_parallel_time} ms, ")
                log_file.write(f"dp time - {data_parallel_time} ms\n")
            for config in oom_configs:
                log_file.write(f"{config}: OOM\n")
        self.estimator.terminate()
