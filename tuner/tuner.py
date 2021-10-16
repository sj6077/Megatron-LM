import os
import time

import torch

from tuner.estimator import Config, Estimator

class Tuner:
    def __init__(self, world_size, num_gpus_per_node, global_batch_size,
                 result_filename):
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.result_filename = result_filename
        self.estimator = Estimator(world_size, num_gpus_per_node)
        self.device_memory = torch.cuda.get_device_properties(0).total_memory

    def _get_explore_target(self, min_value=1, max_value=float("inf"), reverse=False):
        res = []
        value = min_value
        while value <= max_value:
            res.append(value)
            value *= 2
        if reverse:
            res.reverse()
        return res

    def tune(self):
        node_rank = int(os.environ['NODE_RANK'])
        if node_rank != 0:
            self.estimator.init_comm_helper_procs()
            return

        mp_to_explore = self._get_explore_target(
                max_value=self.world_size)
        mb_to_explore = self._get_explore_target(
                max_value=self.global_batch_size)

        oom_configs = []
        all_result = {}
        increase_mp = True
        best_iter_time = float("inf")
        s = time.time()
        for mp in mp_to_explore:
            if not increase_mp:
                continue

            increase_mb = True
            for mb in mb_to_explore:
                if not increase_mb:
                    continue

                decrease_pp = True 
                for pp in self._get_explore_target(
                        max_value=self.world_size//mp, reverse=True):
                    if not decrease_pp:
                        continue

                    dp = self.world_size // mp // pp
                    if self.global_batch_size % (dp, mb) != 0:
                        continue

                    config = Config(micro_batch_size=mb,
                                    global_batch_size=self.global_batch_size,
                                    dp=dp,
                                    mp=mp,
                                    pp=pp)

                    is_oom = False
                    gpu_memory = self.estimator.get_max_gpu_memory(config)
                    is_oom = gpu_memory > self.device_memory
                    if not is_oom:
                        iter_time, mp_time, pp_time, dp_time = self.estimator.get_iter_time(config)
                        is_oom |= iter_time == 0

                    if is_oom:
                        res_str = f"{config}: OOM"
                        oom_configs.append(config)
                        decrease_pp = False
                        increase_mb = False
                    else:
                        res_str = f"{config}: memory - {gpu_memory} MB, iter-time - {iter_time} ms"
                        res_str += f", mp-time - {mp_time} ms, pp-time - {pp_time} ms, dp-time - {dp_time} ms"
                        all_result[config] = (gpu_memory, iter_time, mp_time, pp_time, dp_time)
                        best_iter_time = min(iter_time, best_iter_time)
                        if len(all_result) > 0 and mp_time > best_iter_time * 0.5:
                            increase_mp = False
                        if len(all_result) > 0 and dp_time > best_iter_time * 0.5:
                            decrease_pp = False
                    
                    print(res_str, flush=True)
        e = time.time()

        sorted_result = sorted(all_result.items(), key=lambda item: item[1][1])
        with open(self.result_filename, 'w') as f:
            f.write(f"tuning-time {(int(e-s))} sec\n")
            for config, (gpu_memory, iter_time, mp_time, pp_time, dp_time) in sorted_result:
                f.write(f"{config}: memory - {gpu_memory} MB, iter-time - {iter_time} ms")
                f.write(f", mp-time - {mp_time} ms, pp-time - {pp_time} ms, dp-time - {dp_time} ms\n")
            for config in oom_configs:
                f.write(f"{config}: OOM\n")
        self.estimator.terminate()
