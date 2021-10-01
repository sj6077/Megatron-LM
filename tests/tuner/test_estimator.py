from unittest.mock import patch

import os
import logging
import re
import signal
import subprocess
import sys
import time

from tuner.estimator import Config, Estimator

logging.basicConfig(level=logging.DEBUG)

def get_data_config_str():
    data_dir = os.environ.get('DATA_DIR', '/cmsdata/ssd1/cmslab/gpt2_data')
    config_str = f"--data-path {data_dir}/my-gpt2_text_document "\
                 f"--vocab-file {data_dir}/gpt2-vocab.json "\
                 f"--merge-file {data_dir}/gpt2-merges.txt"
    return config_str

def get_dist_config_str(mp, pp, dp):
    host_list = os.environ.get('HOST_LIST', 'localhost').split(',')
    num_gpus_per_host = int(os.environ.get('NUM_GPUS_PER_HOST', '1'))
    master_port = os.environ.get('MASTER_PORT', '7000')

    assert len(host_list) * num_gpus_per_host >= mp * pp * dp
    num_gpus_per_host = mp * pp * dp // len(host_list)
    assert len(host_list) * num_gpus_per_host == mp * pp * dp
    distributed_args = f"--nproc_per_node {num_gpus_per_host} "\
                       f"--nnodes {len(host_list)} "\
                       f"--master_addr {host_list[0]} "\
                       f"--master_port {master_port}"

    config_str = f"--tensor-model-parallel-size {mp} "\
                 f"--pipeline-model-parallel-size {pp} "
    return config_str, distributed_args

def get_model_config_str(num_layers, hidden_size, num_attention_heads, seq_length, fp16):
    config_str = f"--num-layers {num_layers} "\
                 f"--hidden-size {hidden_size} "\
                 f"--num-attention-heads {num_attention_heads} "\
                 f"--seq-length {seq_length} "\
                 f"--max-position-embeddings {seq_length} "\
                 f"--lr 0.00015 "\
                 f"--min-lr 1.0e-5 "\
                 f"--lr-decay-style cosine "\
                 f"--clip-grad 1.0 "\
                 f"--lr-warmup-fraction .01 "\
                 f"--activations-checkpoint-method uniform " \
                 f"--train-iters 100 "\
                 f"--eval-interval 1000000 "\
                 f"--eval-iters 0 " \
                 f"--log-interval 1"
    if fp16:
        config_str += " --fp16"
    return config_str

def get_345m_model(fp16, num_layers=4):
    return get_model_config_str(num_layers=num_layers,
                                hidden_size=1024,
                                num_attention_heads=16,
                                seq_length=1024,
                                fp16=fp16)

def get_env_str():
    use_master_env = os.environ.get('USE_MASTER_ENV', '1')

    env_str=''
    if use_master_env == '1':
        env_str = f"PATH={os.environ.get('PATH', '')} "\
                  f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')} "\
                  f"CUDA_HOME={os.environ.get('CUDA_HOME', '')} "
        venv_path = os.environ.get('VIRTUAL_ENV', '')
        if venv_path:
            env_str += f'source {venv_path}/bin/activate;'
    return env_str

def get_mock_argv_and_set_env(model_config_str, world_size):
    mock_argv = ['test'] + ['--micro-batch-size', '1', '--global-batch-size', '1']
    mock_argv += model_config_str.split(' ')
    mock_argv += get_data_config_str().split(' ')

    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(world_size)
    return mock_argv

def get_actual_iter_time_and_memory(model_config_str, dist_config_str, distributed_args, global_batch_size, micro_batch_size):
    host_list = os.environ.get('HOST_LIST', 'localhost').split(',')
    subprocs = []

    env_str = get_env_str()
    ssh_user = os.environ.get('SSH_USER', os.environ['USER'])
    data_config_str = get_data_config_str()
    for i, host in enumerate(host_list):
        distributed_args += f" --node_rank {i}"
        full_cmd = f"ssh {ssh_user}@{host} "\
                   f"{env_str} python -m torch.distributed.launch {distributed_args} " \
                   f"{os.path.abspath(os.getcwd())}/pretrain_gpt.py " \
                   f"{data_config_str} " \
                   f"{model_config_str} " \
                   f"{dist_config_str} " \
                   f"--global-batch-size {global_batch_size} "\
                   f"--micro-batch-size {micro_batch_size}"
        subproc = subprocess.Popen([full_cmd],
                                   shell=True,
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   preexec_fn=os.setsid,
                                   bufsize=0)
        subprocs.append(subproc)

    mem_allocs = []
    iter_time_ms = 0
    while subprocs[0].poll() is None:
        output = subprocs[0].stdout.readline()
        output = output.strip().decode('utf-8')
        m = re.search("iteration[ \t]+(\d+)", output)
        if m:
            iteration = int(m[1])
            if iteration > 1:
                m = re.search("elapsed time per iteration \(ms\):[ \t]+(\d+\.\d*)?",
                              output)
                iter_time_ms = float(m[1])
                if len(mem_allocs) > 1:
                    break

        m = re.search("max allocated:[ \t]+(\d+\.\d*)?", output)
        if m:
            mem_alloc = float(m[1])
            mem_allocs.append(mem_alloc)
    
    for proc in subprocs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    return iter_time_ms, mem_allocs[-1]

def test_single_gpu_iter_time():
    model_config_str = get_345m_model(fp16=True, num_layers=8)
    dist_config_str, distributed_args = get_dist_config_str(1, 1, 1)

    world_size = 1
    mock_argv = get_mock_argv_and_set_env(model_config_str, world_size)
    result = []
    with patch.object(sys, 'argv', mock_argv):
        with Estimator(world_size, model='gpt') as estimator:
            for mb in [1, 2, 4]:
                for gb_increase in [1, 2, 4]:
                    gb = mb * gb_increase

                    config = Config(micro_batch_size=mb, global_batch_size=gb)
                    estimated_iter_time = estimator.get_iter_time(config)

                    actual_iter_time, _ = get_actual_iter_time_and_memory(
                            model_config_str,
                            dist_config_str,
                            distributed_args,
                            global_batch_size=gb,
                            micro_batch_size=mb)
                    result.append((config, actual_iter_time, estimated_iter_time))

    for config, actual_iter_time, estimated_iter_time in result:
            if actual_iter_time == 0:
                print(f"OOM for {config}")
                continue

            print(f"Iter time(ms) for {config}:",
                  f"actual-{actual_iter_time},", 
                  f"estimation-{estimated_iter_time}",
                  f"diff-{abs(actual_iter_time-estimated_iter_time)}",
                  f"error-{abs(actual_iter_time-estimated_iter_time)/actual_iter_time}")

def test_single_gpu_memory():
    model_config_str = get_345m_model(fp16=True, num_layers=8)
    dist_config_str, distributed_args = get_dist_config_str(1, 1, 1)

    world_size = 1
    mock_argv = get_mock_argv_and_set_env(model_config_str, world_size)
    result = []
    with patch.object(sys, 'argv', mock_argv):
        with Estimator(world_size, model='gpt') as estimator:
            for mb in [1, 2, 4]:
                for gb_increase in [1, 2, 4]:
                    gb = mb * gb_increase

                    _, actual_gpu_memory = get_actual_iter_time_and_memory(
                            model_config_str,
                            dist_config_str,
                            distributed_args,
                            global_batch_size=gb,
                            micro_batch_size=mb)

                    config = Config(micro_batch_size=mb, global_batch_size=gb)
                    estimated_gpu_memory = estimator.get_max_gpu_memory(config)
                    
                    result.append((config, actual_gpu_memory, estimated_gpu_memory))

    for config, actual_gpu_memory, estimated_gpu_memory in result:
        print(f"GPU memory for {config}:",
              f"actual-{actual_gpu_memory}",
              f"estimaton-{estimated_gpu_memory}",
              f"diff-{abs(actual_gpu_memory-estimated_gpu_memory)}",
              f"error-{abs(actual_gpu_memory-estimated_gpu_memory)/actual_gpu_memory}")
