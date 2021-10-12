import argparse
from collections import defaultdict
import os
import logging
import multiprocessing
import re
import signal
import subprocess
import sys
import time

from tuner.estimator import Config, Estimator

logging.basicConfig(level=logging.DEBUG)

def get_data_config_str(data_dir):
    config_str = f"--data-path {data_dir}/my-gpt2_text_document "\
                 f"--vocab-file {data_dir}/gpt2-vocab.json "\
                 f"--merge-file {data_dir}/gpt2-merges.txt"
    return config_str

def get_dist_config_str(mp, pp, dp):
    host_list = os.environ.get('HOST_LIST', 'localhost').split(',')
    num_gpus_per_node = int(os.environ.get('NUM_GPUS_PER_NODE', '1'))
    master_port = '7000'

    assert len(host_list) * num_gpus_per_node >= mp * pp * dp
    num_gpus_per_node = mp * pp * dp // len(host_list)
    assert len(host_list) * num_gpus_per_node == mp * pp * dp
    distributed_args = f"--nproc_per_node {num_gpus_per_node} "\
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

def get_7b_model(fp16, num_layers):
    # 7.5b model
    return get_model_config_str(num_layers=num_layers,
                                hidden_size=4096,
                                num_attention_heads=32,
                                seq_length=2048,
                                fp16=fp16)

def get_39b_model(fp16, num_layers):
    # 39b model
    return get_model_config_str(num_layers=num_layers,
                                hidden_size=8192,
                                num_attention_heads=64,
                                seq_length=2048,
                                fp16=fp16)

def get_env_str():
    use_master_env = os.environ['USE_MASTER_ENV']

    env_str=''
    if use_master_env == '1':
        env_str = f"PATH={os.environ.get('PATH', '')} "\
                  f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')} "\
                  f"CUDA_HOME={os.environ.get('CUDA_HOME', '')} "
        venv_path = os.environ.get('VIRTUAL_ENV', '')
        if venv_path:
            env_str += f'source {venv_path}/bin/activate;'
    return env_str

def get_mock_argv_and_set_master_env(data_dir, model_config_str, world_size):
    mock_argv = ['test'] + ['--micro-batch-size', '1', '--global-batch-size', '1']
    mock_argv += ['--tensor-model-parallel-size', str(world_size)]
    mock_argv += model_config_str.split(' ')
    mock_argv += get_data_config_str(data_dir).split(' ')

    host_list = os.environ.get('HOST_LIST', 'localhost').split(',')
    num_gpus_per_node = int(os.environ.get('NUM_GPUS_PER_NODE', '1'))
    master_addr = host_list[0]
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(world_size)
    return mock_argv

def get_actual_iter_time_and_memory(data_dir, model_config_str, dist_config_str,
                                    distributed_args, global_batch_size, micro_batch_size):
    host_list = os.environ.get('HOST_LIST', 'localhost').split(',')
    subprocs = []

    env_str = get_env_str()
    ssh_user = os.environ.get('SSH_USER', os.environ['USER'])
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    data_config_str = get_data_config_str(data_dir)
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
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   preexec_fn=os.setsid,
                                   bufsize=0)
        subprocs.append(subproc)

    world_size = len(host_list) * num_gpus_per_node
    mem_allocs = []
    iter_time_ms = []
    is_oom = False
    skipped_iter = 1
    while subprocs[0].poll() is None:
        output = subprocs[0].stdout.readline()
        output = output.strip().decode('utf-8')
        print(output)

        if "CUDA out of memory" in output:
            is_oom = True
            break

        reg_exp = "iteration[ \t]+(\d+).*?elapsed time per iteration \(ms\):[ \t]+(\d+\.\d*)"
        reg_exp += ".*?number of skipped iterations: +(\d+)"
        m = re.search(reg_exp, output)
        if m:
            iteration = int(m[1])
            iter_time = float(m[2])
            skipped_iter = int(m[3])
            if iteration > 1 and skipped_iter == 0:
                iter_time_ms.append(iter_time)
        elif skipped_iter == 0:
            m = re.search("max allocated:[ \t]+(\d+\.\d*)?", output)
            if m:
                print("correct output", output)
                mem_alloc = float(m[1])
                mem_allocs.append(mem_alloc)

        if len(iter_time_ms) > 3 and len(mem_allocs) > 3 * world_size:
            break

    for proc in subprocs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    if is_oom:
        return 0, 0
    return max(iter_time_ms), max(mem_allocs)

def run_comm_helper(world_size):
    host_list = os.environ['HOST_LIST'].split(',')
    subprocs = []

    env_str = get_env_str()
    ssh_user = os.environ['SSH_USER']
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    env_str += f" MASTER_ADDR={host_list[0]}"
    env_str += f" MASTER_PORT={os.environ['MASTER_PORT']}"
    env_str += f" WORLD_SIZE={world_size}"
    rank = 0
    for i, host in enumerate(host_list):
        for local_rank in range(num_gpus_per_node):
            if rank != 0:
                proc_env_str = f"{env_str} RANK={rank} LOCAL_RANK={local_rank}"
                full_cmd = f"ssh {ssh_user}@{host} "\
                           f"{proc_env_str} python {os.getcwd()}/benchmark/tuner/run_comm_helper.py"
                subproc = subprocess.Popen([full_cmd],
                                           shell=True)
                subprocs.append(subproc)
            rank += 1
            if rank == world_size:
                break
    return subprocs

def estimator_run(data_dir, model, world_size, configs_to_test, env, queue):
    os.environ = env
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    mock_argv = get_mock_argv_and_set_master_env(data_dir, model, world_size)
    run_comm_helper(world_size)
    setattr(sys, 'argv', mock_argv)
    try:
        with Estimator(world_size, min(num_gpus_per_node, world_size)) as estimator:
            for config in configs_to_test:
                assert config.mp * config.dp * config.pp == world_size

                estimated_iter_time = estimator.get_iter_time(config)
                if estimated_iter_time == 0: # OOM
                    estimated_gpu_memory = 0
                else:
                    estimated_gpu_memory = estimator.get_max_gpu_memory(config)

                queue.put((estimated_iter_time, estimated_gpu_memory))
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            for _ in configs_to_test:
                queue.put((0, 0))
        else:
            raise e

def get_estimated_iter_time_and_memory(data_dir, model, world_size, configs_to_test):
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    child_env = os.environ.copy()
    proc = ctx.Process(
            target=estimator_run,
            args=(data_dir, model, world_size, configs_to_test, child_env, queue,))
    proc.start()
    result = {}
    for config in configs_to_test:
        result[config] = queue.get()
    proc.join()
    return result

def run_estimator_benchmark(data_dir, model, configs_to_test):
    estimation_results = {}
    configs_to_test_per_world_size = defaultdict(list)
    for config in configs_to_test:
        world_size = config.mp * config.dp * config.pp
        configs_to_test_per_world_size[world_size].append(config)
    
    for world_size, test_configs in configs_to_test_per_world_size.items():
        estimation_results.update(get_estimated_iter_time_and_memory(
            data_dir, model, world_size, test_configs))

    actual_results = []
    for config in configs_to_test:
        dist_config_str, distributed_args = get_dist_config_str(
                mp=config.mp, pp=config.pp, dp=config.dp)
        actual_iter_time, actual_gpu_memory = \
                get_actual_iter_time_and_memory(
                        data_dir,
                        model,
                        dist_config_str,
                        distributed_args,
                        global_batch_size=config.global_batch_size,
                        micro_batch_size=config.micro_batch_size)
        actual_results.append((actual_iter_time, actual_gpu_memory))

    for config, actual in zip(configs_to_test, actual_results):
        estimated_iter_time, estimated_gpu_memory = estimation_results[config]
        actual_iter_time, actual_gpu_memory = actual

        print(config)

        if actual_iter_time == actual_gpu_memory == 0:
            print("OOM in actual execution:",
                  f"estimated-iter-time(ms)-{estimated_iter_time}",
                  f"esitmated-gpu-memory(MB)-{estimated_gpu_memory}")
            continue

        print(f"iter time(ms) :",
              f"actual-{actual_iter_time},", 
              f"estimation-{estimated_iter_time}",
              f"diff-{abs(actual_iter_time-estimated_iter_time)}",
              f"error-{abs(actual_iter_time-estimated_iter_time)/actual_iter_time}")

        print(f"gpu memory :",
              f"actual-{actual_gpu_memory}",
              f"estimaton-{estimated_gpu_memory}",
              f"diff-{abs(actual_gpu_memory-estimated_gpu_memory)}",
              f"error-{abs(actual_gpu_memory-estimated_gpu_memory)/actual_gpu_memory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimator benchmark argument')
    parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'])
    parser.add_argument('--benchmark-type', type=str,
                        choices=['single-gpu', 'vmp-single-machine', 'pmp-single-machine'])
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    if 'HOST_LIST' not in os.environ:
        os.environ['HOST_LIST'] = 'localhost'
    if 'SSH_USER' not in os.environ:
        os.environ['SSH_USER'] = os.environ['USER']
    if 'USE_MASTER_ENV' not in os.environ:
        os.environ['USE_MASTER_ENV'] = '1'
    if 'NUM_GPUS_PER_NODE' not in os.environ:
        os.environ['NUM_GPUS_PER_NODE'] = '8'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '8000'

    if args.model == 'small':
        model = get_345m_model(fp16=args.fp16, num_layers=args.num_layers)
    elif args.model == 'medium':
        model = get_7b_model(fp16=args.fp16, num_layers=args.num_layers)
    elif args.model == 'large':
        model = get_39b_model(fp16=args.fp16, num_layers=args.num_layers)
    else:
        raise NotImplementedError

    if args.benchmark_type == 'single-gpu':
        configs_to_test = [Config(micro_batch_size=1, global_batch_size=1),
                           Config(micro_batch_size=2, global_batch_size=4),
                           Config(micro_batch_size=4, global_batch_size=12)]
    elif args.benchmark_type == 'vmp-single-machine':
        configs_to_test = [Config(micro_batch_size=1, global_batch_size=2, mp=1),
                           Config(micro_batch_size=1, global_batch_size=2, mp=2),
                           Config(micro_batch_size=4, global_batch_size=8, mp=4),
                           Config(micro_batch_size=4, global_batch_size=8, mp=8)]
    elif args.benchmark_type == 'pmp-single-machine':
        configs_to_test = [Config(micro_batch_size=1, global_batch_size=2, pp=1),
                           Config(micro_batch_size=1, global_batch_size=4, pp=2),
                           Config(micro_batch_size=2, global_batch_size=16, pp=4),
                           Config(micro_batch_size=2, global_batch_size=16, pp=8)]
    else:
        raise NotImplementedError

    run_estimator_benchmark(args.data_dir, model, configs_to_test)
