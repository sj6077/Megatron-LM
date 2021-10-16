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
    num_gpus_per_node = int(os.environ.get('NUM_GPUS_PER_NODE', '1'))
    nnodes = max(1, int(mp * pp * dp / num_gpus_per_node))
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    distributed_args = f"--nproc_per_node {num_gpus_per_node} "\
                       f"--nnodes {nnodes} "\
                       f"--master_addr {master_addr} "\
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

    os.environ['WORLD_SIZE'] = str(world_size)
    return mock_argv

def get_actual_iter_time_and_memory(data_dir, model_config_str, dist_config_str,
                                    distributed_args, global_batch_size, micro_batch_size):
    env_str = get_env_str()
    ssh_user = os.environ.get('SSH_USER', os.environ['USER'])
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    data_config_str = get_data_config_str(data_dir)
    node_rank = int(os.environ['NODE_RANK'])
    distributed_args += f" --node_rank {node_rank}"
    full_cmd = f"python -m torch.distributed.launch {distributed_args} " \
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
    if node_rank != 0:
        subproc.wait()
        return

    world_size = int(os.environ['WORLD_SIZE'])
    mem_allocs = []
    iter_time_ms = []
    is_oom = False
    skipped_iter = 1
    while subproc.poll() is None:
        output = subproc.stdout.readline()
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

    try:
        os.killpg(os.getpgid(subproc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    if is_oom:
        return 0, 0
    del iter_time_ms[0] # optimizer initial time is included
    return max(iter_time_ms), max(mem_allocs)

def estimator_run(data_dir, model, world_size, configs_to_test):
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    node_rank = int(os.environ['NODE_RANK'])
    print("create estimator for node rank", node_rank)
    mock_argv = get_mock_argv_and_set_master_env(data_dir, model, world_size)
    setattr(sys, 'argv', mock_argv)

    result = {}
    estimator = Estimator(world_size, min(num_gpus_per_node, world_size))
    if node_rank == 0:
        for config in configs_to_test:
            assert config.mp * config.dp * config.pp == world_size

            estimated_iter_time, mp_time, pp_time, dp_time  = estimator.get_iter_time(config)
            if estimated_iter_time == 0: # OOM
                estimated_gpu_memory = 0
            else:
                estimated_gpu_memory = estimator.get_max_gpu_memory(config)
            result[config] = (estimated_iter_time, estimated_gpu_memory)
        estimator.terminate()
    return result

def get_estimated_iter_time_and_memory(data_dir, model, world_size, configs_to_test):
    #ctx = multiprocessing.get_context('spawn')
    #node_rank = int(os.environ['NODE_RANK'])
    #if node_rank == 0:
    result = estimator_run(data_dir, model, world_size, configs_to_test)
    #else:
    #    queue = ctx.Queue()
    #    child_env = os.environ.copy()
    #    proc = ctx.Process(
    #        target=estimator_run,
    #        args=(data_dir, model, world_size, configs_to_test, child_env))
    #    proc.start()
    #    proc.join()
    return result

def run_estimator_benchmark(data_dir, model, configs_to_test):
    estimation_results = {}
    configs_to_test_per_world_size = defaultdict(list)
    node_rank = int(os.environ['NODE_RANK'])
    for config in configs_to_test:
        world_size = config.mp * config.dp * config.pp
        configs_to_test_per_world_size[world_size].append(config)
    
    for world_size, test_configs in configs_to_test_per_world_size.items():
        result = get_estimated_iter_time_and_memory(
                 data_dir, model, world_size, test_configs)
        if result:
            estimation_results.update(result)

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
        if node_rank == 0:
            actual_results.append((actual_iter_time, actual_gpu_memory))

    if node_rank != 0:
        return

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
                        choices=['single-gpu', 'vmp-single-machine', 'pmp-single-machine',
                                 'vmp-and-pmp-single-machine', 'vmp-and-dp-single-machine'])
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    if 'SSH_USER' not in os.environ:
        os.environ['SSH_USER'] = os.environ['USER']
    if 'USE_MASTER_ENV' not in os.environ:
        os.environ['USE_MASTER_ENV'] = '1'
    if 'NUM_GPUS_PER_NODE' not in os.environ:
        os.environ['NUM_GPUS_PER_NODE'] = '8'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '8000'
    if 'NODE_RANK' not in os.environ:
        os.environ['NODE_RANK'] = '0'

    if args.model == 'small':
        model = get_345m_model(fp16=args.fp16, num_layers=args.num_layers)
    elif args.model == 'medium':
        model = get_7b_model(fp16=args.fp16, num_layers=args.num_layers)
    elif args.model == 'large':
        model = get_39b_model(fp16=args.fp16, num_layers=args.num_layers)
    else:
        raise NotImplementedError

    if args.benchmark_type == 'single-gpu':
        os.environ['WORLD_SIZE'] = '1'
        os.environ['NUM_GPUS_PER_NODE'] = '1'
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
    elif args.benchmark_type == 'vmp-and-pmp-single-machine':
        configs_to_test = [Config(micro_batch_size=1, global_batch_size=2, mp=1, pp=8),
                           Config(micro_batch_size=2, global_batch_size=4, mp=2, pp=4),
                           Config(micro_batch_size=4, global_batch_size=16, mp=4, pp=2),
                           Config(micro_batch_size=8, global_batch_size=16, mp=8, pp=1)]
    elif args.benchmark_type == 'vmp-and-dp-single-machine':
        configs_to_test = [Config(micro_batch_size=1, global_batch_size=16, mp=1, dp=8),
                           Config(micro_batch_size=1, global_batch_size=16, mp=2, dp=4),
                           Config(micro_batch_size=2, global_batch_size=16, mp=4, dp=2),
                           Config(micro_batch_size=2, global_batch_size=16, mp=8, dp=1)]
    else:
        raise NotImplementedError

    run_estimator_benchmark(args.data_dir, model, configs_to_test)
