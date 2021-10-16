"""Execute tuner for gpt pretraining"""
import os

from tuner.tuner import Tuner

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
    node_rank = int(os.environ['NODE_RANK'])
    global_batch_size = int(os.environ['GLOBAL_BATCH_SIZE'])
    file_name = os.environ['TUNER_LOG_FILE']
    tuner = Tuner(world_size, num_gpus_per_node, global_batch_size, file_name)
    if node_rank == 0:
        tuner.tune()
