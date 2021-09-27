#! /bin/bash

# Runs the "345M" parameter model

export MASTER_ADDR=localhost
export MASTER_PORT=8000
RANK=0
WORLD_SIZE=1

DATA_DIR=/cmsdata/ssd1/cmslab/gpt2_data
DATA_PATH=$DATA_DIR/my-gpt2_text_document


python estimator.py \
       --num-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file $DATA_DIR/gpt2-vocab.json \
       --merge-file $DATA_DIR/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
