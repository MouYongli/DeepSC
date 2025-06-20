#!/usr/bin/env bash

GPUS=(1 2 3)
LEARNING_RATES=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

NUM_GPUS=${#GPUS[@]}
NUM_LR=${#LEARNING_RATES[@]}

for ((idx=0; idx<NUM_LR; idx++)); do
  lr=${LEARNING_RATES[$idx]}
  gpu_idx=$(( idx % NUM_GPUS ))
  gpu=${GPUS[$gpu_idx]}
  echo "Running lr=$lr on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH=src python -m deepsc.pretrain.pretrain learning_rate=$lr &
  # 控制并发数为GPU数
  while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
    sleep 1
  done
done

wait
