#!/usr/bin/zsh

# 使用物理编号为 2 和 3 的 GP
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64

NUM_GPUS=1
MASTER_PORT=12620

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.grn_start
