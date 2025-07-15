#!/usr/bin/zsh

# 使用物理编号为 2 和 3 的 GPU
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=64

NUM_GPUS=3
MASTER_PORT=12626

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain
