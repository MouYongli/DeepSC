#!/usr/bin/zsh

# Use GPU/GPUs on one machine
# Here the example is using GPUs with physical numbers 2 and 3
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=64

NUM_GPUS=2
MASTER_PORT=12620

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain \
  --model deepsc
