#!/usr/bin/zsh

# Fine-tuning script
# Task configuration is specified in configs/finetune/finetune.yaml
# To change task: edit the 'defaults' and 'task_type' settings in finetune.yaml

# Load environment variables from .env file
set -a
source .env
set +a

# GPU configuration
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=64

NUM_GPUS=3
MASTER_PORT=12620

# Run training
PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.finetune
