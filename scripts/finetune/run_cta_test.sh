#!/usr/bin/zsh

# Cell Type Annotation Test Script
# This script runs testing on a trained cell type annotation model

# Load environment variables from .env file
set -a
source .env
set +a

# GPU settings - configure the GPUs you want to use
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=64

NUM_GPUS=3
MASTER_PORT=12621  # Different port from training to avoid conflicts

# Run the test script
PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.cta_test
