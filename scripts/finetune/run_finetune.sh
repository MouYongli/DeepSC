#!/usr/bin/zsh

# Fine-tuning script with task type selection
# Usage: ./run_finetune.sh [task_type]
# Example: ./run_finetune.sh cell_type_annotation

# Default task type
TASK_TYPE="${1:-cell_type_annotation}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=64

NUM_GPUS=3
MASTER_PORT=12620

echo "Starting fine-tuning with task type: $TASK_TYPE"

# Run training with task type override
PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.finetune \
  task_type=$TASK_TYPE
