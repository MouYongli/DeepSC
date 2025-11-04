#!/usr/bin/zsh

# Fine-tuning script with task configuration selection
# Usage: ./run_finetune.sh [task_name]
# Example:
#   ./run_finetune.sh                     # Uses default (cell_type_annotation)
#   ./run_finetune.sh cell_type_annotation

# Default task configuration
TASK_NAME="${1:-cell_type_annotation}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=64

NUM_GPUS=3
MASTER_PORT=12620

echo "Starting fine-tuning with task: $TASK_NAME"
echo "Loading task configuration from: configs/finetune/tasks/$TASK_NAME.yaml"

# Run training with task configuration override
# The defaults list in finetune.yaml will be overridden to use the specified task config
PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.finetune \
  tasks=$TASK_NAME \
  task_type=$TASK_NAME
