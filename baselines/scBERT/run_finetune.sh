#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate scbert

# Create save directory if not exists
mkdir -p /home/angli/scBERT/save

# Generate log filename with timestamp
LOG_FILE="/home/angli/scBERT/save/training_$(date +%Y%m%d_%H%M%S).log"

# Use GPU 1, 2, 3 for training (3 GPUs total)
export CUDA_VISIBLE_DEVICES=1,2,3

# Run finetune with torchrun (3 GPUs) and save log
echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

torchrun --nproc_per_node=3 finetune_annotation.py 2>&1 | tee -a "$LOG_FILE"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Training finished at $(date)" | tee -a "$LOG_FILE"

# For other GPU configurations:
# Use GPU 0,1,2,3 (4 GPUs): export CUDA_VISIBLE_DEVICES=0,1,2,3 && torchrun --nproc_per_node=4 finetune_annotation.py
# Use GPU 0 only (1 GPU): export CUDA_VISIBLE_DEVICES=0 && torchrun --nproc_per_node=1 finetune_annotation.py