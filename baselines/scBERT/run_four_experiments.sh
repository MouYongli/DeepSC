#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate scbert

# Create save directory if not exists
mkdir -p /home/angli/scBERT/save

# Use GPU 1, 2, 3 for training (3 GPUs total)
export CUDA_VISIBLE_DEVICES=1,2,3

echo "=========================================="
echo "Starting Four Experiments at $(date)"
echo "=========================================="

# ========== Experiment 1: Segerstolpe (COMMENTED OUT - Run later) ==========
# echo ""
# echo "=========================================="
# echo "Experiment 1: Segerstolpe Dataset"
# echo "=========================================="
# LOG_FILE_1="/home/angli/scBERT/save/training_segerstolpe_1111_new_$(date +%Y%m%d_%H%M%S).log"
# echo "Starting Experiment 1 at $(date)" | tee -a "$LOG_FILE_1"
# echo "Log file: $LOG_FILE_1" | tee -a "$LOG_FILE_1"
# echo "Dataset: segerstolpe_train.h5ad + segerstolpe_val.h5ad" | tee -a "$LOG_FILE_1"
# echo "Epochs: 8" | tee -a "$LOG_FILE_1"
# echo "Model name: segerstolpe_1111_new_" | tee -a "$LOG_FILE_1"
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE_1"
# echo "----------------------------------------" | tee -a "$LOG_FILE_1"
#
# torchrun --nproc_per_node=3 finetune_annotation.py \
#     --data_path /home/angli/scBERT/data/new_preprocessed/segerstolpe_train.h5ad \
#     --val_data_path /home/angli/scBERT/data/new_preprocessed/segerstolpe_val.h5ad \
#     --epoch 8 \
#     --model_name segerstolpe_1111_new_ \
#     2>&1 | tee -a "$LOG_FILE_1"
#
# echo "----------------------------------------" | tee -a "$LOG_FILE_1"
# echo "Experiment 1 finished at $(date)" | tee -a "$LOG_FILE_1"

# ========== Experiment 2: Myeloid ==========
echo ""
echo "=========================================="
echo "Experiment 2: Myeloid Dataset"
echo "=========================================="
LOG_FILE_2="/home/angli/scBERT/save/training_myeloid_1111_new_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Experiment 2 at $(date)" | tee -a "$LOG_FILE_2"
echo "Log file: $LOG_FILE_2" | tee -a "$LOG_FILE_2"
echo "Dataset: myeloid_train.h5ad + myeloid_val.h5ad" | tee -a "$LOG_FILE_2"
echo "Epochs: 8" | tee -a "$LOG_FILE_2"
echo "Model name: myeloid_1111_new_" | tee -a "$LOG_FILE_2"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE_2"
echo "----------------------------------------" | tee -a "$LOG_FILE_2"

torchrun --nproc_per_node=3 finetune_annotation.py \
    --data_path /home/angli/scBERT/data/new_preprocessed/myeloid_train.h5ad \
    --val_data_path /home/angli/scBERT/data/new_preprocessed/myeloid_val.h5ad \
    --epoch 8 \
    --model_name myeloid_1111_new_ \
    2>&1 | tee -a "$LOG_FILE_2"

echo "----------------------------------------" | tee -a "$LOG_FILE_2"
echo "Experiment 2 finished at $(date)" | tee -a "$LOG_FILE_2"

# ========== Experiment 3: Human Pancreas ==========
echo ""
echo "=========================================="
echo "Experiment 3: Human Pancreas Dataset"
echo "=========================================="
LOG_FILE_3="/home/angli/scBERT/save/training_hPancreas_1111_new_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Experiment 3 at $(date)" | tee -a "$LOG_FILE_3"
echo "Log file: $LOG_FILE_3" | tee -a "$LOG_FILE_3"
echo "Dataset: hPancreas_train.h5ad + hPancreas_val.h5ad" | tee -a "$LOG_FILE_3"
echo "Epochs: 8" | tee -a "$LOG_FILE_3"
echo "Model name: hPancreas_1111_new_" | tee -a "$LOG_FILE_3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE_3"
echo "----------------------------------------" | tee -a "$LOG_FILE_3"

torchrun --nproc_per_node=3 finetune_annotation.py \
    --data_path /home/angli/scBERT/data/new_preprocessed/hPancreas_train.h5ad \
    --val_data_path /home/angli/scBERT/data/new_preprocessed/hPancreas_val.h5ad \
    --epoch 8 \
    --model_name hPancreas_1111_new_ \
    2>&1 | tee -a "$LOG_FILE_3"

echo "----------------------------------------" | tee -a "$LOG_FILE_3"
echo "Experiment 3 finished at $(date)" | tee -a "$LOG_FILE_3"

# ========== Experiment 4: Zheng68K (COMMENTED OUT - Run later) ==========
# echo ""
# echo "=========================================="
# echo "Experiment 4: Zheng68K Dataset"
# echo "=========================================="
# LOG_FILE_4="/home/angli/scBERT/save/training_zheng_1111_new_$(date +%Y%m%d_%H%M%S).log"
# echo "Starting Experiment 4 at $(date)" | tee -a "$LOG_FILE_4"
# echo "Log file: $LOG_FILE_4" | tee -a "$LOG_FILE_4"
# echo "Dataset: zheng_train.h5ad + zheng_val.h5ad" | tee -a "$LOG_FILE_4"
# echo "Epochs: 8" | tee -a "$LOG_FILE_4"
# echo "Model name: zheng_1111_new_" | tee -a "$LOG_FILE_4"
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE_4"
# echo "----------------------------------------" | tee -a "$LOG_FILE_4"
#
# torchrun --nproc_per_node=3 finetune_annotation.py \
#     --data_path /home/angli/scBERT/data/new_preprocessed/zheng_train.h5ad \
#     --val_data_path /home/angli/scBERT/data/new_preprocessed/zheng_val.h5ad \
#     --epoch 8 \
#     --model_name zheng_1111_new_ \
#     2>&1 | tee -a "$LOG_FILE_4"
#
# echo "----------------------------------------" | tee -a "$LOG_FILE_4"
# echo "Experiment 4 finished at $(date)" | tee -a "$LOG_FILE_4"

# ========== Summary ==========
echo ""
echo "=========================================="
echo "Two Experiments Completed at $(date)"
echo "=========================================="
echo "Experiment 2 Log: $LOG_FILE_2"
echo "Experiment 3 Log: $LOG_FILE_3"
echo "(Experiment 1 Segerstolpe is commented out - run later)"
echo "(Experiment 4 Zheng68K is commented out - run later)"
echo "=========================================="
