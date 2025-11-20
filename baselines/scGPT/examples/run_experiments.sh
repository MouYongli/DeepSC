#!/bin/bash

# Bash script to run 4 experiments with different datasets
# This script will run finetune_annotation.py with different dataset pairs

# Set GPU device to rank 1
export CUDA_VISIBLE_DEVICES=1

DATA_DIR="/home/angli/baseline/DeepSC/data/processed/baseline/scgpt"

# Array of dataset names (without _train/_val suffix)
datasets=("hPancreas" "myeloid" "segerstolpe" "zheng")

# Associative array for batch sizes
declare -A batch_sizes
batch_sizes["hPancreas"]=10
batch_sizes["myeloid"]=16
batch_sizes["segerstolpe"]=2
batch_sizes["zheng"]=16

echo "Starting fine-tuning experiments on $(date)"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
    echo "=========================================="
    echo "Running experiment with dataset: ${dataset}"
    echo "=========================================="

    TRAIN_DATA="${DATA_DIR}/${dataset}_train.h5ad"
    VAL_DATA="${DATA_DIR}/${dataset}_val.h5ad"
    BATCH_SIZE="${batch_sizes[$dataset]}"

    # Check if files exist
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "Error: Training data file not found: $TRAIN_DATA"
        continue
    fi

    if [ ! -f "$VAL_DATA" ]; then
        echo "Error: Validation data file not found: $VAL_DATA"
        continue
    fi

    echo "Training data: $TRAIN_DATA"
    echo "Validation data: $VAL_DATA"
    echo "Batch size: $BATCH_SIZE"
    echo "Starting at: $(date)"

    # Run the experiment
    python finetune_annotation.py \
        --adata_path "$TRAIN_DATA" \
        --adata_test_path "$VAL_DATA" \
        --batch_size "$BATCH_SIZE"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Experiment ${dataset} completed successfully at $(date)"
    else
        echo "Experiment ${dataset} failed with exit code $EXIT_CODE at $(date)"
    fi

    echo ""
done

echo "All experiments completed at $(date)"
