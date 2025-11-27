#!/bin/bash
# Reference Mapping Script for All DeepSC Checkpoints
# This script runs reference mapping for all checkpoints and saves results to separate files

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepsc

# Base paths
CHECKPOINT_DIR="/home/angli/DeepSC/results/pretraining_1120"
REF_H5AD="/home/angli/scGPT/data/annotation_pancreas/demo_train.h5ad"
QUERY_H5AD="/home/angli/scGPT/data/annotation_pancreas/demo_test.h5ad"
GENE_MAP="/home/angli/DeepSC/scripts/data/preprocessing/gene_map.csv"
CELL_TYPE_KEY="Celltype"
BASE_OUTPUT_DIR="/home/angli/DeepSC/results/reference_mapping_all_checkpoints"

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Array of checkpoint numbers
CHECKPOINTS=(1 2 3 4 5 6 7 8 9 10 11)

echo "=========================================="
echo "Running Reference Mapping for All Checkpoints"
echo "=========================================="
echo "Base output directory: $BASE_OUTPUT_DIR"
echo ""

# Loop through all checkpoints
for i in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/DeepSC_${i}_0.ckpt"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoint_${i}"
    LOG_FILE="${OUTPUT_DIR}/results.log"

    echo "=========================================="
    echo "Processing Checkpoint ${i}"
    echo "=========================================="
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo "Output directory: $OUTPUT_DIR"

    # Create checkpoint-specific output directory
    mkdir -p "$OUTPUT_DIR"

    # Run reference mapping and save output to log file
    python examples/reference_mapping.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --ref_h5ad "$REF_H5AD" \
        --query_h5ad "$QUERY_H5AD" \
        --gene_map "$GENE_MAP" \
        --cell_type_key "$CELL_TYPE_KEY" \
        --visualize \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "Checkpoint ${i} completed. Results saved to:"
    echo "  - UMAP: ${OUTPUT_DIR}/reference_mapping_umap.png"
    echo "  - Log: ${LOG_FILE}"
    echo ""
done

echo "=========================================="
echo "All checkpoints processed!"
echo "=========================================="
echo "Results saved in: $BASE_OUTPUT_DIR"
echo ""
echo "Summary of outputs:"
for i in "${CHECKPOINTS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoint_${i}"
    echo "Checkpoint ${i}:"
    echo "  - Directory: $OUTPUT_DIR"
    if [ -f "${OUTPUT_DIR}/results.log" ]; then
        # Extract accuracy from log file
        ACCURACY=$(grep -oP "Accuracy: \K[0-9.]+(?=\s|$)" "${OUTPUT_DIR}/results.log" | head -1)
        if [ ! -z "$ACCURACY" ]; then
            echo "  - Accuracy: $ACCURACY"
        fi
    fi
done
