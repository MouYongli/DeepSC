#!/bin/bash
N_CELLS=500
INPUT_FILE="/home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad"
OUTPUT_DIR="/home/angli/baseline/DeepSC/data/cell_type_annotation/balanced_dataset"
MIN_THRESHOLD=1000
python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/create_balanced_dataset.py \
    --n_cells $N_CELLS \
    --input "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --min_threshold "$MIN_THRESHOLD"
