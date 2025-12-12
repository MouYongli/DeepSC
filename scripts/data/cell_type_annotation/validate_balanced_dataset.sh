#!/bin/bash

INPUT_FILE="/home/angli/baseline/DeepSC/data/cell_type_annotation/balanced_dataset/balanced_dataset_kidney_100.h5ad"
EXPECTED_N_CELLS=100

python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/validate_balanced_dataset.py \
    --input "$INPUT_FILE" \
    --expected_n_cells $EXPECTED_N_CELLS
