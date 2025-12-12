#!/bin/bash

INPUT_FILE="/home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad"

python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/analyze_obs_columns.py \
    --input "$INPUT_FILE"
