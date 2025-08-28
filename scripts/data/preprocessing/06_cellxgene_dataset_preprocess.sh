#!/bin/bash

set -a
source .env
set +a

python /home/angli/baseline/tool/integrated_preprocess.py \
    --input_dir $DEEPSC_CELLXGENE_PRIMARY_DATA_PATH \
    --output_dir $DATASET_BEFORE_SHUFFEL \
    --gene_map_path $DEEPSC_GENE_MAP_PATH \
    --use_gpu \
    --num_processes 8
