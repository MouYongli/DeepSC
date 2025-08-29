#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.cellxgene_data_preprocess \
    --input_dir $DEEPSC_CELLXGENE_PRIMARY_DATA_PATH \
    --output_dir $DATASET_BEFORE_SHUFFEL \
    --gene_map_path $DEEPSC_GENE_MAP_PATH \
    --num_processes 8
