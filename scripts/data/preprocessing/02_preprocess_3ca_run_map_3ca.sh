#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.preprocess_datasets_3ca \
  --input_dir $DATA_PATH_3CA \
  --output_dir $MAPPED_DATA_PATH_3CA \
  --gene_map_path $DEEPSC_GENE_MAP_PATH
