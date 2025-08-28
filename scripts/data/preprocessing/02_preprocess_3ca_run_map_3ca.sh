#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.preprocess_datasets_3ca \
  --output_dir $MAPPED_DATA_PATH_3CA \
