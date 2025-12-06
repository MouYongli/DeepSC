#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.filter_primary_data \
    --num_workers 32 \
    --cellxgene_dir $DATA_PATH_CELLXGENE \
    --output_base $DEEPSC_CELLXGENE_PRIMARY_DATA_PATH
