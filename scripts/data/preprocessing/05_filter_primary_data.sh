#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.filter_primary_data \
    --NUM_WORKERS 32 \
    --CELLXGENE_DIR $DATA_PATH_CELLXGENE \
    --OUTPUT_BASE $DEEPSC_CELLXGENE_PRIMARY_DATA_PATH
