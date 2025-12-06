#!/bin/bash
set -a
source .env
set +a

python -m deepsc.data.preprocessing.gene_name_normalization \
    --hgnc_database_path $DEEPSC_HGNC_DATABASE \
    --output_path $INTERMEDIATE_ARTIFACTS_TEMP \
    --gene_map_path $DEEPSC_GENE_MAP_PATH \
    --tripleca_path $DATA_PATH_3CA \
    --cellxgene_path $DATA_PATH_CELLXGENE
