#!/bin/bash
set -a
source .env
set +a

python -m deepsc.data.preprocessing.gene_name_normalization \
