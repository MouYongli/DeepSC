#!/bin/bash

cd "$(dirname "$0")/../../.."

set -a
source .env
set +a

python -m scripts.download.tripleca.merge_and_filter_dataset \
    --dataset_root_path "$DATA_PATH_3CA"
