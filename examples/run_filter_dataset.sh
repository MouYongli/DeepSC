#!/bin/bash

DATASET_PATH="/home/angli/DeepSC/data/3ac/raw"

python -m scripts.download.tripleca.merge_and_filter_dataset \
    --dataset_root_path "$DATASET_PATH"
