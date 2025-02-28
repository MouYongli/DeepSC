#!/bin/bash

DATASET_PATH="../../../data/3ac/raw"

# 运行 Python 下载脚本
python process_3ca.py \
    --dataset_path "$DATASET_PATH"
