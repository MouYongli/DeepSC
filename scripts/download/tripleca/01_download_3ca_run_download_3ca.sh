#!/bin/bash

set -a
source .env
set +a

NUM_FILES=131
NUM_PROCESSES=1  # 默认并行进程数

# 创建目录（如果不存在）
mkdir -p "$DATA_PATH_3CA"
mkdir -p "$LOG_PATH"

# 运行 Python 下载脚本
python -m src.deepsc.data.download.tripleca.download_3ca \
    --output_path "$DATA_PATH_3CA" \
    --log_path "$LOG_PATH" \
    --num_files "$NUM_FILES" \
    --num_processes "$NUM_PROCESSES"