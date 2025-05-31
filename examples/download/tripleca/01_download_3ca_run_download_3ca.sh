#!/bin/bash

OUTPUT_PATH="/home/angli/DeepSC/data/3ac/testraw"
LOG_PATH="./logs"
NUM_FILES=131
NUM_PROCESSES=8  # 默认并行进程数

# 创建目录（如果不存在）
mkdir -p "$OUTPUT_PATH"
mkdir -p "$LOG_PATH"

# 运行 Python 下载脚本
python -m scripts.download.tripleca.download_3ca \
    --output_path "$OUTPUT_PATH" \
    --log_path "$LOG_PATH" \
    --num_files "$NUM_FILES" \
    --num_processes "$NUM_PROCESSES"
