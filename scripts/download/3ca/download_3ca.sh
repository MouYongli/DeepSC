#!/bin/bash

CSV_PATH="./data_info.csv"
OUTPUT_PATH="/home/angli/DeepSC/data/3ac/raw"
LOG_PATH="./logs"
NUM_FILES=131
NUM_PROCESSES=8  # 默认并行进程数

# 创建目录（如果不存在）
mkdir -p "$OUTPUT_PATH"
mkdir -p "$LOG_PATH"

# 运行 Python 下载脚本
python download_3ca.py \
    --csv_path "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --log_path "$LOG_PATH" \
    --num_files "$NUM_FILES" \
    --num_processes "$NUM_PROCESSES"
