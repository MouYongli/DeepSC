#!/bin/bash

set -a
source .env
set +a

NUM_PROCESSES=1  # 默认并行进程数


# 运行 Python 下载脚本
python -m deepsc.data.download.tripleca.download_3ca \
    --output_path "$DATA_PATH_3CA" \
    --log_path "$DEEPSC_LOGS_ROOT" \
    --num_processes "$NUM_PROCESSES"
