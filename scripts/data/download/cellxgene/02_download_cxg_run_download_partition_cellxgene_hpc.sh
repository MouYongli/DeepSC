#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH -p cpu

# 加载环境变量 path放这里面
set -a
source .env
set +a

# 激活 conda 环境
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate deepsc # 激活conda环境

QUERY_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$QUERY_PATH_CELLXGENE")
INDEX_DIR="$INDEX_PATH_CELLXGENE"
OUTPUT_DIR="$DATA_PATH_CELLXGENE"

echo "Downloading: ${QUERY_NAME}"
echo "Index path: ${INDEX_DIR}"
echo "Dataset path: ${OUTPUT_DIR}"

MAX_PARTITION_SIZE=200000

TOTAL_NUM=$(wc -l "${INDEX_DIR}/${QUERY_NAME}.idx" | awk '{ print $1 }')
TOTAL_PARTITION=$((TOTAL_NUM / MAX_PARTITION_SIZE))

for i in $(seq 0 $TOTAL_PARTITION)
do
    echo "Downloading ${QUERY_NAME} 的 partition ${i}/${TOTAL_PARTITION}"
    python -m deepsc.data.download.cellxgene.download_partition \
        --query-name ${QUERY_NAME} \
        --index-dir ${INDEX_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --partition-idx ${i} \
        --max-partition-size ${MAX_PARTITION_SIZE}
done
