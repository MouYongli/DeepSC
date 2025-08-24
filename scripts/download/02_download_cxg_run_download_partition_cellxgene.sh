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
source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew # 激活conda环境

# 不要忘记设置query
query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$QUERY_PATH_CELLXGENE")
index_dir="$INDEX_PATH_CELLXGENE"
output_dir="$DATA_PATH_CELLXGENE"

echo "downloading: ${query_name}"
echo "index path: ${index_dir}"
echo "dataset path: ${output_dir}"

MAX_PARTITION_SIZE=200000

total_num=$(wc -l "${index_dir}/${query_name}.idx" | awk '{ print $1 }')
total_partition=$((total_num / MAX_PARTITION_SIZE))

# 分iteration下载
for i in $(seq 0 $total_partition)
do
    echo "Downloading ${query_name} 的 partition ${i}/${total_partition}"
    python3 src/deepsc/data/download/cellxgene/download_partition.py \
        --query-name "${query_name}" \
        --index-dir "${index_dir}" \
        --output-dir "${output_dir}" \
        --partition-idx "${i}" \
        --max-partition-size "${MAX_PARTITION_SIZE}"
done
