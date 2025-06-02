#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --mem=48G
#SBATCH -p cpu

cd "$(dirname "$0")/../../.."

set -a
source .env
set +a

source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH_CELLXGENE)

echo "downloading ${query_name}"

/hpcwork/p0021245/DeepSC/scripts/download/cellxgene/download_partition.sh ${query_name} ${INDEX_PATH_CELLXGENE} ${DATA_PATH_CELLXGENE}
