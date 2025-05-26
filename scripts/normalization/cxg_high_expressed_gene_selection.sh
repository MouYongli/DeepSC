#!/usr/bin/zsh

#SBATCH --job-name=gene_select
#SBATCH --array=0-6
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --partition=standard

source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew

organ=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /hpcwork/rn260358/Data/query_list.txt)

echo "processing organ: $organ"

python /hpcwork/rn260358/Data/high_expressed_gene_selection.py --organ "$organ"
