#!/bin/bash
#SBATCH -A p0021245
#SBATCH --job-name=scbert_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "source /home/rn260358/miniforge3/etc/profile.d/conda.sh"
source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew
echo "conda activate scbertnew"

# 使用作业ID来生成唯一端口，避免端口冲突
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 10000))
#显示出所有主机节点|取第一个
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONPATH=/hpcwork/p0021245/testdeepsc/DeepSC/src


echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "PYTHONPATH=$PYTHONPATH"
echo "NCCL_DEBUG=$NCCL_DEBUG"

# 创建日志目录
mkdir -p logs

# 启动分布式训练
srun python -m deepsc.pretrain.pretrain


# Use the following command to overwrite the parameters
# srun python -m deepsc.pretrain.pretrain \
#     model=deepsc \
#     enable_mse_loss=False \
#     enable_huber_loss=True \
#     seed=44 \
#     num_device=4
#     num_nodes=4
