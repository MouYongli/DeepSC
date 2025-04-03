#!/usr/bin/zsh 


#SBATCH -A p0021245
#SBATCH --time=15:00:00         
#SBATCH --job-name=LArn260finet  
#SBATCH --nodes=2         
#SBATCH --ntasks=8            
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8  
#SBATCH --output=stdout.txt   
#SBATCH --mem-per-cpu=12G
module load CUDA/11.8.0
module load Python/3.9.5 


source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew
module load Python/3.9.5 
module unload Python/3.9.5 
echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Tasks per node $SLURM_TASKS_PER_NODE"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID



srun python -u pretrain.py