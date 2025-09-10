# Scripts for Model Pre-training

This directory contains scripts for pretraining models using different computational environments.

## 0. Environment Setup

First, we need make sure that the conda environment is activated:

```bash
conda activate deepsc
```

## 1. Single Machine Training

### 1.1 Local GPU Training

For training on a single machine with multiple GPUs, use [`./scripts/pretrain/run_deepsc.sh`](./scripts/pretrain/run_deepsc.sh).

This script is configured to:
- Use GPUs 1 and 2 (CUDA_VISIBLE_DEVICES=1,2)
- Run with 2 GPUs using torchrun
- Set master port to 12620

Here, as the example, we use run the pretraining script for DeepSC model.

```bash
bash ./scripts/pretrain/run_deepsc.sh
```


**Requirements:**
- At least 1 GPUs available

## 2. HPC Cluster Training

### 2.1 SLURM Distributed Training

For distributed training on HPC clusters with SLURM, use [`./scripts/pretrain/run_deepsc_hpc.sh`](./scripts/pretrain/run_deepsc_hpc.sh).

This script is configured to:
- Use 2 nodes with 4 tasks per node
- 4 CPUs per task and 4 GPUs per node
- 256GB memory allocation
- 24-hour time limit
- Automatic port configuration to avoid conflicts

```bash
sbatch ./scripts/pretrain/run_deepsc_hpc.sh
```

### 2.2 Log Management

The HPC script automatically creates a `logs` directory and saves job outputs:
- Standard output: `logs/{job_name}_{job_id}.out`
- Error output: `logs/{job_name}_{job_id}.err`

## 3. Training Details

Both scripts execute the DeepSC pretraining module:

```bash
python -m deepsc.pretrain.pretrain
```

### 3.1 Environment Variables

The scripts set up the following key environment variables:
- `MASTER_ADDR`: Master node address (auto-configured in HPC)
- `MASTER_PORT`: Communication port (auto-configured to avoid conflicts)
- `WORLD_SIZE`: Total number of processes
- `CUDA_VISIBLE_DEVICES`: GPU visibility configuration

## 4. Prerequisites

Before running these scripts, ensure that:

1. **Data is preprocessed**: Complete the data preprocessing steps in [`./scripts/data/preprocessing/`](./scripts/data/preprocessing/)
2. **hyper-parameters of model are correctly configured in[`./configs/pretrain/model/deepsc.yaml`](./configs/pretrain/model/deepsc.yaml)**

   The key hyperparameters defined in this configuration file include:

   **Model Architecture:**
   - `embedding_dim: 256` - Dimension of gene embeddings
   - `num_genes: 34683` - Total number of genes in the dataset
   - `num_layers: 10` - Number of transformer layers
   - `num_heads: 8` - Number of attention heads
   - `num_layers_ffn: 2` - Number of layers of FFN in one layer of transformer
   - `gene_embedding_participate_til_layer: 3` - number of layers which gene embeddings participate

   **Regularization:**
   - `attn_dropout: 0.1` - Dropout rate for attention layers
   - `ffn_dropout: 0.1` - Dropout rate for feedforward layers
   - `alpha: 0.3` - Alpha parameter for loss weighting

   **Training Configuration:**
   - `num_bins: ${num_bin}` - Number of bins for value discretization
   - `mask_layer_start: 10` - Starting layer for masking strategy
   - `use_moe_regressor: ${use_moe_regressor}` - Enable MoE regressor for regression task of pretraining
   **Loss Functions (configurable):**
   - `enable_l0: ${enable_l0}` - Enable L0 regularization
   - `enable_mse: ${enable_mse}` - Enable mean squared error loss
   - `enable_ce: ${enable_ce}` - Enable cross-entropy loss

   **Mixture of Experts (MoE) in Feed Forward Network:**
   - `n_moe_layers: 4` - Number of MoE layers
   - `moe_inter_dim: 512` - Intermediate dimension in MoE
   - `n_routed_experts: 2` - Number of routed experts
   - `n_activated_experts: 2` - Number of activated experts
   - `n_shared_experts: 1` - Number of shared experts
   - `use_moe_ffn: ${use_moe_ffn}` - Enable MoE in feedforward networks

   **Advanced Features:**
   - `use_M_matrix: ${use_M_matrix}` - Enable M matrix
   - `fused: False` - Whether to fuse expression and gene embedding at the expression attention layer
3. **Arguments of pretrianing are correctly configured in[`./configs/pretrain/pretrain.yaml`](./configs/pretrain/pretrain.yaml)**


## TODO

- [ ] `src/deepsc/pretrain/pretrain.py`文件中的main函数可以配置的参数列表，并添加到`./scripts/pretrain/run_deepsc.sh`文件中。

```bash
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain \
  --model deepsc
```

- [ ] `configs/pretrain/pretrain.yaml`文件中的`defaults`中的dataset是不是有必要？
