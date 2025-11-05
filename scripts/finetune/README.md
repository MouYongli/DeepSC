# Fine-tuning Scripts

This directory contains scripts for fine-tuning the DeepSC model on downstream tasks, with a focus on cell type annotation.

## Directory Structure

```
scripts/finetune/
├── README.md                  # This file
├── run_cta_test.sh           # Cell type annotation testing script
└── run_finetune.sh           # Generic fine-tuning script

configs/finetune/
├── finetune.yaml              # Main fine-tuning configuration
├── model/
│   └── deepsc.yaml           # Model architecture configuration
└── tasks/
    └── cell_type_annotation.yaml  # Cell type annotation task configuration
```

---

## Cell Type Annotation Task

Fine-tune DeepSC model for automatic cell type annotation based on gene expression data.

### 1. Configure Dataset Paths

Edit the task configuration file: `configs/finetune/tasks/cell_type_annotation.yaml`

**Key configuration parameters:**

```yaml
# Dataset mode selection
seperated_train_eval_dataset: False  # True: use separate train/eval files
                                      # False: use single file with stratified split

# Dataset paths (using environment variables)
data_path: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/myeloid_merged.h5ad"
data_path_eval: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/human_pancreas/hPancreas_test_adata.h5ad"

# Dataset split ratio (only used when seperated_train_eval_dataset: False)
test_size: 0.2  # 20% for validation, 80% for training

# Data format settings
var_name_in_h5ad: "gene_name"        # Column name for gene names
obs_celltype_col: "MajorCluster"     # Column name for cell type labels

# Loss configuration
cls_loss_type: "per_bin"              # Options: "standard" or "per_bin"
enable_l0: False
enable_mse: False
enable_ce: False
```

**Environment Variables:**

The configuration uses environment variables defined in your `.env` file:
- `DEEPSC_DATA_ROOT`: Root directory for datasets
- `DEEPSC_RESULTS_ROOT`: Root directory for results and checkpoints

Make sure your `.env` file contains:
```bash
DEEPSC_DATA_ROOT=/path/to/your/data
DEEPSC_RESULTS_ROOT=/path/to/your/results
```

**Dataset mode options:**

- **Option 1: Single dataset with stratified split** (Current default)
  ```yaml
  seperated_train_eval_dataset: False
  data_path: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/myeloid_merged.h5ad"
  test_size: 0.2
  ```
  The data will be automatically split into train (80%) and validation (20%) sets using stratified sampling.

- **Option 2: Separate train/eval datasets**
  ```yaml
  seperated_train_eval_dataset: True
  data_path: "${oc.env:DEEPSC_DATA_ROOT}/train_data.h5ad"
  data_path_eval: "${oc.env:DEEPSC_DATA_ROOT}/eval_data.h5ad"
  ```

### 2. Configure Training Parameters

Edit the main configuration file: `configs/finetune/finetune.yaml`

**Key training parameters:**

```yaml
# Task selection
task_type: "cell_type_annotation"

# Pretrained model (using environment variable)
pretrained_model_path: "${oc.env:DEEPSC_RESULTS_ROOT}/latest_checkpoint.ckpt"
load_pretrained_model: True

# Fine-tuning mode
finetune_mode: "full"  # Options: "full" or "head_only"

# Path settings
csv_path: "${oc.env:DEEPSC_DATA_ROOT}/gene_map.csv"

# Training hyperparameters
learning_rate: 3e-4
epoch: 10
batch_size: 32
grad_acc: 20
seed: 42

# Model architecture
sequence_length: 1024
num_bin: 5
use_moe_ffn: True

# Distributed training
num_device: 3
num_nodes: 1

# Learning rate scheduler
warmup_ratio: 0.03
```

### 3. Training

**Prerequisites:**

Make sure you have:
1. A `.env` file in the project root with `DEEPSC_DATA_ROOT` and `DEEPSC_RESULTS_ROOT`
2. GPU configuration set in `run_finetune.sh` (default: GPUs 1,2,3)

**Usage:**

```bash
# Run training
./scripts/finetune/run_finetune.sh
```

**Script configuration** (`scripts/finetune/run_finetune.sh`):
```bash
export CUDA_VISIBLE_DEVICES=1,2,3  # GPUs to use
NUM_GPUS=3                         # Number of GPUs
MASTER_PORT=12620                  # Port for distributed training
```

**Training outputs:**

The model will save checkpoints to the directory specified in `save_dir` (default: `./results/cell_type_annotation/`):

- `best_model.pt` - Best model based on validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints (if configured)

Each checkpoint contains:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Cell type mappings (`type2id`, `id2type`, `cell_type_count`)
- Training metrics (`epoch`, `eval_loss`)

**Evaluation outputs:**

During training, evaluation metrics and visualizations are saved to `save_dir`:
- `classification_metrics_epoch_N.png` - Bar charts for recall, precision, F1, and class proportions
- `confusion_matrix_epoch_N.png` - Normalized confusion matrix heatmap
- `classification_metrics_epoch_N.csv` - Detailed metrics in CSV format

---

### 4. Testing

After training, evaluate your model on the test dataset.

**Step 1: Configure test settings**

Edit `configs/finetune/tasks/cell_type_annotation.yaml`:

```yaml
# TEST-ONLY CONFIGURATIONS
checkpoint_path: "${oc.env:DEEPSC_RESULTS_ROOT}/cell_type_annotation/best_model.pt"
test_save_dir: "${oc.env:DEEPSC_RESULTS_ROOT}/cell_type_annotation"
```

**Step 2: Run testing**

```bash
# Run testing
./scripts/finetune/run_cta_test.sh
```

**Script configuration** (`scripts/finetune/run_cta_test.sh`):
```bash
export CUDA_VISIBLE_DEVICES=1,2,3  # GPUs to use
NUM_GPUS=3                         # Number of GPUs
MASTER_PORT=12621                  # Port (different from training)
```

**Test outputs:**

Results are saved to timestamped subdirectories: `test_save_dir/test_YYYYMMDD_HHMMSS/`

Each test run generates:
1. **Classification metrics plot** (`classification_metrics_epoch_0.png`)
   - Recall by cell type
   - Precision by cell type
   - F1 score by cell type
   - Class proportion

2. **Confusion matrix** (`confusion_matrix_epoch_0.png`)
   - Normalized confusion matrix heatmap

3. **Metrics CSV** (`classification_metrics_epoch_0.csv`)
   - Cell Type, Recall, Precision, F1 Score, Support, Proportion

4. **Test summary** (`test_summary.txt`)
   - Checkpoint path
   - Test timestamp
   - Number of test samples and cell types
   - Overall metrics: Test Loss, Error Rate, Accuracy, Precision, Recall, Macro F1
   - List of cell types tested

---

## Examples

### Example 1: Quick Start with Single Dataset (Current Configuration)

**Step 1: Setup environment**
```bash
# .env file
DEEPSC_DATA_ROOT=/home/angli/baseline/DeepSC/data
DEEPSC_RESULTS_ROOT=/home/angli/baseline/DeepSC/results
```

**Step 2: Configure dataset**
```yaml
# configs/finetune/tasks/cell_type_annotation.yaml
seperated_train_eval_dataset: False
data_path: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/myeloid_merged.h5ad"
test_size: 0.2
obs_celltype_col: "MajorCluster"
```

**Step 3: Run training**
```bash
./scripts/finetune/run_finetune.sh
```

**Step 4: Test the best model**
```yaml
# configs/finetune/tasks/cell_type_annotation.yaml (already configured)
checkpoint_path: "${oc.env:DEEPSC_RESULTS_ROOT}/cell_type_annotation/best_model.pt"
```
```bash
./scripts/finetune/run_cta_test.sh
```

---

### Example 2: Training with Separate Train/Eval Datasets

**Step 1: Configure datasets**
```yaml
# configs/finetune/tasks/cell_type_annotation.yaml
seperated_train_eval_dataset: True
data_path: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/train_data.h5ad"
data_path_eval: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/eval_data.h5ad"
obs_celltype_col: "cell_type"
```

**Step 2: Configure training parameters (optional)**
```yaml
# configs/finetune/finetune.yaml
learning_rate: 1e-4
epoch: 20
batch_size: 64
```

**Step 3: Run training**
```bash
./scripts/finetune/run_finetune.sh
```

---


## Important Notes

### Cell Type Mapping Consistency

⚠️ **Critical:** The model's cell type ID mapping is determined during training and saved in the checkpoint. During testing:

1. The checkpoint's `type2id` mapping is loaded
2. Test data is filtered to only include cell types present in the training set
3. The same ID mapping is used to ensure consistency

This means:
- ✅ You can test on datasets with additional cell types (they'll be filtered out)
- ✅ You can test on datasets with fewer cell types (only common types are evaluated)
- ❌ Cell types not seen during training cannot be predicted

## Configuration File Reference

### Main Configuration (`configs/finetune/finetune.yaml`)

```yaml
defaults:
  - _self_
  - model: deepsc                    # Model architecture config
  - tasks: cell_type_annotation     # Task-specific config

task_type: "cell_type_annotation"   # Task selection

# Pretrained model (using environment variable)
pretrained_model_path: "${oc.env:DEEPSC_RESULTS_ROOT}/latest_checkpoint.ckpt"
load_pretrained_model: True

# Fine-tuning mode
finetune_mode: "full"               # "full" or "head_only"

# Paths (using environment variables)
csv_path: "${oc.env:DEEPSC_DATA_ROOT}/gene_map.csv"

# Model architecture
use_moe_ffn: True
sequence_length: 1024
num_bin: 5

# Training hyperparameters
seed: 42
learning_rate: 3e-4
epoch: 10
batch_size: 32
grad_acc: 20

# Distributed training
num_device: 3
num_nodes: 1

# Learning rate scheduler
warmup_ratio: 0.03
```

### Task Configuration (`configs/finetune/tasks/cell_type_annotation.yaml`)

```yaml
# Dataset settings
seperated_train_eval_dataset: False
data_path: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/myeloid_merged.h5ad"
data_path_eval: "${oc.env:DEEPSC_DATA_ROOT}/cell_type_annotation/human_pancreas/hPancreas_test_adata.h5ad"
var_name_in_h5ad: "gene_name"
obs_celltype_col: "MajorCluster"
test_size: 0.2

# Loss configuration
enable_l0: False
enable_mse: False
enable_ce: False
cls_loss_type: "per_bin"           # "standard" or "per_bin"

# TEST-ONLY settings
checkpoint_path: "${oc.env:DEEPSC_RESULTS_ROOT}/cell_type_annotation/best_model.pt"
test_save_dir: "${oc.env:DEEPSC_RESULTS_ROOT}/cell_type_annotation"
```

---
