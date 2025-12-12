# Dual Model Support for Perturbation Fine-tuning

## Overview

The `finetune_perturbation.py` script now supports both **scGPT** and **DeepSC** models for perturbation prediction tasks. You can easily switch between the two models by changing a single configuration variable.

## Quick Start

### 1. Select Model Type

Edit the `MODEL_TYPE` variable at the top of the script (line 36):

```python
MODEL_TYPE = "scgpt"  # Options: "scgpt" or "deepsc"
```

### 2. Run the Script

```bash
cd /home/angli/DeepSC
conda run -n deepsc bash -c "export PYTHONPATH='/home/angli/DeepSC/src:\$PYTHONPATH' && python src/deepsc/finetune/finetune_perturbation.py"
```

## Configuration Details

### scGPT Configuration

When `MODEL_TYPE = "scgpt"`:

**Model Settings:**
- Embedding dimension: 512
- Number of layers: 12
- Number of heads: 8
- Dropout: 0
- Fast transformer: Enabled

**Checkpoint:**
- Location: `../save/scGPT_human`
- Files: `best_model.pt`, `vocab.json`, `args.json`

**Vocabulary:**
- Format: JSON file with GeneVocab structure
- Special tokens: `<pad>`, `<cls>`, `<eoc>`

### DeepSC Configuration

When `MODEL_TYPE = "deepsc"`:

**Model Settings:**
- Embedding dimension: 256
- Number of layers: 10
- Number of heads: 8
- Number of bins: 51
- MoE layers: 4 (with 2/2 routed/activated experts)

**Checkpoint:**
- Location: `/home/angli/DeepSC/results/pretraining_1201/latest_checkpoint.ckpt`

**Vocabulary:**
- Format: CSV file (`gene_map_tp10k.csv`)
- Location: `/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv`
- Special feature: Multiple feature_names can map to the same gene ID
- Note: ID+1 corresponds to model weight indices (0 is padding)

## Key Differences

### Data Processing

**scGPT:**
- Uses continuous expression values directly
- Maps genes using `map_raw_id_to_vocab_id`
- No expression discretization

**DeepSC:**
- Discretizes expression values into bins
- Normalizes expression values
- Uses dual encoding: discrete + continuous
- Maps genes using `build_gene_ids_for_dataset`

### Model Forward Pass

**scGPT:**
```python
output_dict = model(
    gene_ids, values, pert_flags,
    src_key_padding_mask=mask,
    CLS=False, CCE=False, MVC=False, ECS=False
)
output = output_dict["mlm_output"]
```

**DeepSC:**
```python
regression_output, gene_emb, expr_emb = model(
    gene_ids=gene_ids,
    expression_bin=discrete_bins,
    normalized_expr=normalized_values,
    input_pert_flags=pert_flags
)
```

### Loss Function

**scGPT:**
- `masked_mse_loss` (custom implementation)
- Supports masked positions

**DeepSC:**
- Standard `nn.MSELoss()`
- No masking required

## Output Structure

Results are saved to: `./save/dev_perturb_{MODEL_TYPE}_{dataset}-{timestamp}/`

**Files created:**
- `best_model.pt` - Best model weights
- `run.log` - Training log with metrics
- `test_metrics.json` - Test evaluation metrics
- `{perturbation}.png` - Visualization plots (if enabled)

## Common Settings

Both models share these settings:

```python
# Data
data_name = "norman"              # Dataset: norman/adamson
split = "simulation"              # Split type
include_zero_gene = "all"         # Gene inclusion strategy
max_seq_len = 1536               # Maximum sequence length

# Training
batch_size = 64
epochs = 10
lr = 1e-4
early_stop = 10
schedule_interval = 1

# Hardware
device = "cuda" if available else "cpu"
amp = True                        # Mixed precision training
```

## Example Usage

### Train with scGPT

```python
# In finetune_perturbation.py
MODEL_TYPE = "scgpt"
data_name = "norman"
epochs = 10
```

### Train with DeepSC

```python
# In finetune_perturbation.py
MODEL_TYPE = "deepsc"
data_name = "norman"
epochs = 10
```

## Validation Metrics

Both models report the same metrics:

- **Pearson correlation**: Overall correlation
- **Pearson DE**: Correlation on differentially expressed genes
- **Pearson delta**: Correlation of expression changes
- **Pearson DE delta**: Correlation of DE gene changes

## Troubleshooting

### scGPT Issues

**Problem**: Checkpoint not found
```
FileNotFoundError: ../save/scGPT_human/best_model.pt
```

**Solution**: Ensure scGPT checkpoint is copied to the correct location:
```bash
mkdir -p /home/angli/save
cp -r /home/angli/old_tasks/scGPT/examples/save/scGPT_human /home/angli/save/
```

### DeepSC Issues

**Problem**: Vocabulary CSV not found
```
FileNotFoundError: gene_map_tp10k.csv
```

**Solution**: Update the `vocab_csv_path` variable to point to your vocabulary file.

**Problem**: Checkpoint loading fails
```
RuntimeError: Missing keys in state_dict
```

**Solution**: This is expected when fine-tuning. The script loads weights with `strict=False` to allow missing keys.

### Common Issues

**Problem**: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `batch_size` or `max_seq_len`:
```python
batch_size = 32  # Reduced from 64
max_seq_len = 1024  # Reduced from 1536
```

**Problem**: Module not found
```
ModuleNotFoundError: No module named 'deepsc'
```

**Solution**: Set PYTHONPATH correctly:
```bash
export PYTHONPATH="/home/angli/DeepSC/src:$PYTHONPATH"
```

## Implementation Details

### Code Structure

The script uses conditional logic throughout:

```python
if MODEL_TYPE == "scgpt":
    # scGPT-specific code
elif MODEL_TYPE == "deepsc":
    # DeepSC-specific code
```

**Modified Functions:**
1. **Imports**: Conditional imports based on MODEL_TYPE
2. **Vocabulary loading**: Different formats (JSON vs CSV)
3. **Model initialization**: Different architectures
4. **Training loop**: Different forward passes
5. **Evaluation**: Adapted for both models
6. **Prediction**: Unified interface for both models

### Gene ID Mapping

**scGPT approach:**
- Uses `map_raw_id_to_vocab_id` function
- Maps dataset gene indices to vocabulary IDs

**DeepSC approach:**
- Uses `build_gene_ids_for_dataset` function
- Handles multiple feature names per gene ID
- Accounts for padding offset (ID+1)

## Performance Tips

1. **Use all genes** (`include_zero_gene = "all"`) for better stability
2. **Enable AMP** (`amp = True`) for faster training
3. **Adjust learning rate**: 1e-4 for fine-tuning, 1e-3 for training from scratch
4. **Monitor validation metrics** to catch overfitting early
5. **Use larger batch sizes** (64-128) if GPU memory allows

## References

- scGPT paper: Nature Methods (2023)
- DeepSC paper: [Add reference]
- GEARS library: https://github.com/snap-stanford/GEARS
