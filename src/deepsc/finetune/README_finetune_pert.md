# Perturbation Prediction Fine-tuning

This directory contains a self-contained implementation of perturbation prediction fine-tuning, migrated from scGPT examples.

## File Structure

```
/home/angli/DeepSC/
├── src/deepsc/
│   ├── models/scgpt_pert/          # Self-contained model package
│   │   ├── __init__.py             # Package exports
│   │   ├── generation_model.py     # TransformerGenerator model
│   │   ├── losses.py               # Loss functions
│   │   ├── tokenizer.py            # GeneVocab and tokenization
│   │   └── utils.py                # Utility functions
│   └── finetune/
│       ├── finetune_perturbation.py         # Main training script
│       ├── run_finetune_perturbation.sh     # Bash launcher
│       └── README_finetune_pert.md          # This file
```

## Features

- **Self-contained**: All scGPT dependencies are extracted and placed in `models/scgpt_pert/`
- **No external scGPT**: Does not depend on the original scGPT library
- **Complete model implementation**: TransformerGenerator with all necessary components
- **Full training pipeline**: Train, validation, evaluation, and visualization

## Dependencies

Required Python packages (should already be installed):
- torch
- numpy
- torchtext
- torch_geometric
- gears (for perturbation data)
- matplotlib
- scipy
- anndata
- scanpy (if needed by gears)

## Usage

### 1. Basic Usage

```bash
cd /home/angli/DeepSC/src/deepsc/finetune
./run_finetune_perturbation.sh
```

### 2. Custom Configuration

Edit the configuration variables in `finetune_perturbation.py`:

```python
# Dataset selection
data_name = "norman"  # or "adamson"
split = "simulation"

# Model hyperparameters
embsize = 512
d_hid = 512
nlayers = 12
nhead = 8
dropout = 0

# Training settings
lr = 1e-4
batch_size = 64
epochs = 10
early_stop = 10

# Pretrained model
load_model = "../save/scGPT_human"  # or None for training from scratch
```

### 3. Running from Python

```python
# Make sure PYTHONPATH includes the parent directory
import sys
sys.path.insert(0, '/home/angli/DeepSC/src')

# Import and use the components
from deepsc.models.scgpt_pert import (
    TransformerGenerator,
    masked_mse_loss,
    GeneVocab,
    set_seed,
)

# Run the training
exec(open('finetune_perturbation.py').read())
```

## Model Components

### 1. TransformerGenerator (`generation_model.py`)

The main model class with:
- Gene encoder with layer normalization
- Continuous value encoder
- Perturbation encoder
- Transformer encoder (with optional FlashAttention support)
- Affine expression decoder
- Prediction head for perturbed expression values

### 2. Loss Functions (`losses.py`)

- `masked_mse_loss`: Masked MSE for expression prediction
- `criterion_neg_log_bernoulli`: Negative log-likelihood for binary classification
- `masked_relative_error`: Relative error metric

### 3. Tokenizer (`tokenizer.py`)

- `GeneVocab`: Gene vocabulary management
- `tokenize_batch`: Tokenize gene expression data
- `pad_batch`: Pad sequences to uniform length
- `tokenize_and_pad_batch`: Combined tokenization and padding

### 4. Utilities (`utils.py`)

- `set_seed`: Set random seeds for reproducibility
- `map_raw_id_to_vocab_id`: Map gene IDs to vocabulary indices
- `compute_perturbation_metrics`: Compute evaluation metrics
- `load_pretrained`: Load pretrained model weights
- `add_file_handler`: Add file logging

## Output

The training script saves:
- `save/dev_perturb_{dataset}-{timestamp}/`
  - `best_model.pt`: Best model checkpoint
  - `run.log`: Training logs
  - `test_metrics.json`: Test set metrics
  - `{perturbation}.png`: Visualization plots (if enabled)
  - `prediction_vs_target_scatter_*.png`: Scatter plots (if `plot_scatter=True`)

## Metrics

The evaluation computes:
- **Pearson correlation**: Overall expression correlation
- **Pearson correlation (DE genes)**: Correlation for differentially expressed genes
- **Pearson delta**: Correlation of expression changes
- **Pearson delta (DE genes)**: Correlation of changes in DE genes

Additional analysis:
- Subgroup analysis by perturbation type
- Non-dropout gene analysis
- Deeper analysis with gears utilities

## Notes

1. **Data Location**: The script expects GEARS data in `./data` directory
2. **Pretrained Model**: If using pretrained weights, ensure the path is correct
3. **GPU**: Automatically uses CUDA if available, falls back to CPU
4. **FlashAttention**: If `flash-attn` is not installed, automatically falls back to standard PyTorch attention
5. **Memory**: Adjust `batch_size` and `max_seq_len` based on available GPU memory

## Troubleshooting

### Import Errors

If you see import errors, make sure PYTHONPATH is set correctly:

```bash
export PYTHONPATH="${PYTHONPATH}:/home/angli/DeepSC/src"
```

### CUDA Out of Memory

Reduce batch size or max sequence length:

```python
batch_size = 32  # Reduced from 64
max_seq_len = 1024  # Reduced from 1536
```

### Data Not Found

Ensure GEARS data is downloaded:

```python
from gears import PertData
pert_data = PertData("./data")
pert_data.load(data_name="norman")  # This will download if not present
```

## Migration Notes

This implementation was migrated from:
- **Original**: `/home/angli/old_tasks/scGPT/examples/finetune_perturbation.py`
- **New location**: `/home/angli/DeepSC/src/deepsc/finetune/finetune_perturbation.py`
- **Models**: `/home/angli/DeepSC/src/deepsc/models/scgpt_pert/`

All scGPT dependencies have been extracted and made self-contained, so the original scGPT library is not needed.
