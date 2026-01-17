# Perturbation Prediction Fine-tuning for DeepSC

This module implements perturbation prediction fine-tuning for the DeepSC model, based on scGPT's perturbation prediction workflow with DeepSC's gene alignment approach.

## Overview

The `perturbation_finetune.py` module provides a complete pipeline for:
- Fine-tuning DeepSC models on perturbation prediction tasks
- Gene alignment between dataset genes and model vocabulary
- Training with distributed support via Lightning Fabric
- Evaluation using perturbation-specific metrics
- Visualization of perturbation effects

## Key Features

### 1. Gene Alignment Workflow
Similar to `pp_new.py`, this implementation carefully handles gene alignment:
- Maps dataset genes to vocabulary IDs using `build_gene_ids_for_dataset`
- Handles genes not present in the vocabulary (assigns padding ID)
- Maintains `name2col` mapping for perturbation flag construction
- Logs gene matching statistics

### 2. scGPT-based Training Logic
Adopts proven strategies from scGPT:
- Constructs perturbation flags from GEARS data format
- Supports both "all" and "batch-wise" gene inclusion strategies
- Uses MSE loss for regression on expression values
- Implements gradient accumulation and mixed precision training

### 3. Evaluation and Metrics
- Computes perturbation-specific metrics (pearson correlation, etc.)
- Performs deeper analysis and non-dropout analysis
- Subgroup analysis for different perturbation types

### 4. Prediction and Visualization
- Predicts expression changes for arbitrary perturbations
- Visualizes top differentially expressed genes
- Generates publication-ready plots

## Usage

### Basic Usage

```python
from lightning.fabric import Fabric
from deepsc.finetune.perturbation_finetune import PerturbationPredictor
from deepsc.models.deepsc.model import DeepSCModel

# Setup
fabric = Fabric(accelerator="gpu", devices=1)
fabric.launch()

# Initialize model
model = DeepSCModel(...)

# Create predictor
predictor = PerturbationPredictor(
    args=args,
    fabric=fabric,
    model=model
)

# Train
predictor.train()

# Generate plots
predictor.plot_predictions()
```

### Using the Example Script

```bash
python examples/run_perturbation_finetune.py \
    --data_name norman \
    --data_path ./data \
    --csv_path /path/to/vocab.csv \
    --embedding_dim 512 \
    --num_layers 12 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --epoch 10 \
    --pretrained_model \
    --pretrained_model_path /path/to/checkpoint.pt \
    --devices 1 \
    --perts_to_plot "SAMD1+ZBTB1"
```

### Key Arguments

**Data Settings:**
- `--data_name`: Dataset name (norman, adamson, replogle_rpe1_essential)
- `--data_path`: Path to GEARS data directory
- `--split`: Data split (simulation, combo_seen0, etc.)
- `--csv_path`: Path to vocabulary CSV file

**Model Settings:**
- `--embedding_dim`: Embedding dimension (default: 512)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 8)

**Training Settings:**
- `--batch_size`: Training batch size (default: 64)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--epoch`: Number of epochs (default: 10)
- `--grad_acc`: Gradient accumulation steps (default: 1)
- `--early_stop`: Early stopping patience (default: 10)

**Data Processing:**
- `--include_zero_gene`: How to handle zero genes (all/batch-wise)
- `--data_length`: Maximum sequence length (default: 1536)

## Architecture Details

### Data Flow

1. **Data Loading (GEARS format)**
   ```
   batch_data.x: (batch_size * n_genes, 2)
   batch_data.y: (batch_size, n_genes)
   batch_data.pert: list of perturbation strings
   batch_data.de_idx: differentially expressed gene indices
   ```

2. **Gene Selection**
   - All genes (`include_zero_gene='all'`)
   - OR batch-wise non-zero genes (`include_zero_gene='batch-wise'`)
   - Limit to `data_length` if too many genes

3. **Gene ID Mapping**
   ```python
   input_gene_ids (raw) → gene_ids (vocab mapping) → mapped_input_gene_ids
   ```

4. **Expression Processing**
   - Discretize expression values into bins
   - Normalize expression values
   - Construct perturbation flags (1 for perturbed genes, 0 otherwise)

5. **Model Forward Pass**
   ```python
   regression_output, gene_emb, expr_emb = model(
       gene_ids=mapped_input_gene_ids,
       expression_bin=discrete_input_bins,
       normalized_expr=input_values,
       input_pert_flags=input_pert_flags,
   )
   ```

6. **Loss Computation**
   - MSE loss between predicted and target expression values
   - Only computed on selected genes

### Gene Alignment

The gene alignment follows this workflow:

1. **Load Vocabulary**: Build vocab from CSV using `build_vocab_from_csv`
2. **Get Dataset Genes**: Extract from `pert_data.adata.var["gene_name"]`
3. **Build Mapping**: Use `build_gene_ids_for_dataset` to map genes to vocab IDs
4. **Create name2col**: Map gene names to column indices in data
5. **Validate**: Check which genes are matched vs. unmatched

Example:
```python
vocab_size: 20000
dataset_genes: 5000
matched_genes: 4800 / 5000 (96%)
```

### Perturbation Flags

Perturbation flags indicate which genes are perturbed:

```python
# For perturbation "SAMD1+ZBTB1"
pert_flags[batch_idx, gene_idx_SAMD1] = 1
pert_flags[batch_idx, gene_idx_ZBTB1] = 1
# All other genes = 0
```

## Comparison with scGPT and pp_new.py

| Aspect | scGPT | pp_new.py | perturbation_finetune.py |
|--------|-------|-----------|-------------------------|
| Model | TransformerGenerator | DeepSC | DeepSC |
| Gene Alignment | GeneVocab | build_gene_ids_for_dataset | build_gene_ids_for_dataset |
| Training Framework | PyTorch | Lightning Fabric | Lightning Fabric |
| Loss Function | masked_mse_loss | MSELoss | MSELoss |
| Scheduler | StepLR | CosineAnnealing + Warmup | StepLR |
| Perturbation Flags | construct_pert_flags | construct_pert_flags | construct_pert_flags |
| Expression Encoding | continuous/category | discretize + continuous | discretize + continuous |

## Output Structure

```
output_dir/
├── checkpoints/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── best_model.pt
├── logs/
│   └── (training logs)
├── visualizations/
│   └── SAMD1+ZBTB1.png
└── test_metrics.json
```

## Metrics

The evaluation computes several perturbation-specific metrics:

- **Pearson correlation**: Overall correlation between predicted and true expression
- **Pearson delta**: Correlation of expression changes (vs. control)
- **Pearson delta DE**: Correlation on differentially expressed genes only
- **MSE**: Mean squared error
- **Top 20 DE metrics**: Metrics on top 20 DE genes (non-dropout)

## Tips and Best Practices

1. **Gene Coverage**: Ensure good overlap between dataset genes and vocabulary
   - Check matching statistics in logs
   - Consider retraining vocabulary if coverage is low (<80%)

2. **Sequence Length**:
   - Use `data_length=1536` for datasets with ~1000-2000 genes
   - Increase if needed, but watch memory usage

3. **Gene Inclusion Strategy**:
   - `include_zero_gene='all'`: More stable, better for small datasets
   - `include_zero_gene='batch-wise'`: Faster, good for large datasets

4. **Learning Rate**:
   - Start with 1e-4 for fine-tuning pretrained models
   - Use 1e-3 for training from scratch

5. **Early Stopping**:
   - Monitor validation pearson correlation
   - Typical patience: 5-10 epochs

6. **Batch Size**:
   - Larger batch size (64-128) generally better for perturbation tasks
   - Adjust based on GPU memory

## Troubleshooting

**Issue**: Low gene matching rate
- **Solution**: Check vocabulary CSV file, ensure it contains dataset genes

**Issue**: NaN loss during training
- **Solution**: Reduce learning rate, check for zero-division in expression normalization

**Issue**: Poor performance on validation
- **Solution**: Increase model capacity, train for more epochs, check data preprocessing

**Issue**: Out of memory
- **Solution**: Reduce batch_size, reduce data_length, use gradient accumulation

## References

- scGPT perturbation prediction: `/home/angli/old_tasks/scGPT/examples/finetune_perturbation.py`
- DeepSC gene alignment: `/home/angli/DeepSC/src/deepsc/finetune/pp_new.py`
- GEARS library: https://github.com/snap-stanford/GEARS

## Citation

If you use this code, please cite:
- DeepSC paper
- scGPT paper
- GEARS paper
