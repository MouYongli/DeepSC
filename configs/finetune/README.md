# Fine-tuning Configuration Structure

## Overview

This directory contains the configuration files for fine-tuning DeepSC models on various downstream tasks.

## Directory Structure

```
configs/finetune/
├── finetune.yaml              # Main configuration file (common settings)
├── tasks/                     # Task-specific configurations
│   └── cell_type_annotation.yaml
└── README.md                  # This file
```

## Configuration Architecture

### 1. Main Configuration (`finetune.yaml`)

Contains settings common to all fine-tuning tasks:
- Training hyperparameters (learning_rate, batch_size, epoch, etc.)
- Distributed training settings (num_device, num_nodes)
- Model architecture settings (use_moe_ffn, sequence_length, num_bin)
- Pretrained model settings
- Fine-tuning mode selection (full/head_only)
- Checkpoint and logging settings
- Learning rate scheduler settings

### 2. Task-Specific Configurations (`tasks/*.yaml`)

Each task has its own configuration file containing task-specific settings:
- Dataset paths and preprocessing parameters
- Task-specific loss functions and weights
- Task-specific model components
- Evaluation metrics and visualization settings

### 3. Configuration Merging

The `finetune.py` script automatically merges task-specific configs into the main config:

```python
# In finetune.py:
if "tasks" in cfg:
    OmegaConf.set_struct(cfg, False)
    task_cfg = OmegaConf.to_container(cfg.tasks, resolve=True)
    cfg = OmegaConf.merge(cfg, task_cfg)
    del cfg["tasks"]
    OmegaConf.set_struct(cfg, True)
```

After merging, all configurations are accessible via `cfg.key_name` (not `cfg.tasks.key_name`).

## Usage

### Running with Default Task

```bash
# Uses default task (cell_type_annotation)
./scripts/finetune/run_finetune.sh
```

### Running with Specific Task

```bash
# Specify task name as argument
./scripts/finetune/run_finetune.sh cell_type_annotation
```

### Overriding Configuration from Command Line

```bash
# Override specific parameters
PYTHONPATH=src python -m deepsc.finetune.finetune \
    tasks=cell_type_annotation \
    task_type=cell_type_annotation \
    learning_rate=1e-4 \
    batch_size=64
```

## Adding New Tasks

To add a new fine-tuning task:

1. **Create task configuration file:**
   ```bash
   configs/finetune/tasks/your_task_name.yaml
   ```

2. **Add task-specific settings:**
   ```yaml
   # Task-specific WandB tags and run name
   tags: ["your_tags"]
   run_name: "your_run_name"

   # Task-specific dataset settings
   data_path: "path/to/your/data"
   # ... other task-specific configs
   ```

3. **Implement task logic in `finetune.py`:**
   ```python
   elif task_type == "your_task_name":
       # Your task implementation
       pass
   ```

4. **Run your task:**
   ```bash
   ./scripts/finetune/run_finetune.sh your_task_name
   ```

## Configuration Priority

When the same key exists in multiple places, the priority is:
1. Command-line overrides (highest)
2. Task-specific config (`tasks/*.yaml`)
3. Main config (`finetune.yaml`)
4. Model config (`model/*.yaml`)

## Available Tasks

- `cell_type_annotation`: Cell type annotation task for single-cell data

(More tasks to be added...)
