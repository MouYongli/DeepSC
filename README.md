# DeepSC: Deep Transcriptomic Foundation Models for Single-Cell RNA-Sequencing Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![Forks](https://img.shields.io/github/forks/MouYongli/DeepSC?style=social)](https://github.com/MouYongli/DeepSC/network/members)
[![Stars](https://img.shields.io/github/stars/MouYongli/DeepSC?style=social)](https://github.com/MouYongli/DeepSC/stargazers)
[![Issues](https://img.shields.io/github/issues/MouYongli/DeepSC)](https://github.com/MouYongli/DeepSC/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/MouYongli/DeepSC)](https://github.com/MouYongli/DeepSC/pulls)
[![Contributors](https://img.shields.io/github/contributors/MouYongli/DeepSC)](https://github.com/MouYongli/DeepSC/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/MouYongli/DeepSC)](https://github.com/MouYongli/DeepSC/commits/main)
<!-- [![Build Status](https://img.shields.io/github/actions/workflow/status/MouYongli/DeepSC/ci.yml)](https://github.com/MouYongli/DeepSC/actions)
[![Code Quality](https://img.shields.io/lgtm/grade/python/g/MouYongli/DeepSC.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MouYongli/DeepSC/context:python) -->

[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://hub.docker.com/r/YOUR_DOCKER_IMAGE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-yellow)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx)


[![WeChat](https://img.shields.io/badge/WeChat-公众号名称-green)](https://your-wechat-link.com)
[![Weibo](https://img.shields.io/badge/Weibo-关注-red)](https://weibo.com/YOUR_WEIBO_LINK)
<!-- [![Discord](https://img.shields.io/discord/YOUR_DISCORD_SERVER_ID?label=Discord&logo=discord&color=5865F2)](https://discord.gg/YOUR_INVITE_LINK) -->
<!-- [![Twitter](https://img.shields.io/twitter/follow/YOUR_TWITTER_HANDLE?style=social)](https://twitter.com/YOUR_TWITTER_HANDLE) -->

This is official repo for "Deep Transcriptomic Foundation Models for Single-Cell RNA-Sequencing Data" by DBIS and LfB RWTH Aachen University and RWTH Universty Hosptial Aachen
([Yongli Mou*](mou@dbis.rwth-aachen.de), Ang Li, Er Jin, Sikander Hayat, Johannes Stegmaier, Stefan Decker)

## Overview

**DeepSC** is deep foundation model designed for the analysis of single-cell RNA sequencing (scRNA-seq) data. The model is based on transformer architecture. In our models, we design two branches for inputs of gene and expression seperately. The gene branch is used to encode the gene information, and the expression branch is used to encode the expression information. To model the gene regulatory network and the sparse connectivity, we use Gumbel Softmax to generate the sparse connectivity matrix with three signals (depression, activation, and no change).

## Features
- **Pipeline**: We develop a pipeline for scRNA-seq data analysis, including data preprocessing, model pre-training, and finetuning and inferencing for different downstream tasks.

- **Pre-trained Deep Learning Models**: We develop a deep learning model based on transformer architecture and pretrain it on a large-scale scRNA-seq dataset with more than 30,000,000 cells and 34,683 genes.

- **Downstream Tasks**: We develop a series of downstream tasks for scRNA-seq data analysis, including cell-type annotation, batch effect correction, gene regulatory network inference, perturbation analysis, and cell-cell communication inference.

## Installation

#### Conda
1. create conda environment
```
conda create --name deepsc python=3.10
conda activate deepsc
```

2. Install dependencies
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # torch==2.6.0+cu126, torchvision==0.21.0+cu126, torchaudio==2.6.0+cu126
pip install torch_geometric # torch-geometric==2.6.1
pip install -e .
```

## Datasets

In this project, we use the following datasets:
- [Cellxgene Census](https://cellxgene.cziscience.com/datasets)
- [weizmann 3CA](https://www.weizmann.ac.il/sites/3CA)

### Download datasets
To download the datasets, please refer to the [scripts/data/download](./scripts/data/download) directory.

### Data Preprocessing
















## Configuration Management with Hydra

This project uses [Hydra](https://hydra.cc/) to manage configuration files in a hierarchical, modular, and override-friendly way. This allows for clean separation between datasets, model architecture, and training parameters, making experiments easy to reproduce and customize.

### Configuration Structure

The main configuration directory is:
<pre><code>
configs/pretrain/
├── pretrain.yaml           # Entry point config
├── model/
│   └── scbert.yaml         # Model configuration
└── dataset/
      └── tripleca.yaml       # Dataset configuration
</code></pre>

### Running with Config

To run with the default configuration:

```bash
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain

```
To override parameters from the command line:
```bash
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain
  epoch=20 model.dim=512 dataset.num_bin=7
```
Hydra will automatically merge the overrides and save the full resolved configuration and logs.

### Configuration Breakdown (Examples):

Define the base configuration and default components in [pretrain.yaml](configs/pretrain/pretrain.yaml)

Define model architecture parameters (Example: [scbert.yaml](configs/pretrain/model/scbert.yaml))

Define dataset loading configuration (Example: [tripleca.yaml](configs/pretrain/dataset/tripleca.yaml))

### Instantiating Objects with Hydra

This project uses Hydra's `_target_` mechanism and `hydra.utils.instantiate()` to construct Python objects directly from configuration files. This enables dynamic loading of models, datasets, and other components by simply editing YAML files — no code changes are needed.


#### Example: Model Instantiation

In [`configs/pretrain/model/scbert.yaml`](configs/pretrain/model/scbert.yaml):

```yaml
_target_: deepsc.models.scbert.model.PerformerLM
g2v_position_emb: true
dim: 200
num_tokens: 7
max_seq_len: 60664
depth: 6
heads: 10
local_attn_heads: 0
```

This defines how to instantiate the PerformerLM model. The _target_ key points to the full Python path of the class, and the remaining keys are passed as constructor arguments.

In [pretrain.py](src/deepsc/pretrain/pretrain.py), the model is created via:

```python
model: nn.Module = hydra.utils.instantiate(cfg.model)
```

###  Output & Logging:

Hydra automatically logs each run to a timestamped directory under outputs/ (or as configured). Each run directory contains:

1. config.yaml: the full merged config
2. hydra.yaml: Hydra’s internal config
3. overrides.yaml: CLI overrides
4. <job_name>_0.log: stdout log file


## Setup Instructions for Weights & Biases (wandb)

### 1. Get your wandb API token

Visit [https://wandb.ai/authorize](https://wandb.ai/authorize) and log into your wandb account. Your API key (token) will be displayed on the page.

### 2. Set your token (Recommended: via CLI)

Run the following command in your terminal, replacing `<YOUR_TOKEN>` with your actual token:

```bash
wandb login <YOUR_TOKEN>
````

This command will automatically store your token in the ~/.netrc file. All subsequent wandb scripts will be able to read it without requiring manual setup each time.

### Environment Configuration

Before running the scripts, make sure to configure your environment variables in a `.env` file located in the project root. Below is an example of what your `.env` file might look like:

```env
# Path of storing the data index, data result and query list for Cellxgene
INDEX_PATH_CELLXGENE="/home/xxxx/DeepSC/data/index_list"
QUERY_PATH_CELLXGENE="/home/xxxx/DeepSC/scripts/download/cellxgene/query_list.txt"
DATA_PATH_CELLXGENE="/home/xxxx/DeepSC/data"

# Path of storing the data for 3CA
DATA_PATH_3CA="/home/xxxx/DeepSC/data/3ca/raw"
MAPPED_DATA_PATH_3CA="/home/xxxx/DeepSC/mapped_batch_data/3ca"
MERGED_DATA_PATH_3CA="/home/xxxx/DeepSC/data/3ac/merged_batch_data"

# Path of storing the logs
LOG_PATH="/home/xxxx/DeepSC/logs"
```

## Usage

Here’s an example of how to process a dataset:

```python
from deepsc import Model
# Load a pre-trained model
model = Model.load_pretrained("deepsc/model-base")
# Process your dataset
results = model.infer("data/sample.h5ad")
# Visualize the output
model.visualize(results)
```

More detailed tutorials can be found in our [documentation](https://your-project-website.com/docs).

## Quick Start - Pipeline Execution Guide

### 1. Data Download Pipeline

#### Prerequisites
Configure your environment variables in `.env` file (see `.env.example`):
```bash
# Data paths
DATA_PATH_CELLXGENE="/path/to/cellxgene/data"
DATA_PATH_3CA="/path/to/3ca/data"
LOG_PATH="/path/to/logs"
```

#### Download 3CA Dataset
```bash
# Create necessary directories
mkdir -p /path/to/3ca/data /path/to/logs

# Run 3CA data download
python -m src.deepsc.data.download.tripleca.download_3ca \
    --output_path "/path/to/3ca/data" \
    --log_path "/path/to/logs" \
    --num_files 131 \
    --num_processes 8
```

#### Download CellxGene Dataset  
```bash
python -m src.deepsc.data.download.cellxgene.download_partition \
    --output_path "/path/to/cellxgene/data"
```

### 2. Data Preprocessing Pipeline

Process downloaded data for training:
```bash
# Preprocess 3CA datasets
python -m src.deepsc.data.preprocessing.preprocess_datasets_3ca \
    --input_path "/path/to/3ca/data" \
    --output_path "/path/to/processed/3ca"

# Preprocess CellxGene datasets  
python -m src.deepsc.data.preprocessing.preprocess_datasets_cellxgene \
    --input_path "/path/to/cellxgene/data" \
    --output_path "/path/to/processed/cellxgene"
```

### 3. Pretraining Pipeline

#### Single GPU Training
```bash
python -m src.deepsc.pretrain.pretrain \
    model=scbert \
    dataset=tripleca \
    batch_size=32 \
    learning_rate=1e-4 \
    epoch=10
```

#### Multi-GPU Training  
```bash
torchrun \
    --nproc_per_node=4 \
    --master_port=12355 \
    -m src.deepsc.pretrain.pretrain \
    model=scbert \
    dataset=tripleca \
    batch_size=32 \
    learning_rate=1e-4 \
    epoch=10
```

#### Key Configuration Options
- `model={scbert,deepsc,scgpt}`: Choose model architecture
- `dataset={tripleca,cellxgene}`: Choose dataset
- `batch_size=32`: Batch size for training  
- `epoch=10`: Number of training epochs
- `data_path="/path/to/data"`: Override default data path

### 4. Pipeline Verification

Run integration tests to verify everything works:
```bash
# Test all pipelines
pytest tests/integration/ -v

# Test specific pipeline
pytest tests/integration/test_pretrain_pipeline.py -v
```

### 5. Monitoring and Logging

The pipeline uses Weights & Biases for experiment tracking:
1. Set up wandb: `wandb login <your-token>`
2. Configure wandb settings in config files
3. Monitor training at [wandb.ai](https://wandb.ai)

Logs are automatically saved to `./logs/` directory with timestamped files.

## Project Structure

```
📦 DeepSC/
├── 📁 configs/             # Hydra configuration files
│   ├── 📁 pretrain/       # Pretraining configurations
│   │   ├── pretrain.yaml  # Main pretraining config
│   │   ├── 📁 model/      # Model architecture configs
│   │   └── 📁 dataset/    # Dataset loading configs
│   └── 📁 finetune/       # Finetuning configurations
├── 📁 data/               # Sample datasets (empty by default)
├── 📁 src/deepsc/         # Core implementation
│   ├── 📁 data/          # Data loading and preprocessing
│   │   ├── 📁 download/   # Data download scripts
│   │   └── 📁 preprocessing/ # Data preprocessing utilities
│   ├── 📁 models/        # Model architectures
│   │   ├── 📁 scbert/    # ScBERT model implementation
│   │   ├── 📁 deepsc/    # DeepSC model implementation
│   │   └── 📁 scgpt/     # Other model implementations
│   ├── 📁 train/         # Training infrastructure
│   ├── 📁 pretrain/      # Pretraining entry points
│   ├── 📁 finetune/      # Finetuning entry points
│   └── 📁 utils/         # Utility functions
├── 📁 tests/              # Test suite
│   ├── 📁 unit/          # Unit tests
│   ├── 📁 integration/   # Integration tests
│   └── conftest.py       # Shared test fixtures
├── 📁 scripts/            # Utility scripts
│   └── 📁 debug/         # Debugging and analysis scripts
├── 📁 notebooks/          # Jupyter notebooks with tutorials
└── README.md              # Project documentation
```

### Key Directories

- **`configs/`**: All Hydra configuration files for models, datasets, and training
- **`src/deepsc/`**: Core source code organized by functionality
- **`tests/`**: Comprehensive test suite with unit and integration tests
- **`scripts/debug/`**: Development and debugging utilities
- **`data/`**: Placeholder for datasets (configure paths in configs)

## Benchmark Results

| Model         | Accuracy | Batch Effect Score | Training Time |
|--------------|---------:|-------------------:|--------------:|
| sCmilarity | xx.x%    | 0.xx               | xh xxm        |
| scFoundation | xx.x%    | 0.xx               | xh xxm        |
| scGPT   | xx.x%    | 0.xx               | xh xxm        |
| scBERT | xx.x%    | 0.xx               | xh xxm        |
| DeepSC (ours)  | **xx.x%**   | **0.xx**               | xh xxm        |

More benchmarks are available in the [research paper](https://your-project-website.com/paper).

## Contribution

We welcome contributions from the community! Please check our [contribution guide](CONTRIBUTING.md) before submitting a PR.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:
```bibtex
@article{mou2025deepsc,
  author  = {Yongli Mou, Ang Li, Er Jin, Sikander Hayat and Stefan Decker},
  title   = {DeepSC: Deep Transcriptomic Foundation Models for Single-Cell RNA-Sequencing Data},
  journal = {XXX},
  year    = {202X}
}
```

<!-- ---

Developed by **Your Name** | [LinkedIn](https://linkedin.com/in/YOURNAME) | [Twitter](https://twitter.com/YOURHANDLE) -->
