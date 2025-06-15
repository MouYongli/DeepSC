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
([Yongli Mou*](mou@dbis.rwth-aachen.de), Ang Li, Er Jin, Sikander Hayat, Stefan Decker)

## Overview

**DeepSC** is an deep learning framework designed to enhance the analysis of single-cell RNA sequencing (scRNA-seq) data. ...

## Features
- **Pre-trained Deep Learning Models**: Leverage transformer-based and variational autoencoder (VAE) models for enhanced cell-type classification.
- **Zero-Shot & Few-Shot Learning**: Adapt to new datasets with minimal labeled data using transfer learning.
- **Efficient Batch Effect Correction**: Automatically harmonize data from multiple sources while preserving biological variation.
- **Interactive Visualization**: Generate intuitive UMAP/t-SNE plots with improved embedding quality.
- **Seamless Integration**: Compatible with popular tools like transformers.

## Installation

#### Anaconda
1. create conda environment
```
conda create --name deepsc python=3.10
conda activate deepsc
```

2. Install Jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
```

3. Install dependencies
```
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # torch==2.6.0+cu126, torchvision==0.21.0+cu126, torchaudio==2.6.0+cu126
# pip install torch_geometric # torch-geometric==2.6.1
pip install -e .
```
#### Docker

## Datasets

### Download datasets
1. [Cellxgene Census](https://cellxgene.cziscience.com/datasets)
2. [weizmann 3CA](https://www.weizmann.ac.il/sites/3CA)
   * Configure the organs in [organ_list.txt](./data/download/3ac/organ_list.txt)
   * Collect information of data and metadata
3.


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

## Project Structure

```
📦 DeepSC
├── 📁 data         # Sample datasets and preprocessing scripts
├── 📁 models           # Pre-trained models and training scripts
├── 📁 notebooks        # Jupyter notebooks with tutorials
├── 📁 docs             # Documentation and API references
├── 📁 src              # Core implementation of foundation models
└── README.md           # Project description
```

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
