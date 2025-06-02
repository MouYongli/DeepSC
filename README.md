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
[Yongli Mou*](mou@dbis.rwth-aachen.de), Ang Li, Er Jin, Sikander Hayat, Stefan Decker

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
conda create --name deepsc python=3.11
conda activate deepsc
```

2. Install Jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
```

3. Install dependencies
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -r requirements.txt
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
