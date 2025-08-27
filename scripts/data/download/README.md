# Scripts for Downloading Datasets

This directory contains scripts for downloading two main datasets: CellXGene and TripleCA (3CA).


## 0. Environment Setup
First, we need to set up some environment variables in the `.env` file (example in [.env.example](../../../.env.example)).

```bash
cp .env.example .env
```

Or, you can also set these environment variables via the command line.

```bash
export DEEPSC_PROJECT_ROOT=$PWD
export DEEPSC_DATA_ROOT=$DEEPSC_PROJECT_ROOT/data
export DEEPSC_SCRIPTS_ROOT=$DEEPSC_PROJECT_ROOT/scripts
```

## CellXGene Dataset

### 0. CellXGene relatedEnvironment Setup

```bash
export DOWNLOAD_SCRIPTS_ROOT_CELLXGENE=$DEEPSC_SCRIPTS_ROOT/data/download/cellxgene
export DATA_PATH_CELLXGENE=$DEEPSC_DATA_ROOT/cellxgene
export INDEX_PATH_CELLXGENE=$DATA_PATH_CELLXGENE/index_list
export QUERY_PATH_CELLXGENE=$DOWNLOAD_SCRIPTS_ROOT_CELLXGENE/query_list.txt
```

### 1. Build Index
Then we need to build the index list for the specified query using [`./cellxgene/01_download_cxg_run_download_cxg_index.sh`](./cellxgene/01_download_cxg_run_download_cxg_index.sh).

```bash
bash ./scripts/data/download/cellxgene/01_download_cxg_run_download_cxg_index.sh
```

### 2. Download Data

Then we need to download the data using [`./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh`](./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh).
```bash
bash ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene.sh
```
For HPC Slurm users, you can use the following script to download the data:
```bash
sbatch ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene_hpc.sh
```

TODO: 
- [] 测试HPC Slurm 脚本是否可以正常运行。
- [] 修改非HPC Slurm 脚本，使其可以在服务器上（如warhol）正常运行。
- [] 整理准备3CA数据集的下载脚本。
- [] 整理准备Preprocess脚本。

