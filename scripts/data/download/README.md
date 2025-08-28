# Scripts for Data Download

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
export DEEPSC_LOGS_ROOT=$DEEPSC_PROJECT_ROOT/logs
```

## 1. CellXGene Dataset

### 1.1 CellXGene related Environment Setup

Set the environment variables for the CellXGene dataset.

```bash
export DATA_PATH_CELLXGENE=$DEEPSC_DATA_ROOT/cellxgene/raw
export INDEX_PATH_CELLXGENE=$DEEPSC_DATA_ROOT/index_list
export DOWNLOAD_SCRIPTS_ROOT_CELLXGENE=$DEEPSC_SCRIPTS_ROOT/data/download/cellxgene
export QUERY_PATH_CELLXGENE=$DOWNLOAD_SCRIPTS_ROOT_CELLXGENE/query_list.txt
```

### 1.2 Build Index

Then we need to build the index list for the specified query using [`./cellxgene/01_download_cxg_run_download_cxg_index.sh`](./cellxgene/01_download_cxg_run_download_cxg_index.sh).

```bash
bash ./scripts/data/download/cellxgene/01_download_cxg_run_download_cxg_index.sh
```

### 1.3 Download Data

Then we need to download the data using [`./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh`](./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh).
```bash
bash ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene.sh
```
For HPC Slurm users, you can use the following script to download the data:
```bash
sbatch ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene_hpc.sh
```

## 2. TripleCA Dataset (3CA)

### 2.1 TripleCA related Environment Setup

Set the environment variables for the TripleCA dataset.

```bash
export DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/raw
export MAPPED_DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/mapped_batch_data
export MERGED_DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/merged_batch_data
```


TODO: 
- [] 整理准备3CA数据集的下载脚本。
- [] 解释3CA数据集的下载脚本中"num_files"参数的作用。提高代码可读性。
- [] 提高3CA数据集过滤和合并的python脚本的可读性，避免出现硬编码的路径，尽量使用预定义好的环境变量。
- [] 整理需要用到的python包，并添加到pyproject.toml和requirements.txt中。
- [] 整理准备Preprocess脚本。

- [] 联系Sikander，准备会议内容：文章写些什么，专注在哪些任务上，需要哪些数据集（除了公开的benchmark数据集，UKA是否有额外的测试）。
- [] 给Jason update一下我们的内容。然后告诉他我们下周就可以开始写了。
