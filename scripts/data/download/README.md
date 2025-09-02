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

### 1.1 CellXGene related environment setup

Set the environment variables for the CellXGene dataset.

```bash
export DATA_PATH_CELLXGENE=$DEEPSC_DATA_ROOT/cellxgene/raw
export INDEX_PATH_CELLXGENE=$DEEPSC_DATA_ROOT/index_list
export DOWNLOAD_SCRIPTS_ROOT_CELLXGENE=$DEEPSC_SCRIPTS_ROOT/data/download/cellxgene
export QUERY_PATH_CELLXGENE=$DOWNLOAD_SCRIPTS_ROOT_CELLXGENE/query_list.txt
```

### 1.2 Build index

Then we need to build the index list for the specified query using [`./cellxgene/01_download_cxg_run_download_cxg_index.sh`](./cellxgene/01_download_cxg_run_download_cxg_index.sh).

```bash
bash ./scripts/data/download/cellxgene/01_download_cxg_run_download_cxg_index.sh
```

### 1.3 Download data

Then we need to download the data using [`./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh`](./cellxgene/02_download_cxg_run_download_partition_cellxgene.sh).
```bash
bash ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene.sh
```
For HPC Slurm users, you can use the following script to download the data:
```bash
sbatch ./scripts/data/download/cellxgene/02_download_cxg_run_download_partition_cellxgene_hpc.sh
```

## 2. TripleCA Dataset (3CA)

### 2.1 TripleCA related environment setup

Set the environment variables for the TripleCA dataset.

```bash
export DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/raw
export MAPPED_DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/mapped_batch_data
export MERGED_DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/merged_batch_data
```
### 2.2 Crawl and download data from 3CA website

Then we need to crawl and download the data from the 3CA website using [`./3ca/01_download_3ca_run_download_3ca.sh`](./3ca/01_download_3ca_run_download_3ca.sh).

```bash
bash ./scripts/data/download/3ca/01_download_3ca_run_download_3ca.sh
```

### 2.3 Filter and merge data

Since the data downloaded from the 3CA website is not in the expected format, we need to filter and merge the data using [`./3ca/02_download_3ca_run_filter_dataset.sh`](./3ca/02_download_3ca_run_filter_dataset.sh), which will merge the data into `h5ad` format and save the metadata to `csv` file.


```bash
bash ./scripts/data/download/3ca/02_download_3ca_run_filter_dataset.sh
```