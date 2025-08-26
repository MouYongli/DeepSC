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

TODO: 
- [ ] issue: `01_download_cxg_run_download_cxg_index.sh`文件中如何优雅的把环境变量先source进来，然后使用deepsc这个包（通过pip install -e .安装的），然后调用deepsc.scripts.download.cellxgene.build_index_list这个函数。



### 2. Download Data







### 2. 分块下载数据 (`02_download_cxg_run_download_partition_cellxgene.sh`)
- **功能**: 根据索引分块下载 CellxGene 数据
- **配置**: 
  - 使用 SLURM 集群调度系统
  - 每个分块最大 200,000 条记录
  - 支持并行处理（数组任务 1-9）
- **资源要求**: 
  - 时间: 20小时
  - CPU: 4核
  - 内存: 48GB
- **执行**: 调用 `src/deepsc/data/download/cellxgene/download_partition.py` 进行分块下载

## TripleCA 数据集

TripleCA 数据集下载过程也分为两个步骤：

### 1. 下载数据 (`01_download_3ca_run_download_3ca.sh`)
- **功能**: 下载 TripleCA 数据集文件
- **配置**:
  - 总文件数: 131个
  - 并行进程数: 8个
- **输出**: 数据保存到 `DATA_PATH_3CA` 目录
- **日志**: 保存到 `LOG_PATH` 目录
- **执行**: 调用 `src.deepsc.data.download.tripleca.download_3ca` 模块

### 2. 数据过滤和合并 (`02_download_3ca_run_filter_dataset.sh`)
- **功能**: 对下载的数据进行过滤和合并处理
- **输入**: 从 `DATA_PATH_3CA` 读取数据集
- **执行**: 调用 `src.deepsc.data.download.tripleca.merge_and_filter_dataset` 模块

## 环境配置

所有脚本都需要以下环境变量（在 `.env` 文件中配置）：
- `DATA_PATH_CELLXGENE`: CellxGene 数据存储路径
- `DATA_PATH_3CA`: TripleCA 数据存储路径
- `QUERY_PATH_CELLXGENE`: CellxGene 查询文件路径
- `INDEX_PATH_CELLXGENE`: CellxGene 索引文件路径
- `LOG_PATH`: 日志文件路径

## 使用说明

1. 确保已配置 `.env` 文件中的相关路径
2. 对于 CellxGene 数据集，先运行索引构建脚本，再运行下载脚本
3. 对于 TripleCA 数据集，先运行下载脚本，再运行过滤脚本
4. CellxGene 下载脚本支持 SLURM 集群调度，可根据需要调整资源参数
