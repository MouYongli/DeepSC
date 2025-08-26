# 数据下载脚本

本目录包含用于下载两个主要数据集的脚本：CellxGene 和 TripleCA。

## CellxGene 数据集

CellxGene 是一个单细胞基因表达数据集，下载过程分为两个步骤：

### 1. 构建索引 (`01_download_cxg_run_download_cxg_index.sh`)
- **功能**: 为指定的查询构建索引列表
- **输入**: 从环境变量 `QUERY_PATH_CELLXGENE` 读取查询名称
- **输出**: 索引文件保存到 `DATA_PATH_CELLXGENE` 目录
- **执行**: 对每个查询名称调用 `scripts.download.cellxgene.build_index_list` 模块

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
