# Scripts for Data Preprocessing

This directory contains scripts for preprocessing the data.

## 0. Environment Setup

First, we need to set up some environment variables in the `.env` file (example in [.env.example](../../../.env.example)).

```bash
cp .env.example .env
```

## 1. Gene Name ...
描述一下
```bash
bash ./scripts/data/preprocessing/01_preprocess_run_preprocess_gene_name.sh
```

## 1. CellXGene Dataset

TODO:
- [] `batch_normalize.py` ok.
- [] `cellxgene_data_preprocess.py` 修改logging的配置（utils.py已经实现了的话），以及删除sys path的配置方法。
- [] `config.py` 可以再.env中设置
- [] `filter_primary_data.py` ok.
- [] `gene_name_normalization.py` GENE_MAP_PATH, HGNC_DATABASE, SCRIPTS_ROOT环境变量设置，路径硬编码问题。
- [] `get_feature_name_3ca_cxg.py` 环境变量设置，路径硬编码问题。

- [] scripts/data/download/README.md 中，确认环境变量和.env中的环境变量的一致性。
- [] preprocessing/README.md的文档说明。