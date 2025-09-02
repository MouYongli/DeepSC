# Scripts for Data Preprocessing

This directory contains scripts for preprocessing the data.

## 1. CellXGene Dataset

`raw` -> `processed`

TODO:
- [] `batch_normalize.py` ok.
- [] `cellxgene_data_preprocess.py` 修改logging的配置（utils.py已经实现了的话），以及删除sys path的配置方法。
- [] `config.py` 可以再.env中设置
- [] `filter_primary_data.py` ok.
- [] `gene_name_normalization.py` GENE_MAP_PATH, HGNC_DATABASE, SCRIPTS_ROOT环境变量设置，路径硬编码问题。
- [] `get_feature_name_3ca_cxg.py` 环境变量设置，路径硬编码问题。