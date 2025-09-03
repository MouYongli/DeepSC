# Scripts for Data Preprocessing

This directory contains scripts for preprocessing the data.

## 0. Environment Setup

First, we need to set up some environment variables in the `.env` file (example in [.env.example](../../../.env.example)).

```bash
cp .env.example .env
```
```bash
export DEEPSC_HGNC_DATABASE=$DEEPSC_SCRIPTS_ROOT/data/preprocessing/HGNC_database.txt
export DEEPSC_GENE_MAP_PATH=$DEEPSC_SCRIPTS_ROOT/data/preprocessing/gene_map.csv

export INTERMEDIATE_ARTIFACTS_TEMP=$DEEPSC_DATA_ROOT/intermediate_artifacts
export TEMPORARY_DATA_PATH_3CA=$DEEPSC_DATA_ROOT/3ca/TEMPORARY_DATA
export TEMPORARY_DATA_PATH_CXG=$DEEPSC_DATA_ROOT/cellxgene/TEMPORARY_DATA

export MAPPED_DATA_PATH_3CA=$TEMPORARY_DATA_PATH_3CA/mapped_batch_data
export MERGED_DATA_PATH_3CA=$TEMPORARY_DATA_PATH_3CA/merged_batch_data

export DEEPSC_CELLXGENE_PRIMARY_DATA_PATH=$TEMPORARY_DATA_PATH_CXG/primary
export DATASET_BEFORE_SHUFFEL=$DEEPSC_DATA_ROOT/processed/npz_data_before_shuffel
export SHUFFEL_PLAN_PATH=$INTERMEDIATE_ARTIFACTS_TEMP/shuffel_plan.csv
export DATASET_SHUFFELED=$DEEPSC_DATA_ROOT/processed/npz_data_after_shuffel
```
## 1. Gene Name Normalization

This section is responsible for gene name normalization. Since gene names may vary across different datasets (due to aliases, case differences, or historical naming conventions), it is necessary to map all gene names to standardized HGNC symbols. This step uses database from HGNC `HGNC_database.txt` to generate a mapping file `gene_map.csv`. It can convert original gene names into consistent, official names.

```bash
bash ./scripts/data/preprocessing/01_preprocess_run_preprocess_gene_name.sh
```
## 2. 3CA Gene Name Mapping

Run the following command to perform gene name mapping for the raw 3CA dataset. It calls the corresponding Python script to convert original gene names to standardized HGNC symbols using the `gene_map.csv` file.

```bash
bash ./scripts/data/preprocessing/02_preprocess_3ca_run_map_3ca.sh
```

## 3. 3CA Dataset Merging

Run the following command to merges the mapped 3CA data batches. It combines multiple mapped batch files into a single h5ad file and generates the corresponding metadata, facilitating downstream analysis and processing.

```bash
bash ./scripts/data/preprocessing/03_preprocess_3ca_run_merge_3ca.sh
```

## 4. 3CA Dataset Batch Normalization

This section performs batch normalization (log1p normalization) on the merged 3CA dataset. This step ensures that the data is normalized and ready for downstream analysis.

Run the following command to execute batch normalization on the merged 3CA data:

```bash
bash ./scripts/data/preprocessing/04_batch_normalize.sh
```

## 5. 3CA Primary Data Filtering

This section is used to filter the primary data from the cellxgene dataset. It ensures that all cells only appear once in the whole datasets.

Run the following command to filter the primary data from the cellxgene dataset:

```bash
bash ./scripts/data/preprocessing/05_filter_primary_data.sh
```

## 6. CellXGene Dataset Preprocessing

This section preprocesses the filtered cellxgene primary data. It applies gene name mapping and converts the data into the format required for the training pipeline.

Run the following command to preprocess the cellxgene dataset:

```bash
bash ./scripts/data/preprocessing/06_cellxgene_dataset_preprocess.sh
```

## 7. Dataset Shuffle Index Generation

This section generates a shuffle plan for the preprocessed dataset. It creates an index file that defines how the dataset should be shuffled for training purposes.

Run the following command to generate the shuffle index:

```bash
bash ./scripts/data/preprocessing/07_shuffel_dataset_generate_index.sh
```

## 8. Final Dataset Shuffling

This section performs the final dataset shuffling based on the generated shuffle plan. It creates the final shuffled dataset that will be used for training.

Run the following command to perform the final dataset shuffling:

```bash
bash ./scripts/data/preprocessing/08_shuffel_dataset_final.sh
```

## Notes

TODO:
- [] `batch_normalize.py` ok.
- [] `cellxgene_data_preprocess.py` 修改logging的配置（utils.py已经实现了的话），以及删除sys path的配置方法。
- [] `config.py` 可以再.env中设置
- [] `filter_primary_data.py` ok.
- [] `gene_name_normalization.py` GENE_MAP_PATH, HGNC_DATABASE, SCRIPTS_ROOT环境变量设置，路径硬编码问题。
- [] `get_feature_name_3ca_cxg.py` 环境变量设置，路径硬编码问题。

- [x] scripts/data/download/README.md 中，确认环境变量和.env中的环境变量的一致性。
- [x] preprocessing/README.md的文档说明。
