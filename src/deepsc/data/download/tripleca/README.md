# 3CA Data Download and Preprocessing Pipeline

This project provides a full pipeline for downloading and preprocessing datasets from the Triple-CA data source. It includes metadata crawling, raw data downloading, and dataset merging and filtering.

## File Structure

```text
.
├── config.py                 # Configuration file with base URL and organ list
├── crawl_3ca.py           # Script to crawl metadata from Triple-CA
├── download_3ca.py         # Script to download raw data based on metadata CSV
├── merge_and_filter_dataset.py  # Script to merge and filter downloaded data

---
```

## Step-by-Step Guide

### 1. Crawl Metadata and Download Raw Data

Use download_3ca.py to download data files listed in data_info.csv.

**Recommended Bash Script**: See [`DeepSC/examples/run_download_3ca.sh`](DeepSC/examples/run_download_3ca.sh)

```bash
#!/bin/bash

OUTPUT_PATH="/home/angli/DeepSC/data/3ac/testraw"
LOG_PATH="./logs"
NUM_FILES=131
NUM_PROCESSES=8  # default number of process

mkdir -p "$OUTPUT_PATH"
mkdir -p "$LOG_PATH"

python -m scripts.download.tripleca.download_3ca \
    --output_path "$OUTPUT_PATH" \
    --log_path "$LOG_PATH" \
    --num_files "$NUM_FILES" \
    --num_processes "$NUM_PROCESSES"
```

### 2. Merge and Filter Datasets
After downloading, run merge_and_filter_dataset.py to generate a merged and filtered CSV for further processing.
**Recommended Bash Script**: See [`DeepSC/examples/run_filter_dataset.sh`](DeepSC/examples/run_filter_dataset.sh)

```bash
#!/bin/bash

DATASET_PATH="/home/angli/DeepSC/data/3ac/raw"

python -m scripts.download.tripleca.merge_and_filter_dataset \
    --dataset_root_path "$DATASET_PATH"

```

**Output**:
    updated_target_dataset_with_datasetid.csv: Includes a unique dataset_id for each entry.

### Configuration (`config.py`)

The target URL for crawling and the list of organs are defined in the `config.py` file:

```python
BASE_URL = "https://tripleca.org/..."  # Base URL for downloading files
ORGAN_LIST = ["lung", "liver", "heart", ...]  # List of target organs
