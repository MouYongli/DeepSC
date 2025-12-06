# Cell Type Annotation Scripts

This directory contains bash scripts for analyzing and processing h5ad files for cell type annotation tasks.

## Available Scripts

### 1. analyze_obs_columns.sh

Analyzes the columns in the obs dataframe of an h5ad file.

**Usage:**
```bash
# Use default input file
./analyze_obs_columns.sh

# Specify custom input file
./analyze_obs_columns.sh --input /path/to/your/file.h5ad

# Show help
./analyze_obs_columns.sh --help
```

**Arguments:**
- `--input PATH`: Path to input h5ad file (default: partition_3.h5ad)
- `-h, --help`: Show help message

---

### 2. create_balanced_dataset.sh

Creates a balanced dataset by sampling cells from each cell type. Only includes cell types with sufficient cells, and samples a fixed number from each.

**Usage:**
```bash
# Create dataset with 500 cells per type (default)
./create_balanced_dataset.sh

# Create dataset with 1000 cells per type
./create_balanced_dataset.sh --n_cells 1000

# Sample 200 cells per type, but only from cell types with >500 cells
./create_balanced_dataset.sh --n_cells 200 --min_threshold 500

# Use custom input and output paths
./create_balanced_dataset.sh --input /path/to/input.h5ad --output_dir /path/to/output

# Show help
./create_balanced_dataset.sh --help
```

**Arguments:**
- `--n_cells N`: Number of cells to sample per cell type (default: 500)
- `--input PATH`: Path to input h5ad file
- `--output_dir PATH`: Directory to save output file
- `--min_threshold N`: Minimum number of cells required for a cell type to be included (default: same as --n_cells)
- `-h, --help`: Show help message

**Output:**
The output file will be automatically named `balanced_dataset_kidney_{n_cells}.h5ad` where `{n_cells}` is the number specified.

**Default Output Directory:**
`/home/angli/baseline/DeepSC/data/cell_type_annotation/balanced_dataset/`

---

### 3. validate_balanced_dataset.sh

Validates a balanced dataset by checking:
1. All cells have `is_primary_data == True`
2. All cell types have the same number of cells

**Usage:**
```bash
# Validate a balanced dataset
./validate_balanced_dataset.sh --input /path/to/balanced_dataset_kidney_500.h5ad

# Validate and verify expected cell count per type
./validate_balanced_dataset.sh --input /path/to/balanced_dataset_kidney_500.h5ad --expected_n_cells 500

# Show help
./validate_balanced_dataset.sh --help
```

**Arguments:**
- `--input PATH`: Path to the h5ad file to validate (required)
- `--expected_n_cells N`: Expected number of cells per cell type (optional)
- `-h, --help`: Show help message

**Exit Codes:**
- `0`: All validation tests passed
- `1`: One or more validation tests failed

---

## Examples

### Example 1: Analyze an h5ad file
```bash
./analyze_obs_columns.sh --input /home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad
```

### Example 2: Create balanced dataset with different cell numbers

```bash
# 300 cells per type
./create_balanced_dataset.sh --n_cells 300

# 800 cells per type
./create_balanced_dataset.sh --n_cells 800

# 1500 cells per type
./create_balanced_dataset.sh --n_cells 1500
```

### Example 3: Sample fewer cells from abundant cell types
```bash
# Sample 200 cells from each type that has at least 1000 cells
./create_balanced_dataset.sh --n_cells 200 --min_threshold 1000
```

### Example 4: Complete workflow - Create and validate
```bash
# Step 1: Analyze the original dataset
./analyze_obs_columns.sh --input /home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad

# Step 2: Create a balanced dataset with 500 cells per type
./create_balanced_dataset.sh --n_cells 500

# Step 3: Validate the balanced dataset
./validate_balanced_dataset.sh \
    --input /home/angli/baseline/DeepSC/data/cell_type_annotation/balanced_dataset/balanced_dataset_kidney_500.h5ad \
    --expected_n_cells 500
```

---

## Python Scripts

The bash scripts are wrappers around Python scripts located in:
- `/home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/analyze_obs_columns.py`
- `/home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/create_balanced_dataset.py`
- `/home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/validate_balanced_dataset.py`

You can also run these Python scripts directly:

```bash
# Analyze obs columns
python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/analyze_obs_columns.py --input /path/to/file.h5ad

# Create balanced dataset
python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/create_balanced_dataset.py \
    --n_cells 500 \
    --input /path/to/input.h5ad \
    --output_dir /path/to/output \
    --min_threshold 500

# Validate balanced dataset
python /home/angli/baseline/DeepSC/src/deepsc/data/cell_type_annotation/validate_balanced_dataset.py \
    --input /path/to/balanced_dataset_kidney_500.h5ad \
    --expected_n_cells 500
```

---

## Requirements

- Conda environment: `deepsc`
- Required Python packages: `anndata`, `pandas`, `numpy`

The scripts will automatically activate the `deepsc` conda environment.
