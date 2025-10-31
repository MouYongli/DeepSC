# 数据处理流程对比：CellxGene vs 3CA

## 📊 总结

### ✅ 结论：**两个数据集现在使用相同的归一化方法**

两个数据集最终都经过了：
- **TP10K 归一化** (scale_factor = 10000)
- **log1p 变换** (自然对数)

---

## 🔄 详细流程对比

### 1. CellxGene 数据集

#### 脚本流程：
```bash
06_cellxgene_dataset_preprocess.sh
  ↓
cellxgene_data_preprocess.py
```

#### 处理步骤：
1. **基因映射** - 将基因名映射到标准 ID
2. **格式转换** - h5ad → npz (稀疏矩阵)
3. **TP10K 归一化** - 消除测序深度差异 ✅
4. **log1p 变换** - 稳定方差 ✅

#### 关键代码：
```python
# src/deepsc/data/preprocessing/cellxgene_data_preprocess.py
def normalize_tensor(
    csr_matrix,
    min_genes: int = 200,
    scale_factor: float = 1e4,  # TP10K
    apply_log1p: bool = True,   # log1p (自然对数)
):
    # 1. 过滤细胞
    valid_cells = np.diff(csr_matrix.indptr) >= min_genes
    csr_filtered = csr_matrix[valid_cells]

    # 2. 计算 library size
    library_sizes = np.array(csr_filtered.sum(axis=1)).flatten()
    library_sizes[library_sizes == 0] = 1

    # 3. TP10K 归一化（对角矩阵优化）
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")
    csr_filtered = D @ csr_filtered

    # 4. log1p 变换
    if apply_log1p:
        csr_filtered.data = np.log1p(csr_filtered.data)  # ln(1 + x)

    return csr_filtered
```

#### 默认参数：
- `--scale_factor 10000` (TP10K)
- `--min_genes 200`
- `--no_log1p` 未使用 → log1p 开启

---

### 2. 3CA 数据集

#### 脚本流程：
```bash
02_preprocess_3ca_run_map_3ca.sh
  ↓
preprocess_datasets_3ca.py → process_h5ad_to_sparse_tensor()
  ↓
[输出原始数据，无归一化]

03_preprocess_3ca_run_merge_3ca.sh
  ↓
preprocess_3ca_merge.py
  ↓
[批次合并，无归一化]

04_batch_normalize.sh
  ↓
batch_normalize.py → normalize_with_tp10k()
  ↓
[TP10K + log1p 归一化] ✅
```

#### 处理步骤：

**阶段 1：基因映射和格式转换 (02_)**
- 使用 `preprocess_datasets.py::process_h5ad_to_sparse_tensor()`
- **只做基因映射，不做归一化** ❌
- 输出：原始 count 数据

```python
# src/deepsc/data/preprocessing/preprocess_datasets.py
def process_h5ad_to_sparse_tensor(h5ad_path, output_path, gene_map_path):
    # ... 基因映射 ...
    X_final = csr_matrix(...)  # 保持原始 counts
    save_npz(output_path, X_final)  # 无归一化
    return {"status": "saved", "path": output_path}
```

**阶段 2：批次合并 (03_)**
- 使用 `preprocess_3ca_merge.py`
- **只做文件合并，不做归一化** ❌

**阶段 3：归一化 (04_)**
- 使用 `batch_normalize.py::normalize_with_tp10k()`
- **TP10K + log1p 归一化** ✅

```python
# src/deepsc/data/preprocessing/batch_normalize.py
def normalize_with_tp10k(
    csr: sparse.csr_matrix,
    scale_factor: float = 1e4,  # TP10K
    min_genes: int = 200,
    apply_log1p: bool = True,   # log1p (自然对数)
) -> sparse.csr_matrix:
    # 1. 过滤细胞
    valid_cells = np.diff(csr.indptr) >= min_genes
    csr = csr[valid_cells]

    # 2. 计算 library size
    library_sizes = np.array(csr.sum(axis=1)).flatten()
    library_sizes[library_sizes == 0] = 1

    # 3. TP10K 归一化（对角矩阵优化）
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")
    csr = D @ csr

    # 4. log1p 变换
    if apply_log1p:
        csr.data = np.log1p(csr.data)  # ln(1 + x)

    return csr
```

#### 默认参数：
- `--scale_factor 10000` (TP10K)
- `--min_genes 200`
- `--no_log1p` 未使用 → log1p 开启

---

## ⚖️ 关键对比

| 特性 | CellxGene | 3CA |
|------|-----------|-----|
| **处理流程** | 一步完成 | 三步完成 |
| **归一化方法** | TP10K | TP10K |
| **scale_factor** | 10000 | 10000 |
| **log 变换** | log1p (ln) | log1p (ln) |
| **优化方法** | 对角稀疏矩阵 | 对角稀疏矩阵 |
| **min_genes** | 200 | 200 |
| **最终结果** | ✅ 相同 | ✅ 相同 |

---

## 🔍 代码一致性检查

### normalize_with_tp10k() vs normalize_tensor()

这两个函数实现**完全相同**的归一化逻辑：

```python
# batch_normalize.py (3CA)
def normalize_with_tp10k(csr, scale_factor=1e4, apply_log1p=True):
    library_sizes = np.array(csr.sum(axis=1)).flatten()
    library_sizes[library_sizes == 0] = 1
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")
    csr = D @ csr
    if apply_log1p:
        csr.data = np.log1p(csr.data)  # ← log1p (自然对数)
    return csr

# cellxgene_data_preprocess.py (CellxGene)
def normalize_tensor(csr_matrix, scale_factor=1e4, apply_log1p=True):
    library_sizes = np.array(csr_filtered.sum(axis=1)).flatten()
    library_sizes[library_sizes == 0] = 1
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")
    csr_filtered = D @ csr_filtered
    if apply_log1p:
        csr_filtered.data = np.log1p(csr_filtered.data)  # ← log1p (自然对数)
    return csr_filtered
```

✅ **完全一致！**

---

## ✅ 验证

两个数据集的归一化后数据应该具有相同的特性：

1. **Library size** = 10000 ± 误差
2. **所有值 ≥ 0** (log1p 后)
3. **CV ≈ 0** (测序深度差异消除)
4. **稀疏性保持**

---

## 📝 历史差异（已修复）

### ⚠️ 之前的问题（现已修复）：

在我们更新之前，`cellxgene_data_preprocess.py` 使用的是 **log2**：

```python
# 旧版本（已修复）
csr_filtered.data = np.log2(1 + csr_filtered.data)  # ❌ log2
```

**现在已统一为 log1p（自然对数）：**

```python
# 新版本（当前）
csr_filtered.data = np.log1p(csr_filtered.data)  # ✅ log1p (ln)
```

---

## 🎯 总结

✅ **两个数据集现在使用完全相同的归一化流程：**
- TP10K 归一化 (scale_factor = 10000)
- log1p 变换 (自然对数，ln(1+x))
- 对角稀疏矩阵优化
- 相同的默认参数 (min_genes = 200)

**区别只在于处理的阶段：**
- CellxGene：在初始预处理时归一化
- 3CA：在批次合并后再归一化

**最终输出数据完全一致！** ✅
