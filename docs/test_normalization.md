# TP10K 归一化测试指南

## 🧪 测试方法

### 方法 1: 单元测试（推荐先做）

运行单元测试来验证归一化的数学逻辑：

```bash
# 运行所有归一化测试
pytest tests/test_tp10k_normalization.py -v

# 运行特定测试
pytest tests/test_tp10k_normalization.py::TestTP10KNormalization::test_basic_normalization -v
```

**测试内容：**
- ✅ 基本的 TP10K 归一化数学公式是否正确
- ✅ log1p 变换是否正确应用
- ✅ 稀疏性是否保持
- ✅ 低质量细胞过滤是否正常
- ✅ 零值处理是否正确
- ✅ 对角矩阵优化与朴素实现结果是否一致

**期望结果：** 所有测试都应该 PASS ✅

---

### 方法 2: 真实数据验证

对实际数据进行归一化并验证：

```bash
# 1. 先对数据进行归一化
python -m deepsc.data.preprocessing.batch_normalize \
  --input_dir /path/to/raw/data \
  --output_dir /path/to/normalized/data \
  --scale_factor 10000 \
  --min_genes 200

# 2. 验证归一化结果
python scripts/test/validate_normalization.py \
  --input /path/to/normalized/data/your_file_norm.npz \
  --output_plot validation_result.png
```

**验证内容：**
- ✅ Library size 是否接近 10000
- ✅ 稀疏性是否保持
- ✅ log1p 后的值是否都非负
- ✅ 每个细胞表达的基因数分布

**期望看到：**
```
============================================================
Normalization Validation Report
============================================================

1. Loading normalized data from: ...
   Shape: (50000, 30000) (cells × genes)
   Non-zero elements: 15,000,000 (1.00%)

2. Checking library sizes...
   Mean library size:   10000.00
   Std library size:    5.23
   Min library size:    9995.12
   Max library size:    10004.88
   ✅ Library sizes are close to 10000

3. Checking sparsity...
   Sparsity: 99.00% zeros

4. Checking value ranges (after log1p)...
   Min value:  0.0000
   Max value:  9.2103
   Mean value: 1.2345
   Median:     0.8765
   ✅ All values are non-negative (correct for log1p)

5. Checking genes per cell...
   Mean genes/cell: 300
   Min genes/cell:  200
   Max genes/cell:  1500

============================================================
Validation complete!
============================================================
```

---

### 方法 3: 归一化前后对比

比较归一化前后的数据分布变化：

```bash
python scripts/test/validate_normalization.py \
  --input /path/to/normalized/data.npz \
  --before /path/to/raw/data.npz \
  --output_plot comparison.png
```

**期望看到：**
```
============================================================
Before/After Comparison
============================================================

Before normalization:
  Mean library size: 5243
  Std library size:  2891
  CV (coefficient of variation): 0.5514  # 变异系数很大

After TP10K normalization:
  Mean library size: 10000
  Std library size:  0.52
  CV (coefficient of variation): 0.0001  # 变异系数接近0

✅ CV reduced by 99.98%
```

**关键指标：**
- CV (coefficient of variation) 应该从 ~0.3-0.6 降低到接近 0
- 这说明测序深度差异被成功消除

---

## 📊 可视化检查

运行验证脚本会生成 4 个图表：

### 1. Library Size 分布
- 应该看到一个很窄的峰值集中在 10000 附近
- 所有细胞的 library size 应该基本一致

### 2. 表达值分布（log1p 后）
- 应该是右偏分布
- 没有负值
- 大部分值在 0-8 之间

### 3. 每个细胞的基因数分布
- 应该符合单细胞数据的典型分布
- 最小值应该 >= min_genes 参数（默认 200）

### 4. 稀疏性模式
- 应该看到大量的黑色（零值）
- 白色点（非零值）应该随机分布

---

## ✅ 通过标准

### 归一化被认为是正确的，如果：

1. **Library size 检查**
   - 平均值接近 10000（误差 < 1%）
   - 标准差很小（< 50）
   - CV < 0.01

2. **数学正确性**
   - 所有单元测试通过
   - 没有 NaN 或 Inf 值
   - log1p 后所有值 >= 0

3. **稀疏性保持**
   - 归一化前后非零元素数量一致
   - 稀疏度不变

4. **细胞过滤**
   - 表达基因数 < min_genes 的细胞被过滤
   - 剩余细胞数量符合预期

---

## 🚨 常见问题

### Q1: Library size 不是准确的 10000？
**A:** 由于浮点数精度，会有微小误差（< 1%），这是正常的。

### Q2: log1p 后为什么还有很多零值？
**A:** log1p(0) = 0，所以原始数据中的零值在 log1p 后仍然是零。

### Q3: 如何验证对角矩阵优化是否正确？
**A:** 运行 `test_diagonal_matrix_optimization` 测试，它会比较优化方法和朴素方法的结果。

---

## 🔬 手动验证示例

如果你想手动验证某个细胞的归一化：

```python
import numpy as np
from scipy import sparse

# 加载归一化后的数据
data = sparse.load_npz("normalized.npz")

# 选择第一个细胞
cell_0 = data[0].toarray().flatten()

# 还原 log1p
cell_0_original = np.expm1(cell_0)

# 计算 library size
library_size = cell_0_original.sum()

print(f"Cell 0 library size: {library_size}")  # 应该接近 10000

# 验证归一化公式（以第一个基因为例）
# 假设原始 count 是 x，归一化后应该是：
# y = log1p(x / library_size * 10000)
```

---

## 📝 推荐测试流程

1. **先运行单元测试** → 确保代码逻辑正确
2. **对小样本数据测试** → 归一化 100-1000 个细胞
3. **可视化检查** → 生成验证图表
4. **对完整数据集测试** → 运行完整的预处理流程
5. **下游任务验证** → 检查归一化后的数据是否改善模型性能

---

## 💡 Tips

- 如果数据已经归一化过，不要重复归一化
- 保存归一化前的原始数据用于对比
- 记录归一化的参数（scale_factor, min_genes 等）
- 不同数据集可能需要不同的 min_genes 阈值
