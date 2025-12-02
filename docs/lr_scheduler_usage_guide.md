# Learning Rate Scheduler 使用指南

**更新日期**: 2025-12-01

---

## 概述

DeepSC现在支持两种学习率调度策略:

1. **`constant`**: 恒定学习率,从头到尾学习率不变
2. **`cosine`**: Cosine annealing with warmup (默认)

---

## 快速开始

### 方法1: 修改主配置文件

编辑 `configs/finetune/finetune.yaml`:

```yaml
# 使用恒定学习率
lr_scheduler_type: "constant"
learning_rate: 1e-4

# 或使用cosine annealing (默认)
lr_scheduler_type: "cosine"
learning_rate: 1e-4
warmup_ratio: 0.03
```

### 方法2: 命令行覆盖

```bash
# 使用恒定学习率
python -m src.deepsc.finetune.finetune lr_scheduler_type=constant learning_rate=1e-4

# 使用cosine annealing
python -m src.deepsc.finetune.finetune lr_scheduler_type=cosine learning_rate=1e-4
```

### 方法3: 使用示例配置文件

```bash
# hPancreas with constant LR
python -m src.deepsc.finetune.finetune --config-name=examples/constant_lr_hpancreas

# Myeloid with constant LR
python -m src.deepsc.finetune.finetune --config-name=examples/constant_lr_myeloid
```

---

## 详细配置说明

### 1. Constant Learning Rate (恒定学习率)

**配置**:
```yaml
lr_scheduler_type: "constant"
learning_rate: 1e-4  # 学习率保持不变
```

**特点**:
- ✅ 简单直接,无需调参
- ✅ 训练稳定,可预测
- ✅ 适合小数据集或简单任务
- ✅ 不需要warmup
- ⚠️ 可能需要更多epochs

**运行时输出**:
```
================================================================================
Using CONSTANT learning rate: 0.0001
No learning rate scheduling will be applied.
================================================================================
```

**推荐场景**:
- hPancreas (15类,简单任务)
- 快速实验和原型
- 训练不稳定时尝试

**推荐配置**:
```yaml
# 简单数据集
lr_scheduler_type: "constant"
learning_rate: 1e-4
epoch: 10

# 复杂数据集
lr_scheduler_type: "constant"
learning_rate: 5e-5  # 更低的LR
epoch: 20            # 更多epochs
```

---

### 2. Cosine Annealing (余弦退火)

**配置**:
```yaml
lr_scheduler_type: "cosine"
learning_rate: 1e-4
warmup_ratio: 0.03  # 3% steps用于warmup
```

**特点**:
- 📈 学习率动态变化
- 📈 包含warmup和cosine decay
- ✅ 通常能获得更好的最终性能
- ⚠️ 需要调整warmup_ratio
- ⚠️ 可能在某些数据集上不稳定

**运行时输出**:
```
================================================================================
Using COSINE ANNEALING with warmup:
  - Warmup steps: 150
  - Total steps: 5000
  - Initial LR: 0.0001
================================================================================
```

**学习率变化曲线**:
```
LR
^
|     /\
|    /  \___
|   /       \___
|  /            \___
| /                 \___
|/________________________> Steps
  ^       ^              ^
  |       |              |
warmup  restart1      restart2
```

**推荐场景**:
- 大规模数据集
- 长时间训练(50+ epochs)
- 追求最佳性能

**推荐配置**:
```yaml
lr_scheduler_type: "cosine"
learning_rate: 1e-4
warmup_ratio: 0.03
epoch: 30
```

---

## 不同数据集的推荐配置

### hPancreas (15类, 11,847细胞)

#### 选项1: Constant LR (推荐)
```yaml
lr_scheduler_type: "constant"
learning_rate: 1e-4
epoch: 10
batch_size: 32
grad_acc: 20
```

**预期性能**: 95-97% 准确率

#### 选项2: Cosine Annealing
```yaml
lr_scheduler_type: "cosine"
learning_rate: 1e-4
warmup_ratio: 0.03
epoch: 10
batch_size: 32
grad_acc: 20
```

**预期性能**: 96-98% 准确率

---

### Myeloid (39类, 56,911细胞)

#### 选项1: Constant LR (推荐用于稳定性)
```yaml
lr_scheduler_type: "constant"
learning_rate: 5e-5  # 更低的LR
epoch: 20            # 更多epochs
batch_size: 32
grad_acc: 20
```

**预期性能**: 73-76% 准确率
**优点**: 训练稳定,避免波动

#### 选项2: Cosine Annealing
```yaml
lr_scheduler_type: "cosine"
learning_rate: 1e-4
warmup_ratio: 0.05  # 更长的warmup
epoch: 30
batch_size: 32
grad_acc: 20
```

**预期性能**: 74-77% 准确率
**注意**: 可能出现训练不稳定

---

### Zheng (11类, 52,748细胞, 高稀疏度)

#### 推荐: Constant LR
```yaml
lr_scheduler_type: "constant"
learning_rate: 1e-4
epoch: 15
batch_size: 32
grad_acc: 20
```

**原因**: 高稀疏度(97%)需要稳定的学习率

---

### Segerstolpe (需要先归一化!)

```yaml
# 必须先对数据进行log1p归一化!
lr_scheduler_type: "constant"
learning_rate: 5e-5
epoch: 20
batch_size: 32
grad_acc: 20
```

---

## 学习率选择指南

### 基本原则

| 数据集复杂度 | 细胞类型数 | 推荐LR (constant) | 推荐LR (cosine) |
|------------|-----------|------------------|-----------------|
| 简单 | < 20 | 1e-4 | 1e-4 |
| 中等 | 20-40 | 5e-5 | 1e-4 |
| 复杂 | > 40 | 1e-5 | 5e-5 |

### 调整建议

**如果训练loss下降太慢**:
- Constant LR: 增大learning_rate (如1e-4 → 3e-4)
- Cosine: 增大learning_rate或减少warmup_ratio

**如果训练不稳定**:
- Constant LR: 降低learning_rate (如1e-4 → 5e-5)
- Cosine: 考虑切换到constant LR

**如果验证集性能不提升**:
- 增加epochs
- 降低learning_rate
- 尝试不同的scheduler类型

---

## 实验对比

### 实验记录

| 实验 | 数据集 | Scheduler | LR | Epochs | 最佳准确率 | 训练稳定性 |
|------|--------|-----------|-----|--------|-----------|-----------|
| 1 | hPancreas | constant | 1e-4 | 10 | 97.09% | ⭐⭐⭐⭐⭐ |
| 2 | hPancreas | cosine | 1e-4 | 10 | 97.30% | ⭐⭐⭐⭐⭐ |
| 3 | Myeloid | cosine | 1e-4 | 10 | 73.86% | ⭐⭐⭐ (不稳定) |
| 4 | Myeloid | constant | 5e-5 | 20 | ? | 待测试 |
| 5 | Zheng | cosine | 1e-4 | 10 | 79.04% | ⭐⭐⭐⭐ |

**结论**:
- hPancreas: 两种策略都很好,cosine略优
- Myeloid: cosine不稳定,建议尝试constant
- Zheng: cosine表现良好

---

## 故障排查

### 问题1: "Unknown lr_scheduler_type" 错误

**原因**: 配置文件中lr_scheduler_type值无效

**解决**:
```yaml
# 错误
lr_scheduler_type: "linear"  # ❌ 不支持

# 正确
lr_scheduler_type: "constant"  # ✅
# 或
lr_scheduler_type: "cosine"    # ✅
```

### 问题2: Constant LR但学习率仍在变化

**检查**: 确认运行日志中显示:
```
Using CONSTANT learning rate: 0.0001
No learning rate scheduling will be applied.
```

如果看到cosine相关信息,检查配置文件是否正确加载。

### 问题3: 训练不稳定

**尝试**:
1. 切换到constant LR
2. 降低learning_rate
3. 增加grad_acc (减少更新频率)
4. 检查数据是否需要归一化

---

## 高级技巧

### 1. 动态切换策略

先用constant LR快速收敛,再用cosine精调:

```bash
# 阶段1: 快速收敛 (epoch 1-10)
python -m src.deepsc.finetune.finetune \
    lr_scheduler_type=constant \
    learning_rate=1e-4 \
    epoch=10

# 阶段2: 精调 (epoch 11-20)
python -m src.deepsc.finetune.finetune \
    lr_scheduler_type=cosine \
    learning_rate=5e-5 \
    epoch=20 \
    resume_last_training=True
```

### 2. 学习率范围测试

快速测试不同学习率:

```bash
for lr in 1e-5 5e-5 1e-4 5e-4; do
    python -m src.deepsc.finetune.finetune \
        lr_scheduler_type=constant \
        learning_rate=$lr \
        epoch=5 \
        run_name="lr_test_${lr}"
done
```

### 3. 监控学习率

在训练循环中,学习率会被自动记录到wandb。
查看 "learning_rate" 曲线来验证调度器是否正常工作。

---

## 总结

### 快速决策树

```
开始
  │
  ├─ 数据集简单 (< 20类)?
  │  ├─ 是 → 使用 constant @ 1e-4
  │  └─ 否 → 继续
  │
  ├─ 训练稳定性重要?
  │  ├─ 是 → 使用 constant @ 5e-5
  │  └─ 否 → 使用 cosine @ 1e-4
  │
  └─ 训练时间充足 (>20 epochs)?
     ├─ 是 → 使用 cosine @ 1e-4
     └─ 否 → 使用 constant @ 1e-4
```

### 默认推荐

**初次尝试**: 使用 `constant @ 1e-4`
- 简单
- 稳定
- 易于调试

**追求性能**: 使用 `cosine @ 1e-4`
- 可能获得更好的结果
- 需要更多调参

---

## 参考资料

- 配置文件: `configs/finetune/finetune.yaml`
- 示例配置: `configs/finetune/examples/`
- 源代码: `src/deepsc/finetune/cell_type_annotation.py` (第199-254行)

---

**最后更新**: 2025-12-01
**维护者**: DeepSC Team
