# MoERegressor 测试文档

## 概述

`test_moe_regressor.py` 是为 `MoERegressor` (Mixture of Experts Regressor) 模块编写的全面测试套件。

## 测试内容

该测试文件包含以下测试：

1. **Initialization（初始化测试）**
   - 验证基本属性（embedding_dim, num_experts, gate_temperature）
   - 验证网络结构（gate 网络和 experts）
   - 验证所有 experts 都是 RegressorExpert 实例

2. **Forward Shape（前向传播形状测试）**
   - 测试不同 batch_size 和 seq_len 下的输入输出形状
   - 验证输出形状为 (batch_size, seq_len)
   - 验证 gate_weights 形状为 (batch_size, seq_len, num_experts)

3. **Gate Weights Validity（Gate权重有效性测试）**
   - 验证 gate_weights 在 [0, 1] 范围内
   - 验证每个位置的 gate_weights 和为 1（softmax 特性）
   - 验证没有 NaN 或 Inf 值

4. **Different Expert Numbers（不同专家数量测试）**
   - 测试 2, 3, 4, 5, 8 个专家的配置
   - 验证所有配置都能正常工作

5. **Gate Temperature Effect（温度参数效果测试）**
   - 测试不同 temperature 值（0.1, 0.5, 1.0, 2.0, 5.0）
   - 验证所有 temperature 下 gate_weights 仍然有效

6. **Gradient Flow（梯度流测试）**
   - 验证梯度能够正确反向传播
   - 检查输入和所有参数的梯度
   - 确保没有 NaN 或 Inf 梯度

7. **Weight Initialization（权重初始化测试）**
   - 验证 gate 网络的权重初始化
   - 验证 expert 网络的权重初始化
   - 确保偏置初始化为零

8. **Output Range（输出范围测试）**
   - 验证输出值在合理范围内
   - 确保没有 NaN 或 Inf 输出

9. **Zero Input（零输入测试）**
   - 测试边界情况：全零输入
   - 验证模型在极端情况下的稳定性

10. **Determinism（确定性测试）**
    - 验证在 eval 模式下，相同输入产生相同输出
    - 确保模型的可重复性

11. **Dropout Effect（Dropout效果测试）**
    - 验证 dropout 在训练模式下有效
    - 验证 eval 模式下输出是确定性的

12. **RegressorExpert（Expert模块测试）**
    - 单独测试 RegressorExpert 模块
    - 验证输出形状为 (B, L, 1)

## 运行测试

### 使用 Python 直接运行

```bash
python tests/test_moe_regressor.py
```

### 使用 pytest 运行

```bash
pytest tests/test_moe_regressor.py -v
```

### 运行特定测试

```bash
pytest tests/test_moe_regressor.py::TestMoERegressor::test_initialization -v
```

## 已修复的 Bug

在编写测试过程中，发现并修复了以下 bug：

1. **Bug in `MoERegressor._initialize_weights()`** (src/deepsc/models/deepsc/model.py:551)
   - **问题**: 代码试图直接迭代 `expert`，但 `RegressorExpert` 不是可迭代对象
   ```python
   # 错误的代码
   for m in expert:
   ```
   - **修复**: 使用 `expert.modules()` 来迭代子模块
   ```python
   # 修复后的代码
   for m in expert.modules():
   ```

2. **Bug in `RegressorExpert.forward()`** (src/deepsc/models/deepsc/model.py:606-609)
   - **问题**: 第二子层中，尝试将 (B, L, 1) 的 fc3 输出与 (B, L, embedding_dim) 的 residual 相加，维度不匹配
   ```python
   # 错误的代码
   residual = x  # (B, L, embedding_dim)
   x = self.fc3(x)  # (B, L, 1)
   x = x + residual  # 维度不匹配！
   ```
   - **修复**: 移除第二子层的残差连接和不必要的 LayerNorm，直接返回 fc3 的输出
   ```python
   # 修复后的代码
   x = self.fc3(x)  # (B, L, 1)
   return x
   ```

## 测试结果

所有 12 个测试均已通过：

```
============================================================
Testing MoERegressor
============================================================

Using device: cuda

[Testing] Initialization... ✅ PASSED
[Testing] Forward shape... ✅ PASSED
[Testing] Gate weights validity... ✅ PASSED
[Testing] Different expert numbers... ✅ PASSED
[Testing] Gate temperature effect... ✅ PASSED
[Testing] Gradient flow... ✅ PASSED
[Testing] Weight initialization... ✅ PASSED
[Testing] Output range... ✅ PASSED
[Testing] Zero input... ✅ PASSED
[Testing] Determinism... ✅ PASSED
[Testing] Dropout effect... ✅ PASSED
[Testing] RegressorExpert... ✅ PASSED

============================================================
Test Results: 12 passed, 0 failed
============================================================
```

## 依赖

- PyTorch
- pytest（可选，用于更好的测试体验）

## 注意事项

- 测试会自动检测 CUDA 可用性，并在可用时使用 GPU
- 某些测试涉及随机性（如 dropout 测试），使用了随机种子以确保可重复性
- 梯度测试使用了 `.retain_grad()` 来捕获非叶子张量的梯度

## MoERegressor 架构说明

`MoERegressor` 是一个 Mixture of Experts 回归器，包含以下组件：

- **Gate 网络**: 决定每个 expert 的权重
  - 输入: embedding_dim
  - 隐藏层: embedding_dim // 2
  - 输出: num_experts (经过 softmax)

- **Expert 网络**: 多个并行的 RegressorExpert
  - 每个 expert 独立处理输入
  - 输出维度: (B, L, 1)

- **加权组合**: 使用 gate weights 对所有 experts 的输出进行加权求和
  - 最终输出: (B, L)

### 使用示例

```python
from src.deepsc.models.deepsc.model import MoERegressor
import torch

# 创建模型
model = MoERegressor(
    embedding_dim=256,
    dropout=0.1,
    number_of_experts=3,
    gate_temperature=1.0
)

# 准备输入
batch_size, seq_len = 4, 64
x = torch.randn(batch_size, seq_len, 256)

# 前向传播
output, gate_weights = model(x)

print(f"Output shape: {output.shape}")  # (4, 64)
print(f"Gate weights shape: {gate_weights.shape}")  # (4, 64, 3)
```
