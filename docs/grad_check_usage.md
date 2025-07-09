# 梯度检查功能使用指南

## 概述

`check_grad_flow` 函数是一个用于诊断深度学习模型梯度传导问题的工具。它可以帮助你：

- 检测哪些参数没有梯度（grad is None）
- 识别梯度为0的参数
- 监控有效梯度的数值范围
- 在训练过程中及时发现梯度消失或爆炸问题

## 功能特点

✅ **自动检测**: 自动遍历模型所有参数并检查梯度状态
✅ **详细报告**: 提供每个参数的梯度统计信息
✅ **分类统计**: 将参数按梯度状态分类（有效/零梯度/无梯度）
✅ **集成监控**: 可与wandb等工具集成进行训练监控

## 使用方法

### 1. 基本使用

```python
import torch
import torch.nn as nn
from deepsc.utils.utils import check_grad_flow

# 创建模型和损失
model = YourModel()
loss = compute_loss(model_output, target)

# 执行梯度检查
grad_stats = check_grad_flow(model, loss, verbose=True)
```

### 2. 在训练循环中使用

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 梯度检查（每100步检查一次）
        if batch_idx % 100 == 0:
            grad_stats = check_grad_flow(model, loss, verbose=False)
            print(f"有效梯度参数: {len(grad_stats['ok'])}")

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 在DeepSC项目中使用

在配置文件中启用梯度检查：

```yaml
# configs/pretrain/pretrain.yaml
check_grad_flow: True  # 启用梯度检查
```

训练时会自动每100步检查一次梯度状态，并将结果记录到wandb。

## 输出说明

### 控制台输出示例

```
============================================================
➡️ [检查开始] 反向传播中梯度传导情况...
[✅ OK  ] model.layer1.weight: grad max=1.2345e-03, min=1.2345e-06
[✅ OK  ] model.layer1.bias: grad max=2.3456e-04, min=2.3456e-04
[⚠️ ZERO] model.layer2.weight: grad == 0
[❌ NONE ] model.frozen_layer.weight: grad is None
------------------------------------------------------------
✅ 有效梯度参数数：2
⚠️ 梯度为0的参数数：1
❌ grad is None 的参数数：1
============================================================
```

### 返回值

函数返回一个字典，包含：

```python
{
    "ok": ["param1", "param2"],      # 有效梯度的参数名列表
    "zero": ["param3"],              # 梯度为0的参数名列表
    "none": ["param4"]               # 无梯度的参数名列表
}
```

## 常见问题诊断

### 1. 梯度消失 (Gradient Vanishing)

**症状**: 大量参数梯度为0或接近0
**可能原因**:
- 学习率过小
- 激活函数选择不当（如sigmoid在深层网络）
- 权重初始化不当

**解决方案**:
```python
# 检查梯度范围
grad_stats = check_grad_flow(model, loss)
if len(grad_stats["zero"]) > len(grad_stats["ok"]):
    print("⚠️ 可能存在梯度消失问题")
    # 调整学习率或使用更好的初始化
```

### 2. 梯度爆炸 (Gradient Exploding)

**症状**: 梯度值异常大
**解决方案**:
```python
# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. 参数冻结

**症状**: 某些参数显示 "grad is None"
**原因**: 参数被设置为 `requires_grad=False`
**检查方法**:
```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"参数 {name} 被冻结")
```

## 高级用法

### 1. 自定义检查频率

```python
# 在训练循环中
if batch_idx % 50 == 0:  # 每50步检查一次
    grad_stats = check_grad_flow(model, loss, verbose=False)
```

### 2. 与wandb集成

```python
import wandb

grad_stats = check_grad_flow(model, loss, verbose=False)
wandb.log({
    "grad_check/ok_params": len(grad_stats["ok"]),
    "grad_check/zero_params": len(grad_stats["zero"]),
    "grad_check/none_params": len(grad_stats["none"]),
})
```

### 3. 条件检查

```python
# 只在损失异常时检查梯度
if loss.item() > threshold:
    grad_stats = check_grad_flow(model, loss, verbose=True)
```

## 注意事项

1. **性能影响**: 梯度检查会增加少量计算开销，建议在调试时使用
2. **内存使用**: 检查过程会创建梯度副本，注意内存使用
3. **分布式训练**: 在分布式环境中，确保只在主进程执行检查
4. **梯度累积**: 在使用梯度累积时，注意检查的时机

## 故障排除

### 问题1: 所有参数都显示 "grad is None"

**可能原因**:
- 损失函数没有正确连接到模型参数
- 模型参数被冻结

**检查方法**:
```python
# 检查损失是否与模型参数相连
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# 检查模型参数状态
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

### 问题2: 梯度检查后模型状态异常

**解决方案**: 在检查前保存模型状态，检查后恢复
```python
# 保存原始状态
original_grads = {name: param.grad.clone() if param.grad is not None else None
                 for name, param in model.named_parameters()}

# 执行梯度检查
check_grad_flow(model, loss)

# 恢复原始状态
for name, param in model.named_parameters():
    param.grad = original_grads[name]
```
