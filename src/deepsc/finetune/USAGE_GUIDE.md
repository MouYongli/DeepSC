# DeepSC Perturbation Prediction - 使用指南

## 概述

本模块实现了基于scGPT逻辑的DeepSC perturbation预测,并结合了pp_new.py中的基因对齐流程。

## 关键配置文件

### 1. 预训练模型配置
在 `/home/angli/DeepSC/configs/pp/pp.yaml` 中已经配置:

```yaml
# 预训练模型
pretrained_model: True
pretrained_model_path: "/home/angli/baseline/DeepSC/results/pretraining_1201/DeepSC_11_0.ckpt"

# 词汇表
csv_path: "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"
```

### 2. 模型结构配置
在 `/home/angli/DeepSC/configs/pp/model/deepsc.yaml` 中配置:

```yaml
_target_: src.deepsc.models.generation_deepsc.model.DeepSC
embedding_dim: 256
num_genes: 35210
num_layers: 10
num_heads: 8
use_moe_regressor: True  # 使用MoE回归器
```

### 3. 训练配置

```yaml
# 数据集
data_name: "norman"  # norman, adamson, replogle_rpe1_essential
split: "simulation"
data_path: "./data"

# 训练超参数
batch_size: 32
learning_rate: 3e-4
epoch: 10
grad_acc: 20

# 基因选择策略
include_zero_gene: "all"  # all 或 batch-wise
data_length: 3000  # 最大序列长度

# Loss设置
use_scgpt_loss: True
enable_mse: True
target_mse_loss_weight: 3.0
```

## 使用方法

### 方法1: 使用Hydra配置运行 (推荐)

```bash
cd /home/angli/DeepSC

# 基本运行
python src/deepsc/finetune/run_perturbation_hydra.py

# 覆盖配置参数
python src/deepsc/finetune/run_perturbation_hydra.py \
    data_name=adamson \
    batch_size=64 \
    learning_rate=1e-4 \
    epoch=15

# 使用不同的预训练模型
python src/deepsc/finetune/run_perturbation_hydra.py \
    pretrained_model_path=/path/to/your/checkpoint.pt

# 多GPU训练
python src/deepsc/finetune/run_perturbation_hydra.py \
    num_device=4 \
    batch_size=16
```

### 方法2: 使用命令行参数运行

```bash
python examples/run_perturbation_finetune.py \
    --data_name norman \
    --data_path ./data \
    --csv_path /home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv \
    --embedding_dim 256 \
    --num_layers 10 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --epoch 10 \
    --pretrained_model \
    --pretrained_model_path /home/angli/baseline/DeepSC/results/pretraining_1201/DeepSC_11_0.ckpt \
    --devices 1
```

### 方法3: 在代码中使用

```python
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric
from hydra.utils import instantiate

from deepsc.finetune.perturbation_finetune import PerturbationPredictor

# 加载配置
with hydra.initialize(config_path="../configs/pp", version_base=None):
    cfg = hydra.compose(config_name="pp")

# 设置Fabric
fabric = Fabric(accelerator="gpu", devices=1)
fabric.launch()

# 实例化模型 (使用Hydra自动实例化)
model = instantiate(cfg.model)

# 创建Predictor
predictor = PerturbationPredictor(
    args=cfg,
    fabric=fabric,
    model=model
)

# 训练
predictor.train()

# 可视化
predictor.plot_predictions()
```

## 模型接口说明

### DeepSC模型的forward接口

DeepSC模型已经支持perturbation预测所需的接口:

```python
regression_output, gene_emb, expr_emb = model(
    gene_ids=mapped_input_gene_ids,      # (batch_size, seq_len) - 词汇表ID
    expression_bin=discrete_input_bins,   # (batch_size, seq_len) - 离散化的表达值
    normalized_expr=input_values,         # (batch_size, seq_len) - 归一化的表达值
    input_pert_flags=input_pert_flags,    # (batch_size, seq_len) - perturbation标记
)
```

返回值:
- `regression_output`: (batch_size, seq_len) - 预测的表达值
- `gene_emb`: (batch_size, seq_len, embedding_dim) - 基因嵌入
- `expr_emb`: (batch_size, seq_len, embedding_dim) - 表达嵌入

## 数据流程

### 1. 基因对齐

```python
# 从CSV加载词汇表
vocab = build_vocab_from_csv(csv_path)  # ~35210个基因

# 数据集基因
dataset_genes = pert_data.adata.var["gene_name"].tolist()  # 例如: 5000个基因

# 构建映射
gene_ids = build_gene_ids_for_dataset(dataset_genes, vocab)
# gene_ids: array of vocab indices, 0 for unmatched genes

# 匹配统计
matched = np.sum(gene_ids != 0)  # 例如: 4800/5000 (96%)
```

### 2. Perturbation标记构建

```python
# 从GEARS数据中提取perturbation信息
# batch_data.pert = ["SAMD1+ZBTB1", "FEV+ctrl", ...]

# 构建标记矩阵
pert_flags = torch.zeros(batch_size, n_genes)
for i, pert_str in enumerate(batch_data.pert):
    for gene_name in pert_str.split("+"):
        if gene_name != "ctrl":
            gene_idx = name2col[gene_name]
            pert_flags[i, gene_idx] = 1
```

### 3. 训练数据处理

```python
# 1. 提取原始表达值
ori_gene_values = batch_data.x[:, 0].view(batch_size, n_genes)
target_gene_values = batch_data.y

# 2. 选择基因 (all 或 batch-wise)
if include_zero_gene == "all":
    input_gene_ids = torch.arange(n_genes)
else:
    input_gene_ids = ori_gene_values.nonzero()[:, 1].unique()

# 限制长度
if len(input_gene_ids) > max_seq_len:
    input_gene_ids = input_gene_ids[:max_seq_len]

# 3. 提取对应的值
input_values = ori_gene_values[:, input_gene_ids]
target_values = target_gene_values[:, input_gene_ids]
input_pert_flags = pert_flags[:, input_gene_ids]

# 4. 映射到词汇表ID
mapped_input_gene_ids = gene_ids[input_gene_ids]

# 5. 离散化表达值
discrete_bins = discretize_expression(input_values, num_bins=5)

# 6. 前向传播
regression_output, _, _ = model(
    gene_ids=mapped_input_gene_ids,
    expression_bin=discrete_bins,
    normalized_expr=input_values,
    input_pert_flags=input_pert_flags,
)

# 7. 计算loss
loss = criterion(regression_output, target_values)
```

## 输出结构

运行后会在Hydra配置的输出目录中生成:

```
/home/angli/DeepSC/results/perturbation_prediction/
└── 2024-12-09/
    └── 14-30-00/
        ├── checkpoints/
        │   ├── epoch_1.pt
        │   ├── epoch_2.pt
        │   ├── ...
        │   └── best_model.pt
        ├── logs/
        ├── visualizations/
        │   ├── SAMD1+ZBTB1.png
        │   └── KCTD16+ctrl.png
        └── test_metrics.json
```

## 评估指标

模型会计算以下指标:

1. **Pearson相关系数**: 总体预测与真实表达的相关性
2. **Pearson delta**: 表达变化(相对于control)的相关性
3. **Pearson delta DE**: 仅在差异表达基因上的相关性
4. **MSE**: 均方误差
5. **Top 20 DE metrics**: 在top 20差异表达基因上的指标

示例输出:
```json
{
  "pearson": 0.78,
  "pearson_delta": 0.72,
  "pearson_delta_de": 0.85,
  "mse": 0.024
}
```

## 常见问题

### Q1: 基因匹配率低怎么办?
如果匹配率 < 80%:
- 检查CSV词汇表文件是否正确
- 确认数据集基因名称格式与词汇表一致
- 考虑更新词汇表或使用基因ID映射

### Q2: 如何调整训练超参数?
在配置文件中修改或在命令行覆盖:
```bash
python src/deepsc/finetune/run_perturbation_hydra.py \
    learning_rate=1e-4 \
    batch_size=64 \
    grad_acc=10
```

### Q3: 内存不足怎么办?
- 减小 `batch_size`
- 减小 `data_length` (最大序列长度)
- 增加 `grad_acc` (梯度累积)
- 使用 `include_zero_gene=batch-wise`

### Q4: 如何添加新的数据集?
1. 确保数据集在GEARS中可用
2. 修改配置: `data_name=your_dataset`
3. 添加对应的`perts_to_plot`配置

### Q5: 训练时loss是NaN?
- 降低学习率 (例如从3e-4降到1e-4)
- 检查数据预处理是否正确
- 启用梯度裁剪 (已默认启用)

## 与scGPT和pp_new.py的对比

| 功能 | scGPT | pp_new.py | perturbation_finetune.py |
|------|-------|-----------|-------------------------|
| 基因对齐 | GeneVocab | build_gene_ids_for_dataset | build_gene_ids_for_dataset ✓ |
| Perturbation标记 | construct_pert_flags | construct_pert_flags | construct_pert_flags ✓ |
| 模型 | TransformerGenerator | DeepSC | DeepSC ✓ |
| 配置管理 | Python变量 | Hydra | Hydra ✓ |
| 预训练模型 | scGPT checkpoint | DeepSC checkpoint | DeepSC checkpoint ✓ |
| MoE回归器 | ✗ | ✓ | ✓ |
| 分布式训练 | PyTorch DDP | Lightning Fabric | Lightning Fabric ✓ |

## 下一步

1. **运行基准测试**: 在norman数据集上测试
2. **调优超参数**: 根据验证集结果调整
3. **多数据集评估**: 在adamson、replogle等数据集上测试
4. **消融实验**: 测试不同的基因选择策略、loss权重等

## 参考

- scGPT代码: `/home/angli/old_tasks/scGPT/examples/finetune_perturbation.py`
- pp_new.py: `/home/angli/DeepSC/src/deepsc/finetune/pp_new.py`
- 配置文件: `/home/angli/DeepSC/configs/pp/pp.yaml`
- GEARS文档: https://github.com/snap-stanford/GEARS
