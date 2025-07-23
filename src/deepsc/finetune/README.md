# DeepSC 细胞类型注释微调

这个目录包含了使用预训练的DeepSC模型进行细胞类型注释微调的完整流程。

## 📋 文件说明

- `cell_type_annotation_finetune.py`: 主微调脚本
- `finetune_data_loader.py`: 数据加载和预处理模块
- `test_checkpoint_loading.py`: 测试checkpoint加载功能
- `test_finetune_pipeline.py`: 测试完整微调pipeline
- `README.md`: 使用说明文档

## 🚀 快速开始

### 1. 环境准备

确保已激活正确的conda环境：
```bash
conda activate deepsc
```

需要安装以下依赖（如果尚未安装）：
```bash
pip install scanpy anndata pandas scikit-learn wandb
```

### 2. 数据准备

您需要准备以下数据：

#### h5ad文件要求：
- 包含单细胞RNA-seq表达数据
- `adata.var`中需要有基因名称（优先级：`gene_name` > `gene_symbols` > `var_names`）
- `adata.obs`中需要有`celltype`列包含细胞类型标签

#### 基因映射文件：
系统会自动使用：`/home/angli/DeepSC/scripts/preprocessing/gene_map.csv`

该文件包含以下列：
- `feature_name`: 基因名称
- `Ensembl id`: Ensembl ID
- `id`: 数值ID（模型使用）

### 3. 运行测试

在使用真实数据之前，建议先运行测试：

```bash
cd /home/angli/baseline/DeepSC/src/deepsc/finetune
python test_finetune_pipeline.py
```

这会：
- 创建虚拟的h5ad测试数据
- 测试数据加载功能
- 测试模型集成
- 验证完整pipeline

### 4. 运行微调

#### 使用测试数据：
```bash
python cell_type_annotation_finetune.py data_path=/home/angli/baseline/DeepSC/data/finetune/test_data.h5ad
```

#### 使用您的真实数据：
```bash
python cell_type_annotation_finetune.py data_path=path/to/your/data.h5ad
```

## ⚙️ 配置说明

主要配置文件：`../../../configs/finetune/finetune.yaml`

### 关键配置项：

```yaml
# 数据路径
data_path: "path/to/your/data.h5ad"

# 模型配置（需要匹配预训练checkpoint）
model:
  embedding_dim: 256
  num_genes: 34682    # 基因数量
  num_layers: 10      # Transformer层数
  num_bins: 5         # 表达值离散化bin数

# 训练配置
learning_rate: 1e-4   # 微调使用较低学习率
batch_size: 16
epoch: 20
patience: 5           # 早停patience

# Checkpoint路径
checkpoint_path: "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"
```

## 📊 数据处理流程

### 1. 数据加载与质控
- 从h5ad文件加载数据
- 过滤低表达基因（至少在10个细胞中表达）
- 过滤低质量细胞（至少表达200个基因）

### 2. 基因映射
- 将h5ad中的基因名映射到gene_map.csv中的数值ID
- 只保留映射表中存在的基因
- 输出映射覆盖率统计

### 3. 细胞类型处理
- 自动从`celltype`列提取标签
- 使用LabelEncoder编码为数值标签
- 输出类别分布统计

### 4. 表达数据预处理
- 处理稀疏矩阵格式
- 为每个细胞选择表达量最高的前1024个基因
- 按细胞独立进行表达值离散化

### 5. 数据增强（仅用于分类）
- **不进行掩码**：与预训练不同，微调时保留所有基因信息
- 添加CLS token用于分类
- 截断或填充到固定长度

## 🧬 关于掩码的说明

**在细胞类型分类微调中，我们不使用掩码**，原因：

1. **目标不同**：预训练使用掩码进行自监督学习，微调进行监督分类
2. **信息完整性**：分类需要利用所有基因的完整表达信息
3. **避免信息损失**：掩码会丢失对分类重要的基因表达特征

## 📈 训练流程

### 1. 模型架构
```
预训练DeepSC Encoder (冻结) → 平均池化 → 分类器 (可训练)
```

### 2. 优化策略
- **学习率调度**：Warmup + Cosine Annealing
- **早停**：基于验证集准确率，patience=5
- **权重衰减**：1e-5

### 3. 评估指标
- Accuracy（准确率）
- F1-score（宏平均）
- Precision（宏平均）
- Recall（宏平均）

## 📂 输出文件

训练过程会生成：

```
ckpts/finetune/
├── best_model.pth          # 最佳模型checkpoint
└── ...

outputs/finetune/
├── logs/
└── ...
```

最佳模型包含：
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `best_val_acc`: 最佳验证准确率
- `config`: 训练配置

## 🔧 故障排除

### 常见问题：

1. **基因映射覆盖率低**
   - 检查h5ad文件中的基因名格式
   - 确认基因名列的名称（gene_name/gene_symbols）

2. **内存不足**
   - 减小batch_size
   - 减小max_length
   - 减少num_workers

3. **找不到celltype列**
   - 检查h5ad文件的obs中是否有celltype列
   - 确认列名拼写正确

4. **Checkpoint加载失败**
   - 确认checkpoint路径正确
   - 检查模型配置是否匹配checkpoint

### 调试技巧：

```bash
# 详细日志
python cell_type_annotation_finetune.py --config-path . --config-name finetune hydra.verbose=true

# 小批次测试
python cell_type_annotation_finetune.py batch_size=2 epoch=1

# 使用CPU（调试用）
CUDA_VISIBLE_DEVICES="" python cell_type_annotation_finetune.py
```

## 📊 性能监控

如果配置了wandb，可以实时监控：
- 训练/验证损失
- 训练/验证准确率
- 各项评估指标
- 学习率变化

## 🎯 使用建议

1. **数据质量**：确保细胞类型标注准确，数据质控充分
2. **类别平衡**：检查各细胞类型的样本数量分布
3. **超参数调优**：根据数据规模调整学习率和batch_size
4. **验证策略**：使用分层采样保持类别平衡
5. **模型解释**：分析预测错误的案例，优化数据或超参数

## 📝 引用

如果使用此代码，请引用相关论文和DeepSC项目。
