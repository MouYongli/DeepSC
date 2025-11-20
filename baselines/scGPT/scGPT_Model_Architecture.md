# scGPT Model Architecture Documentation

## 概述

scGPT是一个用于单细胞RNA测序(scRNA-seq)数据分析的深度学习框架。该项目包含多个模型架构,用于处理单细胞基因表达数据的不同任务,如细胞类型注释、基因表达预测、扰动预测等。

## 目录结构

```
scgpt/model/
├── __init__.py                    # 模块导出
├── model.py                       # 核心TransformerModel
├── generation_model.py            # 生成式模型(扰动预测)
├── multiomic_model.py             # 多组学模型
├── dsbn.py                        # 领域特定批归一化
└── grad_reverse.py                # 梯度反转层
```

## 核心架构

### 1. TransformerModel (model.py)

这是scGPT的核心模型,基于Transformer架构设计。

#### 1.1 主要组件

```
TransformerModel
├── GeneEncoder              # 基因编码器
├── ValueEncoder             # 表达值编码器
│   ├── ContinuousValueEncoder    # 连续值编码
│   ├── CategoryValueEncoder      # 类别值编码
│   └── Identity (scaling模式)     # 缩放模式
├── BatchLabelEncoder        # 批次标签编码器(可选)
├── DomainSpecificBatchNorm  # 领域特定批归一化(可选)
├── TransformerEncoder       # Transformer编码器
│   ├── FlashTransformerEncoderLayer  # Flash Attention版本
│   └── TransformerEncoderLayer       # 标准版本
├── ExprDecoder              # 表达值解码器
├── ClsDecoder               # 分类解码器
├── MVCDecoder               # 掩码值预测解码器(可选)
└── AdversarialDiscriminator # 对抗判别器(可选)
```

#### 1.2 关键参数

- **ntoken**: 基因词汇表大小
- **d_model**: 嵌入维度(默认512)
- **nhead**: 注意力头数(默认8)
- **d_hid**: 前馈网络隐藏层维度
- **nlayers**: Transformer层数
- **input_emb_style**: 输入嵌入风格("continuous", "category", "scaling")
- **cell_emb_style**: 细胞嵌入风格("cls", "avg-pool", "w-pool")
- **use_fast_transformer**: 是否使用Flash Attention

#### 1.3 前向传播流程

```python
输入: (gene_ids, expression_values, padding_mask, batch_labels)
    ↓
1. 基因编码
   gene_ids → GeneEncoder → gene_embeddings (batch, seq_len, d_model)
    ↓
2. 表达值编码
   expression_values → ValueEncoder → value_embeddings (batch, seq_len, d_model)
    ↓
3. 嵌入融合
   - continuous/category模式: total_emb = gene_emb + value_emb
   - scaling模式: total_emb = gene_emb * value_emb
    ↓
4. 批归一化(可选)
   total_emb → DSBN/BatchNorm → normalized_emb
    ↓
5. Transformer编码
   normalized_emb → TransformerEncoder → transformer_output (batch, seq_len, d_model)
    ↓
6. 多任务输出
   ├── MLM任务: transformer_output → ExprDecoder → 表达值预测
   ├── CLS任务: cell_emb → ClsDecoder → 细胞类型分类
   ├── MVC任务: cell_emb + gene_emb → MVCDecoder → 掩码值预测
   ├── CCE任务: 对比学习损失
   └── ECS任务: 弹性细胞相似性损失
```

#### 1.4 核心模块详解

**GeneEncoder** (model.py:723-739)
- 使用Embedding层将基因ID映射到向量空间
- LayerNorm归一化
- 捕捉基因的语义信息

**ContinuousValueEncoder** (model.py:765-792)
- 两层MLP编码连续表达值
- 结构: Linear(1, d_model) → ReLU → Linear(d_model, d_model) → LayerNorm
- 限制最大值为512(max_value)

**ExprDecoder** (model.py:848-888)
- 3层全连接网络
- 结构: Linear → LeakyReLU → Linear → LeakyReLU → Linear(→1)
- 可选显式零概率预测(explicit_zero_prob)

**ClsDecoder** (model.py:890-918)
- 多层分类头
- 每层包含: Linear → Activation → LayerNorm
- 最后一层输出分类logits

**MVCDecoder** (model.py:921-1011)
- 掩码值预测解码器
- 三种架构风格:
  - "inner product": 基于内积的预测
  - "concat query": 拼接查询向量
  - "sum query": 求和查询向量

### 2. TransformerGenerator (generation_model.py)

用于扰动预测和生成任务的模型。

#### 2.1 特点

- 支持扰动标志(perturbation flags)编码
- AffineExprDecoder: 仿射形式的解码器 `Ax + b`
- 用于预测基因扰动后的表达变化

#### 2.2 关键组件

**PerturbationEncoder** (generation_model.py:83)
- 3维嵌入: (无扰动, 上调, 下调)
- padding_idx=2

**AffineExprDecoder** (generation_model.py:376-441)
- 两个ExprDecoder分别预测系数A和偏置b
- 输出: `pred = A * values + b`
- 可选adaptive_bias: 根据非零均值调整偏置

**pred_perturb方法** (generation_model.py:295-349)
- 批量扰动预测
- 支持两种模式:
  - "all": 包含所有基因
  - "batch-wise": 只包含非零基因

### 3. MultiOmicTransformerModel (multiomic_model.py)

扩展的多组学数据模型。

#### 3.1 新增功能

- **ModEncoder**: 模态类型编码器
- 支持多种组学数据类型(RNA, ATAC, protein等)
- 模态嵌入与批次嵌入融合

#### 3.2 前向传播

```python
输入: (gene_ids, values, padding_mask, batch_labels, mod_types)
    ↓
编码阶段: 同TransformerModel
    ↓
模态嵌入融合:
if use_batch_labels and use_mod:
    cat_emb = batch_emb + mod_emb
elif use_batch_labels:
    cat_emb = batch_emb
elif use_mod:
    cat_emb = mod_emb
    ↓
解码: cat([transformer_output, cat_emb], dim=-1) → Decoder
```

## 辅助模块

### 4. Domain-Specific Batch Normalization (dsbn.py)

#### 4.1 DomainSpecificBatchNorm1d

- 为不同批次/域维护独立的BatchNorm统计量
- 减少批次效应
- 结构: `bns = [BN_1, BN_2, ..., BN_n]`

#### 4.2 使用场景

- 多批次数据整合
- 跨数据集训练
- 批次效应校正

### 5. Gradient Reversal Layer (grad_reverse.py)

#### 5.1 GradReverse

- 实现梯度反转
- 前向传播: 恒等映射
- 反向传播: 梯度取反并乘以lambda

#### 5.2 应用

- 对抗训练(Domain Adaptation)
- AdversarialDiscriminator中使用
- 实现批次不变的细胞嵌入

## Flash Attention优化

### FlashTransformerEncoderLayer

所有模型都支持Flash Attention v2:
- 减少内存占用
- 加速训练和推理
- 支持pre-norm和post-norm两种方案

**Pre-norm vs Post-norm**:
```python
# Pre-norm (norm_scheme="pre")
x = LayerNorm(x)
x = x + Attention(x)
x = LayerNorm(x)
x = x + FFN(x)

# Post-norm (norm_scheme="post")
x = x + Attention(x)
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)
```

## 训练目标

### 1. MLM (Masked Language Modeling)
- 预测掩码基因的表达值
- 损失: MSE或Huber loss

### 2. CLS (Cell Type Classification)
- 细胞类型分类
- 损失: CrossEntropy

### 3. MVC (Masked Value Prediction for Cell Embedding)
- 基于细胞嵌入预测掩码基因表达
- 损失: MSE

### 4. CCE (Contrastive Cell Embedding)
- 对比学习
- 同一细胞的两次增强应该相似
- 损失: InfoNCE

### 5. ECS (Elastic Cell Similarity)
- 弹性细胞相似性
- 控制细胞嵌入的分布
- 损失: `mean(1 - (cos_sim - threshold)^2)`

### 6. DAB (Domain Adversarial for Batch correction)
- 对抗性批次校正
- 使细胞嵌入对批次信息不敏感
- 损失: CrossEntropy (with gradient reversal)

## 模型输入输出

### 输入格式

```python
{
    "gene_ids": Tensor[batch, seq_len],        # 基因ID
    "values": Tensor[batch, seq_len],          # 表达值
    "src_key_padding_mask": Tensor[batch, seq_len],  # padding掩码
    "batch_labels": Tensor[batch],             # 批次标签(可选)
}
```

### 输出格式

```python
{
    "mlm_output": Tensor[batch, seq_len],      # 表达值预测
    "cell_emb": Tensor[batch, d_model],        # 细胞嵌入
    "cls_output": Tensor[batch, n_cls],        # 分类输出(可选)
    "mvc_output": Tensor[batch, seq_len],      # MVC预测(可选)
    "mlm_zero_probs": Tensor[batch, seq_len],  # 零概率(可选)
    "loss_cce": Tensor[],                      # CCE损失(可选)
    "loss_ecs": Tensor[],                      # ECS损失(可选)
    "dab_output": Tensor[batch, n_batches],    # DAB输出(可选)
}
```

## 关键文件路径参考

- 核心模型: `scgpt/model/model.py:28-440`
- 基因编码器: `scgpt/model/model.py:723-739`
- Flash Attention层: `scgpt/model/model.py:595-720`
- MVC解码器: `scgpt/model/model.py:921-1011`
- 对抗判别器: `scgpt/model/model.py:1013-1045`
- 生成模型: `scgpt/model/generation_model.py:25-293`
- 多组学模型: `scgpt/model/multiomic_model.py:25-533`

## 设计理念

1. **模块化**: 各组件独立,易于组合和扩展
2. **多任务**: 支持多种预训练目标,提升模型泛化能力
3. **灵活性**: 支持多种编码方式、细胞嵌入方式
4. **可扩展**: 易于添加新的编码器、解码器、损失函数
5. **优化**: Flash Attention、混合精度训练支持

## 总结

scGPT框架提供了一套完整的单细胞分析工具:
- **TransformerModel**: 通用基础模型,支持多种预训练目标
- **TransformerGenerator**: 扰动预测模型
- **MultiOmicTransformerModel**: 多组学数据整合模型

所有模型都基于Transformer架构,支持预训练和微调范式,可应用于细胞类型注释、批次校正、扰动响应预测等多种任务。
