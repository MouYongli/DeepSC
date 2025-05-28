# 数学建模

## 0. 问题定义

单细胞RNA测序数据（scRNA-seq）是现代生物学研究中的一项重要技术，它允许我们同时测量数千个细胞中数千个基因的表达水平。scRNA-seq数据通常以矩阵的形式表示，其中行代表细胞，列代表基因。scRNA-seq数据中，每个细胞的基因表达量是离散的值，通常是0或1，或者是一个连续的值。

## 1. 数据表示

单细胞RNA测序数据可以表示为一个矩阵 $\mathbf{X} \in \mathbb{R}^{n \times g}$，其中：
- $n$ 是细胞数量
- $g$ 是基因数量  
- $x_{i,j}$ 表示第 $i$ 个细胞中第 $j$ 个基因的表达量

对于每个细胞 $i$，我们有基因表达向量 $\mathbf{x}_i = [x_{i,1}, x_{i,2}, ..., x_{i,g}]^T$。

## 2. 模型设计

### 2.1 Embedding

Embedding的输入是离散的符号，输出是连续的向量。Embedding的输出向量通常用于神经网络的输入层，也可以用于神经网络的隐藏层。

#### 2.1.1 Gene Embedding分支

Gene Embedding分支专注于捕捉基因的内在生物学特性和基因间的调控关系：

$$\mathbf{E}_{gene} = f_{gene}(\mathbf{G}) \in \mathbb{R}^{g \times d}$$

其中：
- $\mathbf{G} = [g_1, g_2, ..., g_g]$ 是基因ID序列
- $d$ 是嵌入维度
- 每个基因 $g_j$ 对应一个可学习的向量 $\mathbf{e}^{gene}_{j} \in \mathbb{R}^d$

这个分支学习基因的语义表示，包括：
- **功能相似性**：功能相关的基因在嵌入空间中距离较近
- **通路关系**：同一生物学通路的基因具有相似的表示
- **调控关系**：转录因子与其靶基因之间的关系

#### 2.1.2 Expression Embedding分支

Expression Embedding分支专注于捕捉表达量的数值特征和上下文依赖：

$$\mathbf{E}_{expr} = f_{expr}(\mathbf{x}) \in \mathbb{R}^{g \times d}$$

其中：
- $f_{expr}$ 是表达量编码函数
- $\mathbf{x} = [x_1, x_2, ..., x_g]$ 是表达量向量
- $\mathbf{e}^{expr}_{j} \in \mathbb{R}^d$ 是第 $j$ 个基因的表达量嵌入向量

考虑到scRNA-seq数据的特点，我们设计分层编码策略：

**Step 1: 表达量归一化与离散化**
$$\tilde{x}_{j} = \text{log}(x_{j} + 1)$$
$$b_{j} = \text{Discretize}_{N}(\tilde{x}_{j})$$

**Step 2: 分层表达嵌入**
$$\mathbf{e}^{expr}_{j} = \mathbf{W}_{bin} \cdot \text{OneHot}_{N}(b_{j}) + \alpha \cdot \tilde{x}_{j} \cdot \mathbf{v}_{cont}$$

其中：
- $\mathbf{W}_{bin} \in \mathbb{R}^{d \times \text{}}$ 是离散表达水平的嵌入矩阵
- $\mathbf{v}_{cont} \in \mathbb{R}^d$ 是连续值的投影向量
- $\alpha$ 是平衡离散和连续特征的权重参数
- $N$ 是离散化的bin数量

### 2.2 双分支Transformer架构

传统的多头自注意力机制如下：

$$\text{MultiHead-Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)$$

其中：
- $\text{head}_i = \text{Attention}_{i}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ 是第 $i$ 个头
- $\text{Attention}_{i}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$ 是注意力机制
- $h$ 是头数
- $d$ 是维度
- $\mathbf{Q} = \mathbf{H} \cdot \mathbf{W}^{Q}_{i}$ 是查询向量的矩阵
- $\mathbf{K} = \mathbf{H} \cdot \mathbf{W}^{K}_{i}$ 是键向量的矩阵
- $\mathbf{V} = \mathbf{H} \cdot \mathbf{W}^{V}_{i}$ 是值向量的矩阵 

隐藏状态的更新过程如下：

$$\mathbf{H}^{(l+1)} = \text{MultiHead-Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) + \mathbf{H}^{(l)}$$


#### 2.2.1 初始化基因和表达的编码

结合生物学背景，我们考虑基因编码信息和表达量编码信息的双分支的注意力机制，并引入生物学约束的注意力权重。其中，基因编码分支的注意力机制只考虑基因编码信息，表达量编码分支的注意力机制表达量编码信息以外还考虑基因调控信息（从基因编码分支的注意力机制中提取）。

首先，我们初始化基因编码分支和表达量编码分支的隐藏状态：

$$\mathbf{H}_{gene}^{(0)} = \mathbf{E}_{gene}$$ 

$$\mathbf{H}_{expr}^{(0)} = \mathbf{E}_{expr}$$

然后，新基因编码分支和表达量编码分支的隐藏状态，多头注意力机制的计算过程如下：

#### 2.2.2 基因编码分支的注意力机制

$$\mathbf{H}_{gene}^{(l+1)} = \text{Attention}_{gene}(\mathbf{Q}_{gene}, \mathbf{K}_{gene}, \mathbf{V}_{gene}) + \mathbf{H}_{gene}^{(l)}$$

其中：
- $\mathbf{Q}_{gene} = \mathbf{H}_{gene}^{(l)} \cdot \mathbf{W}^{Q}_{gene}$ 是基因编码分支的查询向量的矩阵
- $\mathbf{K}_{gene} = \mathbf{H}_{gene}^{(l)} \cdot \mathbf{W}^{K}_{gene}$ 是基因编码分支的键向量的矩阵
- $\mathbf{V}_{gene} = \mathbf{H}_{gene}^{(l)} \cdot \mathbf{W}^{V}_{gene}$ 是基因编码分支的值向量的矩阵

- $\mathbf{W}^{Q}_{gene} \in \mathbb{R}^{d \times d}$ 是基因编码分支的查询矩阵
- $\mathbf{W}^{K}_{gene} \in \mathbb{R}^{d \times d}$ 是基因编码分支的键矩阵
- $\mathbf{W}^{V}_{gene} \in \mathbb{R}^{d \times d}$ 是基因编码分支的值矩阵


#### 2.2.3 表达量编码分支的注意力机制