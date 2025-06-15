import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

class GeneEmbedding(nn.Module):
    """
    Gene Embedding分支：专注于捕捉基因的语义表示
    
    学习基因的语义表示，包括：
    - 功能相似性：功能相关的基因在嵌入空间中距离较近
    - 通路关系：同一生物学通路的基因具有相似的表示
    - 调控关系：转录因子与其靶基因之间的关系
    """
    def __init__(self, embedding_dim: int, num_genes: int):
        super(GeneEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.gene_embedding = nn.Embedding(num_embeddings=num_genes, embedding_dim=embedding_dim)
        
    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_ids: 基因ID序列 G = [g_1, g_2, ..., g_g], shape: (batch_size, g)
            
        Returns:
            gene_embeddings: 基因嵌入 E_gene ∈ R^{g×d}, shape: (batch_size, g, d)
        """
        return self.gene_embedding(gene_ids)
    

class ExpressionEmbedding(nn.Module):
    """
    Expression Embedding分支：专注于捕捉表达量的数值特征和上下文依赖
    
    考虑到scRNA-seq数据的特点，设计分层编码策略：
    1. 表达量归一化与离散化
    2. 分层表达嵌入
    """
    def __init__(self, embedding_dim: int, num_bins: int = 50, alpha: float = 0.1):
        """
        Args:
            embedding_dim: 嵌入维度 d
            num_bins: 离散化的bin数量 N
            alpha: 平衡离散和连续特征的权重参数
        """
        super(ExpressionEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.alpha = alpha
        
        # 离散表达水平的嵌入矩阵 W_bin ∈ R^{d×N}
        self.bin_embedding = nn.Embedding(
            num_embeddings=num_bins,
            embedding_dim=embedding_dim
        )
        
        # 连续值的投影向量 v_cont ∈ R^d
        self.continuous_projection = nn.Parameter(
            torch.randn(embedding_dim)
        )
        
        # 初始化权重
        nn.init.xavier_uniform_(self.bin_embedding.weight)
        nn.init.xavier_uniform_(self.continuous_projection.unsqueeze(0))
        
    def normalize_expression(self, expression: torch.Tensor) -> torch.Tensor:
        """
        表达量归一化：x̃_j = log(x_j + 1)
        
        Args:
            expression: 原始表达量 x = [x_1, x_2, ..., x_g], shape: (batch_size, g)
            
        Returns:
            normalized_expr: 归一化表达量 x̃, shape: (batch_size, g)
        """
        return torch.log(expression + 1)
    
    def discretize_expression(self, normalized_expr: torch.Tensor) -> torch.Tensor:
        """
        表达量离散化：b_j = Discretize_N(x̃_j)
        
        Args:
            normalized_expr: 归一化表达量 x̃, shape: (batch_size, g)
            
        Returns:
            bin_indices: 离散化的bin索引 b, shape: (batch_size, g)
        """
        # 找到表达量的范围
        min_val = normalized_expr.min()
        max_val = normalized_expr.max()
        print(min_val, max_val)
        # 将表达量映射到 [0, num_bins-1] 的范围
        normalized_range = (normalized_expr - min_val) / (max_val - min_val + 1e-8)
        bin_indices = torch.floor(normalized_range * (self.num_bins - 1)).long()
        
        # 确保bin索引在有效范围内
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices
    
    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            expression: 表达量向量 x = [x_1, x_2, ..., x_g], shape: (batch_size, g)
            
        Returns:
            expr_embeddings: 表达量嵌入 E_expr ∈ R^{g×d}, shape: (batch_size, g, d)
        """
        # Step 1: 表达量归一化与离散化
        normalized_expr = self.normalize_expression(expression)  # x̃_j = log(x_j + 1)
        bin_indices = self.discretize_expression(normalized_expr)  # b_j = Discretize_N(x̃_j)
        
        # Step 2: 分层表达嵌入
        # 离散部分：W_bin · OneHot_N(b_j)
        discrete_embeddings = self.bin_embedding(bin_indices)
        
        # 连续部分：α · x̃_j · v_cont
        # 扩展 continuous_projection 到 (batch_size, g, d)
        continuous_component = (
            self.alpha * 
            normalized_expr.unsqueeze(-1) * 
            self.continuous_projection.unsqueeze(0).unsqueeze(0)
        )
        # 组合离散和连续特征
        # e^expr_j = W_bin · OneHot_N(b_j) + α · x̃_j · v_cont
        expr_embeddings = discrete_embeddings + continuous_component
        
        return expr_embeddings


class CategoricalGumbelSoftmax(nn.Module):
    """
    Gumbel Softmax 函数：用于将离散分布转换为连续分布
    学习基因调控网络当中的关系，包括：抑制、激活、未知

    Args:
        num_categories: 类别数 default: 3
        hard: 是否使用hard模式，即是否使用one-hot编码
        
    """

    def __init__(self, embedding_dim: int, num_categories: int = 3, hard: bool = True):
        super(CategoricalGumbelSoftmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.hard = hard

        # Gumbel Softmax Reparametrization Learnable Parameters
        self.gumbel_softmax_reparametrization = nn.Linear(2*embedding_dim, 3)



    def forward(self, gene_ids_source: torch.Tensor, gene_ids_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_ids_source: 源基因ID序列 G_source = [g_1, g_2, ..., g_g], shape: (batch_size, g)
            gene_ids_target: 目标基因ID序列 G_target = [g_1, g_2, ..., g_g], shape: (batch_size, g)
        """
        gene_embeddings_source = self.gene_embedding(gene_ids_source)
        gene_embeddings_target = self.gene_embedding(gene_ids_target)
        # [batch_size, g, embedding_dim]

        # 将gene_embeddings_source和gene_embeddings_target拼接起来 [batch_size, g, 2*embedding_dim]
        gene_embeddings = torch.cat([gene_embeddings_source, gene_embeddings_target], dim=-1)

        # 将gene_embeddings通过gumbel_softmax_reparametrization进行重参数化 [batch_size, 3]
        gumbel_softmax_reparametrization_params = self.gumbel_softmax_reparametrization(gene_embeddings)



    def reparametrize(self, params: torch.Tensor) -> torch.Tensor:
        """
        """
        if self.training:
            u = torch.rand_like(logits)
            return torch.log(-torch.log(u + 1e-8)) + logits
        else:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard)

