import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gene_embedding = nn.Embedding(
            num_embeddings=num_genes, embedding_dim=embedding_dim
        )

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
            num_embeddings=num_bins, embedding_dim=embedding_dim
        )

        # 连续值的投影向量 v_cont ∈ R^d
        self.continuous_projection = nn.Parameter(torch.randn(embedding_dim))

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
        bin_indices = self.discretize_expression(
            normalized_expr
        )  # b_j = Discretize_N(x̃_j)

        # Step 2: 分层表达嵌入
        # 离散部分：W_bin · OneHot_N(b_j)
        discrete_embeddings = self.bin_embedding(bin_indices)

        # 连续部分：α · x̃_j · v_cont
        # 扩展 continuous_projection 到 (batch_size, g, d)
        continuous_component = (
            self.alpha
            * normalized_expr.unsqueeze(-1)
            * self.continuous_projection.unsqueeze(0).unsqueeze(0)
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
        embedding_dim: 基因嵌入维度
        num_categories: 类别数 default: 3
        hard: 是否使用hard模式，即是否使用one-hot编码
        temperature: Gumbel-Softmax温度参数
    """

    def __init__(
        self,
        embedding_dim: int,
        num_categories: int = 3,
        hard: bool = True,
        temperature: float = 1.0,
    ):
        super(CategoricalGumbelSoftmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.hard = hard
        self.temperature = temperature
        # 只保留MLP
        self.gumbel_softmax_reparametrization = nn.Linear(
            2 * embedding_dim, num_categories
        )

    def forward(self, gene_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_emb: 基因嵌入, shape: (batch, g, d)
        Returns:
            M: 门控矩阵，shape: (batch, g, g)
        """
        # 构造所有基因对 (i, j)
        src_exp = gene_emb.unsqueeze(2)  # (batch, g, 1, d)
        tgt_exp = gene_emb.unsqueeze(1)  # (batch, 1, g, d)
        pair_emb = torch.cat(
            [
                src_exp.expand(-1, -1, gene_emb.size(1), -1),
                tgt_exp.expand(-1, gene_emb.size(1), -1, -1),
            ],
            dim=-1,
        )  # (batch, g, g, 2d)
        logits = self.gumbel_softmax_reparametrization(pair_emb)  # (batch, g, g, 3)
        y = F.gumbel_softmax(
            logits, tau=self.temperature, hard=self.hard, dim=-1
        )  # (batch, g, g, 3)
        M = y[..., 0] * (-1) + y[..., 1] * 0 + y[..., 2] * (+1)  # (batch, g, g)
        return M


class GeneAttentionLayer(nn.Module):
    """
    基因编码分支的注意力机制，支持门控稀疏化和带符号归一化。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        M: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Args:
            Q: (batch, g, d)
            K: (batch, g, d)
            V: (batch, g, d)
            M: (batch, g, g) 门控矩阵
            eps: 防止除零
        Returns:
             output: (batch, g, d) 稀疏归一化注意力输出
        """
        d = Q.size(-1)
        # 计算原始注意力权重
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (d**0.5)  # (batch, g, g)
        A = F.softmax(attn_logits, dim=-1)  # (batch, g, g)
        # 稀疏化：逐元素乘以门控矩阵
        A_sparse = A * M  # (batch, g, g)
        # 带符号归一化
        norm = (
            torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
        )  # (batch, g, 1)
        A_bar = A_sparse / norm  # (batch, g, g)
        # 注意力输出
        output = torch.matmul(A_bar, V)  # (batch, g, d)
        return output


class ExpressionAttentionLayer(nn.Module):
    """
    表达值编码分支的注意力机制，融合基因和表达的K/Q，支持门控稀疏化。
    """

    def __init__(self, d, fused: bool = True):
        super().__init__()
        self.fused = fused
        if fused:
            self.W_K_fused = nn.Linear(2 * d, d)
            self.W_Q_fused = nn.Linear(2 * d, d)

    def forward(
        self,
        Q_gene: torch.Tensor,
        K_gene: torch.Tensor,
        Q_expr: torch.Tensor,
        K_expr: torch.Tensor,
        V_expr: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Q_gene, K_gene: (batch, g, d)
            Q_expr, K_expr, V_expr: (batch, g, d)
            M: (batch, g, g)
        Returns:
            output: (batch, g, d)
        """
        d = Q_gene.size(-1)
        if self.fused:
            # 融合K/Q
            K_fused = self.W_K_fused(
                torch.cat([K_gene, K_expr], dim=-1)
            )  # (batch, g, d)
            Q_fused = self.W_Q_fused(
                torch.cat([Q_gene, Q_expr], dim=-1)
            )  # (batch, g, d)
        else:
            K_fused = K_expr
            Q_fused = Q_expr
        # 注意力
        attn_logits = torch.matmul(Q_fused, K_fused.transpose(-2, -1)) / (
            d**0.5
        )  # (batch, g, g)
        A = F.softmax(attn_logits, dim=-1)  # (batch, g, g)
        # 稀疏化
        A_sparse = A * M  # (batch, g, g)
        # 注意力输出
        output = torch.matmul(A_sparse, V_expr)  # (batch, g, d)
        return output


class GeneBranchQKV(nn.Module):
    """
    基因编码分支的 Q/K/V 计算模块
    输入: 基因嵌入 (batch, g, embedding_dim)
    输出: Q_gene, K_gene, V_gene (batch, g, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, gene_emb: torch.Tensor) -> tuple:
        Q = self.W_Q(gene_emb)
        K = self.W_K(gene_emb)
        V = self.W_V(gene_emb)
        return Q, K, V


class ExpressionBranchQKV(nn.Module):
    """
    表达值编码分支的 Q/K/V 计算模块
    输入: 表达嵌入 (batch, g, embedding_dim)
    输出: Q_expr, K_expr, V_expr (batch, g, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, expr_emb: torch.Tensor) -> tuple:
        Q = self.W_Q(expr_emb)
        K = self.W_K(expr_emb)
        V = self.W_V(expr_emb)
        return Q, K, V


# 用法示例：
# gene_qkv = GeneBranchQKV(d)
# Q_gene, K_gene, V_gene = gene_qkv(gene_emb)
# expr_qkv = ExpressionBranchQKV(d)
# Q_expr, K_expr, V_expr = expr_qkv(expr_emb)
#
# gene_attn = GeneAttentionLayer()
# O_gene = gene_attn(Q_gene, K_gene, V_gene, M)
# expr_attn = ExpressionAttentionLayer(d)
# O_expr = expr_attn(Q_gene, K_gene, Q_expr, K_expr, V_expr, M)


class FeedForward(nn.Module):
    def __init__(self, d, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d * 4
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DeepSCTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, fused=True):
        super().__init__()
        self.gene_qkv = GeneBranchQKV(embedding_dim)
        self.expr_qkv = ExpressionBranchQKV(embedding_dim)
        self.gene_attn = GeneAttentionLayer()
        self.expr_attn = ExpressionAttentionLayer(embedding_dim, fused=fused)
        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)
        # FFN
        self.ffn_gene = FeedForward(embedding_dim, dropout=dropout)
        self.ffn_expr = FeedForward(embedding_dim, dropout=dropout)

    def forward(self, gene_emb, expr_emb, M):
        Q_gene, K_gene, V_gene = self.gene_qkv(gene_emb)
        attn_gene = self.gene_attn(Q_gene, K_gene, V_gene, M)
        h_gene = self.norm_gene1(attn_gene)
        ffn_gene = self.ffn_gene(h_gene)
        out_gene = self.norm_gene2(h_gene + ffn_gene)

        # Expression branch: 融合 gene 的 Q/K 参与注意力
        Q_expr, K_expr, V_expr = self.expr_qkv(expr_emb)
        attn_expr = self.expr_attn(Q_gene, K_gene, Q_expr, K_expr, V_expr, M)
        h_expr = self.norm_expr1(expr_emb + attn_expr)
        ffn_expr = self.ffn_expr(h_expr)
        out_expr = self.norm_expr2(h_expr + ffn_expr)

        return out_gene, out_expr


class DeepSCTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_genes,
        num_layers=4,
        dropout=0.1,
        fused=True,
        num_bins=50,
        alpha=0.1,
    ):
        super().__init__()
        self.gene_embedding = GeneEmbedding(embedding_dim, num_genes)
        self.expr_embedding = ExpressionEmbedding(
            embedding_dim, num_bins=num_bins, alpha=alpha
        )
        self.layers = nn.ModuleList(
            [
                DeepSCTransformerBlock(embedding_dim, dropout, fused)
                for _ in range(num_layers)
            ]
        )

    def forward(self, gene_ids, expression, M):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression: (batch, g)  # 表达量
        M: (batch, g, g)  # 门控矩阵
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression)  # (batch, g, d)
        for layer in self.layers:
            gene_emb, expr_emb = layer(gene_emb, expr_emb, M)
        return gene_emb, expr_emb
