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


# 需修改，有问题： 归一化和分箱应该放到data_collator里面，这里只做embedding
# 其次：在data_collator里面还要做好trunctuation和padding以及mask.
# 这里没有mask直接做normalization肯定会有问题
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
    基因编码分支的多头注意力机制，支持门控稀疏化和带符号归一化。
    """

    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        Q: torch.Tensor,  # (batch, g, num_heads, head_dim)
        K: torch.Tensor,  # (batch, g, num_heads, head_dim)
        V: torch.Tensor,  # (batch, g, num_heads, head_dim)
        M: torch.Tensor,  # (batch, num_heads, g, g) or (batch, g, g)
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # 自动扩展M到多头
        if M.dim() == 3:
            M = M.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # (batch, num_heads, g, g)
        # 转换为 (batch, num_heads, g, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        # Q, K, V: (batch, num_heads, g, head_dim)
        # 计算注意力分数
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (batch, num_heads, g, g)
        A = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, g, g)
        A = self.dropout(A)  # 注意力dropout
        # 稀疏化
        A_sparse = A * M  # (batch, num_heads, g, g)
        # 带符号归一化
        norm = (
            torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
        )  # (batch, num_heads, g, 1)
        A_bar = A_sparse / norm  # (batch, num_heads, g, g)
        # 注意力输出
        output = torch.matmul(A_bar, V)  # (batch, num_heads, g, head_dim)
        # 转回 (batch, g, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)  # (batch, g, d)
        output = self.out_proj(output)
        return output  # (batch, g, d)


class ExpressionAttentionLayer(nn.Module):
    """
    表达值编码分支的多头注意力机制，融合基因和表达的K/Q，支持门控稀疏化。
    """

    def __init__(self, d, num_heads, fused: bool = True, attn_dropout=0.1):
        super().__init__()
        self.fused = fused
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.dropout = nn.Dropout(attn_dropout)
        if fused:
            self.W_K_fused = nn.Linear(2 * self.head_dim, self.head_dim)
            self.W_Q_fused = nn.Linear(2 * self.head_dim, self.head_dim)
        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        Q_gene: torch.Tensor,  # (batch, g, num_heads, head_dim)
        K_gene: torch.Tensor,  # (batch, g, num_heads, head_dim)
        Q_expr: torch.Tensor,  # (batch, g, num_heads, head_dim)
        K_expr: torch.Tensor,  # (batch, g, num_heads, head_dim)
        V_expr: torch.Tensor,  # (batch, g, num_heads, head_dim)
        M: torch.Tensor,  # (batch, num_heads, g, g) or (batch, g, g)
    ) -> torch.Tensor:
        # 自动扩展M到多头
        if M.dim() == 3:
            M = M.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # (batch, num_heads, g, g)
        # 转换为 (batch, num_heads, g, head_dim)
        Q_gene = Q_gene.permute(0, 2, 1, 3)
        K_gene = K_gene.permute(0, 2, 1, 3)
        Q_expr = Q_expr.permute(0, 2, 1, 3)
        K_expr = K_expr.permute(0, 2, 1, 3)
        V_expr = V_expr.permute(0, 2, 1, 3)
        # 融合K/Q
        if self.fused:
            K_fused = self.W_K_fused(
                torch.cat([K_gene, K_expr], dim=-1)
            )  # (batch, num_heads, g, head_dim)
            Q_fused = self.W_Q_fused(
                torch.cat([Q_gene, Q_expr], dim=-1)
            )  # (batch, num_heads, g, head_dim)
        else:
            K_fused = K_expr
            Q_fused = Q_expr
        # 注意力
        attn_logits = torch.matmul(Q_fused, K_fused.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (batch, num_heads, g, g)
        A = F.softmax(attn_logits, dim=-1)  # (batch, num_heads, g, g)
        A = self.dropout(A)  # 注意力dropout
        # 稀疏化
        A_sparse = A * M  # (batch, num_heads, g, g)
        # 注意力输出
        output = torch.matmul(A_sparse, V_expr)  # (batch, num_heads, g, head_dim)
        # 转回 (batch, g, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)  # (batch, g, d)
        output = self.out_proj(output)
        return output  # (batch, g, d)


class GeneBranchQKV(nn.Module):
    """
    基因编码分支的 Q/K/V 计算模块，支持多头
    输入: 基因嵌入 (batch, g, embedding_dim)
    输出: Q_gene, K_gene, V_gene (batch, g, num_heads, head_dim)
    """

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, gene_emb: torch.Tensor) -> tuple:
        # gene_emb: (batch, g, embedding_dim)
        batch, g, _ = gene_emb.shape
        Q = self.W_Q(gene_emb).view(batch, g, self.num_heads, self.head_dim)
        K = self.W_K(gene_emb).view(batch, g, self.num_heads, self.head_dim)
        V = self.W_V(gene_emb).view(batch, g, self.num_heads, self.head_dim)
        return Q, K, V


class ExpressionBranchQKV(nn.Module):
    """
    表达值编码分支的 Q/K/V 计算模块，支持多头
    输入: 表达嵌入 (batch, g, embedding_dim)
    输出: Q_expr, K_expr, V_expr (batch, g, num_heads, head_dim)
    """

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, expr_emb: torch.Tensor) -> tuple:
        # expr_emb: (batch, g, embedding_dim)
        batch, g, _ = expr_emb.shape
        Q = self.W_Q(expr_emb).view(batch, g, self.num_heads, self.head_dim)
        K = self.W_K(expr_emb).view(batch, g, self.num_heads, self.head_dim)
        V = self.W_V(expr_emb).view(batch, g, self.num_heads, self.head_dim)
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
    def __init__(self, embedding_dim, num_heads, attn_dropout, ffn_dropout, fused):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.gene_qkv = GeneBranchQKV(embedding_dim, num_heads)
        self.expr_qkv = ExpressionBranchQKV(embedding_dim, num_heads)
        self.gene_attn = GeneAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.expr_attn = ExpressionAttentionLayer(
            embedding_dim, num_heads, fused, attn_dropout
        )
        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)
        # FFN
        self.ffn_gene = FeedForward(embedding_dim, dropout=ffn_dropout)
        self.ffn_expr = FeedForward(embedding_dim, dropout=ffn_dropout)

    def forward(self, gene_emb, expr_emb, M):
        # 和金尔确认这里是否可以这样改写？把layer norm放到前面? 我没见过
        #         # QKV: (batch, g, num_heads, head_dim)
        # Q_gene, K_gene, V_gene = self.gene_qkv(gene_emb)
        # attn_gene = self.gene_attn(Q_gene, K_gene, V_gene, M)  # (batch, g, d)
        # # 合并多头（已在注意力层完成，无需reshape）
        # h_gene = self.norm_gene1(gene_emb + attn_gene)
        # ffn_gene = self.ffn_gene(h_gene)
        # out_gene = self.norm_gene2(h_gene + ffn_gene)
        # Pre-LN for gene branch
        h_gene = gene_emb + self.gene_attn(
            self.gene_qkv(self.norm_gene1(gene_emb))[0],
            self.gene_qkv(self.norm_gene1(gene_emb))[1],
            self.gene_qkv(self.norm_gene1(gene_emb))[2],
            M,
        )
        out_gene = h_gene + self.ffn_gene(self.norm_gene2(h_gene))
        # Expression branch: 融合 gene 的 Q/K 参与注意力
        # Q_expr, K_expr, V_expr = self.expr_qkv(expr_emb)
        # attn_expr = self.expr_attn(Q_gene, K_gene, Q_expr, K_expr, V_expr, M)  # (batch, g, d)
        # # 合并多头（已在注意力层完成，无需reshape）
        # h_expr = self.norm_expr1(expr_emb + attn_expr)
        # ffn_expr = self.ffn_expr(h_expr)
        # out_expr = self.norm_expr2(h_expr + ffn_expr)
        # Pre-LN for expression branch
        h_expr = expr_emb + self.expr_attn(
            self.gene_qkv(self.norm_expr1(gene_emb))[0],
            self.gene_qkv(self.norm_expr1(gene_emb))[1],
            self.expr_qkv(self.norm_expr1(expr_emb))[0],
            self.expr_qkv(self.norm_expr1(expr_emb))[1],
            self.expr_qkv(self.norm_expr1(expr_emb))[2],
            M,
        )
        out_expr = h_expr + self.ffn_expr(self.norm_expr2(h_expr))

        return out_gene, out_expr


class DeepSC(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_genes,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        fused=True,
        num_bins=50,
        alpha=0.1,
    ):
        super().__init__()
        self.gene_embedding = GeneEmbedding(embedding_dim, num_genes)
        self.expr_embedding = ExpressionEmbedding(
            embedding_dim, num_bins=num_bins, alpha=alpha
        )
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                DeepSCTransformerBlock(
                    embedding_dim, num_heads, attn_dropout, ffn_dropout, fused
                )
                for _ in range(num_layers)
            ]
        )
        self.gumbel_softmax = CategoricalGumbelSoftmax(embedding_dim)  # 默认参数

    def forward(self, gene_ids, expression):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression: (batch, g)  # 表达量
        M: (batch, num_heads, g, g) 门控矩阵
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression)  # (batch, g, d)
        # 问题：是否需要M是每层独立的？
        M = self.gumbel_softmax(gene_emb)  # (batch, g, g)
        for layer in self.layers:
            gene_emb, expr_emb = layer(gene_emb, expr_emb, M)
        return gene_emb, expr_emb
