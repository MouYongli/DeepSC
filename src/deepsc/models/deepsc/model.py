import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO：embedding 的行数不只是num_genes，而是num_genes+1，因为还有<cls> token
class GeneEmbedding(nn.Module):
    """
    Gene Embedding分支：专注于捕捉基因的语义表示

    学习基因的语义表示，包括：
    - 功能相似性：功能相关的基因在嵌入空间中距离较近
    - 通路关系：同一生物学通路的基因具有相似的表示
    - 调控关系：转录因子与其靶基因之间的关系
    """

    # num_genes 是基因数量，包括<cls>和<pad>
    def __init__(self, embedding_dim: int, num_genes: int):
        super(GeneEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.gene_embedding = nn.Embedding(
            num_embeddings=num_genes + 2, embedding_dim=embedding_dim, padding_idx=0
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
class ExpressionEmbedding(nn.Module):
    """
    Expression Embedding分支：专注于捕捉表达量的数值特征和上下文依赖

    考虑到scRNA-seq数据的特点，设计分层编码策略：
    1. 表达量归一化与离散化
    2. 分层表达嵌入
    """

    # num_bins 是bin数量，包括<cls>和<pad>以及<mask>
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
        # alpba是否可以作为可学习的参数
        self.alpha = alpha
        # 离散表达水平的嵌入矩阵 W_bin ∈ R^{d×N}
        self.bin_embedding = nn.Embedding(
            num_embeddings=num_bins + 3, embedding_dim=embedding_dim, padding_idx=0
        )
        # 连续值的投影向量 v_cont ∈ R^d
        self.continuous_projection = nn.Parameter(torch.randn(embedding_dim))

        # 初始化权重
        nn.init.xavier_uniform_(self.bin_embedding.weight)
        nn.init.xavier_uniform_(self.continuous_projection.unsqueeze(0))

    def forward(
        self, discrete_expression: torch.Tensor, normalized_expr: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            expression: 表达量向量 x = [x_1, x_2, ..., x_g], shape: (batch_size, g)

        Returns:
            expr_embeddings: 表达量嵌入 E_expr ∈ R^{g×d}, shape: (batch_size, g, d)
        """

        discrete_embeddings = self.bin_embedding(discrete_expression)
        continuous_component = (
            self.alpha
            * normalized_expr.unsqueeze(-1)
            * self.continuous_projection.unsqueeze(0).unsqueeze(0)
        )

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
        temperature: float = 0.3,
    ):
        super(CategoricalGumbelSoftmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.hard = hard
        self.temperature = temperature
        # 只保留MLP
        self.gumbel_softmax_reparametrization = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_categories),
        )

    def forward(self, gene_emb: torch.Tensor):
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
        return M, y


# TODO： 这里对V的操作有些冗余，其实可以直接用multi_head_attention_forward这个函数，虽然复杂了些，但是能够减少一个V的计算。稍后再做
class GeneAttentionLayer(nn.Module):
    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(d, d)
        self.mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

    def forward(
        self,
        x,
        V,
        M=None,
        eps: float = 1e-8,
    ):
        if M is not None and M.dim() == 3:
            M = M.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # (batch, num_heads, g, g)
        V = V.permute(0, 2, 1, 3)
        output, A = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        if M is not None:
            A_sparse = A * M  # (batch, num_heads, g, g)
            norm = (
                torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
            )  # (batch, num_heads, g, 1)
            A_bar = A_sparse / norm  # (batch, num_heads, g, g)
        else:
            A_bar = A
        output = torch.matmul(A_bar, V)  # (batch, num_heads, g, head_dim)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)  # (batch, g, d)
        output = self.out_proj(output)
        return output


class ExpressionAttentionLayer(nn.Module):
    def __init__(self, d, num_heads, attn_dropout=0.1, fused: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(d, d)
        self.mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        # TODO: 这里的改动需要确认：之前是把映射好QK的给链接然后映射，现在略有不同
        if fused:
            self.fused_emb_proj = nn.Linear(2 * d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        gene_emb,
        expr_emb,
        V,
        M=None,
        eps: float = 1e-8,
    ):
        if M is not None and M.dim() == 3:
            M = M.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # (batch, num_heads, g, g)
        fused_emb = torch.cat([gene_emb, expr_emb], dim=-1)
        fused_emb = self.fused_emb_proj(fused_emb)
        V = V.permute(0, 2, 1, 3)
        output, A = self.mha(
            fused_emb,
            fused_emb,
            expr_emb,
            need_weights=True,
            average_attn_weights=False,
        )
        if M is not None:
            A_sparse = A * M  # (batch, num_heads, g, g)
            norm = (
                torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
            )  # (batch, num_heads, g, 1)
            A_bar = A_sparse / norm  # (batch, num_heads, g, g)
        else:
            A_bar = A
        output = torch.matmul(A_bar, V)  # (batch, num_heads, g, head_dim)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)  # (batch, g, d)
        output = self.out_proj(output)
        return output


class NewGeneAttentionLayer(nn.Module):
    """
    手动实现的多头注意力层，用于基因分支
    """

    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.d = d

        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)

        # 输出投影
        self.out_proj = nn.Linear(d, d)

        # Dropout
        self.dropout = nn.Dropout(attn_dropout)

        # 缩放因子
        self.scale = self.head_dim**-0.5

    def forward(self, x, M=None, eps: float = 1e-8):
        """
        Args:
            x: 输入张量, shape: (batch_size, seq_len, d)
            M: 门控矩阵, shape: (batch_size, seq_len, seq_len) 或 None
            eps: 数值稳定性参数
        Returns:
            output: 注意力输出, shape: (batch_size, seq_len, d)
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: (batch_size, num_heads, seq_len, seq_len)

        # 应用门控矩阵 M（如果提供）
        if M is not None:
            if M.dim() == 3:
                M = M.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # 将门控矩阵应用到注意力分数上
            scores = scores * M

        # 应用 softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 如果有门控矩阵，进行稀疏化处理
        if M is not None:
            A_sparse = attn_weights * M
            norm = torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
            A_bar = A_sparse / norm
        else:
            A_bar = attn_weights

        # 计算输出
        output = torch.matmul(A_bar, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 转置并重塑
        output = output.transpose(
            1, 2
        ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d)  # (batch_size, seq_len, d)

        # 输出投影
        output = self.out_proj(output)

        return output


class NewExpressionAttentionLayer(nn.Module):
    """
    手动实现的多头注意力层，用于表达分支
    使用融合的基因和表达嵌入作为Q和K，表达嵌入作为V
    """

    def __init__(self, d, num_heads, attn_dropout=0.1, fused: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.d = d
        self.fused = fused

        # 融合投影（如果启用）
        if fused:
            self.fused_emb_proj = nn.Linear(2 * d, d)

        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)

        # 输出投影
        self.out_proj = nn.Linear(d, d)

        # Dropout
        self.dropout = nn.Dropout(attn_dropout)

        # 缩放因子
        self.scale = self.head_dim**-0.5

    def forward(self, gene_emb, expr_emb, M=None, eps: float = 1e-8):
        """
        Args:
            gene_emb: 基因嵌入, shape: (batch_size, seq_len, d)
            expr_emb: 表达嵌入, shape: (batch_size, seq_len, d)
            M: 门控矩阵, shape: (batch_size, seq_len, seq_len) 或 None
            eps: 数值稳定性参数
        Returns:
            output: 注意力输出, shape: (batch_size, seq_len, d)
        """
        batch_size, seq_len, _ = gene_emb.shape

        # 融合基因和表达嵌入（如果启用）
        if self.fused:
            fused_emb = torch.cat([gene_emb, expr_emb], dim=-1)
            fused_emb = self.fused_emb_proj(fused_emb)
        else:
            fused_emb = gene_emb

        # 计算 Q, K, V
        Q = self.W_Q(fused_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_K(fused_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_V(expr_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: (batch_size, num_heads, seq_len, seq_len)

        # 应用门控矩阵 M（如果提供）
        if M is not None:
            if M.dim() == 3:
                M = M.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # 将门控矩阵应用到注意力分数上
            scores = scores * M

        # 应用 softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 如果有门控矩阵，进行稀疏化处理
        if M is not None:
            A_sparse = attn_weights * M
            norm = torch.sum(torch.abs(A_sparse), dim=-1, keepdim=True) + eps
            A_bar = A_sparse / norm
        else:
            A_bar = attn_weights

        # 计算输出
        output = torch.matmul(A_bar, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 转置并重塑
        output = output.transpose(
            1, 2
        ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d)  # (batch_size, seq_len, d)

        # 输出投影
        output = self.out_proj(output)

        return output


class BranchV(nn.Module):
    """
    通用的 V 计算模块，支持多头
    输入: (batch, g, embedding_dim)
    输出: (batch, g, num_heads, head_dim)
    """

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, g, _ = x.shape
        V = self.W_V(x).view(batch, g, self.num_heads, self.head_dim)
        return V


class FeedForward(nn.Module):
    def __init__(self, d, hidden_dim=None, dropout=0.1, num_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or d * 4
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 创建每一层的模块
        self.layers = nn.ModuleList()

        # 第一层：d -> hidden_dim
        first_layer = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.layers.append(first_layer)

        # 中间层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            middle_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.layers.append(middle_layer)

        # 最后一层：hidden_dim -> d
        last_layer = nn.Sequential(nn.Linear(hidden_dim, d), nn.Dropout(dropout))
        self.layers.append(last_layer)

    def forward(self, x):
        if self.num_layers == 2:
            # 只有两层的情况：d -> hidden_dim -> d
            h = self.layers[0](x)  # d -> hidden_dim
            h = self.layers[1](h)  # hidden_dim -> d
            return x + h  # 残差连接

        else:
            # 多层的情况
            # 第一层
            h = self.layers[0](x)  # d -> hidden_dim

            # 中间层，每层都有残差连接
            for i in range(1, self.num_layers - 1):
                residual = h
                h = self.layers[i](h)  # hidden_dim -> hidden_dim
                h = h + residual  # 残差连接

            # 最后一层
            h = self.layers[-1](h)  # hidden_dim -> d
            return x + h  # 最终残差连接


class DeepSCTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        ffn_dropout,
        fused,
        num_layers_ffn=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.gene_v = BranchV(embedding_dim, num_heads)
        self.expr_v = BranchV(embedding_dim, num_heads)
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
        self.ffn_gene = FeedForward(
            embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        )
        self.ffn_expr = FeedForward(
            embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        )
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, gene_emb, expr_emb, M=None):

        # Gene branch
        x = self.norm_gene1(gene_emb)
        V_gene = self.gene_v(x)
        attn_gene = self.gene_attn(x, V_gene, M)
        x = gene_emb + self.dropout(attn_gene)  # <-- 添加 residual dropout
        x_ln = self.norm_gene2(x)
        ffn_gene = self.ffn_gene(x_ln)
        out_gene = x + self.dropout(ffn_gene)  # <-- 再次 residual dropout

        y = self.norm_expr1(expr_emb)
        V_expr = self.expr_v(y)
        attn_expr = self.expr_attn(gene_emb, expr_emb, V_expr, M)
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_gene, out_expr


class NewDeepSCTransformerBlock(nn.Module):
    """
    使用新的手动实现注意力层的Transformer块
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        ffn_dropout,
        fused,
        num_layers_ffn=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # 使用新的注意力层，不再需要BranchV
        self.gene_attn = NewGeneAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.expr_attn = NewExpressionAttentionLayer(
            embedding_dim, num_heads, attn_dropout, fused
        )

        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)

        # FFN
        self.ffn_gene = FeedForward(
            embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        )
        self.ffn_expr = FeedForward(
            embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        )
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, gene_emb, expr_emb, M=None):
        """
        Args:
            gene_emb: 基因嵌入, shape: (batch_size, seq_len, embedding_dim)
            expr_emb: 表达嵌入, shape: (batch_size, seq_len, embedding_dim)
            M: 门控矩阵, shape: (batch_size, seq_len, seq_len) 或 None
        Returns:
            out_gene: 更新后的基因嵌入
            out_expr: 更新后的表达嵌入
        """
        # Gene branch
        x = self.norm_gene1(gene_emb)
        attn_gene = self.gene_attn(x, M)
        x = gene_emb + self.dropout(attn_gene)
        x_ln = self.norm_gene2(x)
        ffn_gene = self.ffn_gene(x_ln)
        out_gene = x + self.dropout(ffn_gene)

        # Expression branch
        y = self.norm_expr1(expr_emb)
        attn_expr = self.expr_attn(gene_emb, y, M)
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

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
        num_bins=8,
        alpha=0.1,
        mask_layer_start=None,
        enable_l0=True,
        enable_mse=True,
        enable_ce=True,
        num_layers_ffn=2,
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
                    embedding_dim,
                    num_heads,
                    attn_dropout,
                    ffn_dropout,
                    fused,
                    num_layers_ffn,
                )
                for _ in range(num_layers)
            ]
        )
        self.gumbel_softmax = CategoricalGumbelSoftmax(embedding_dim)  # 默认参数
        self.mask_layer_start = (
            mask_layer_start if mask_layer_start is not None else len(self.layers) - 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),  # 升维
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),  # 降回原维
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_bins + 1),  # 输出类别
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.enable_l0 = enable_l0
        self.enable_mse = enable_mse
        self.enable_ce = enable_ce
        # 初始化 classifier 内所有 Linear 层的权重和偏置
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        gene_ids,
        expression_bin,
        normalized_expr,
        return_encodings=False,
        return_mask_prob=True,
    ):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression: (batch, g)  # 表达量
        M: (batch, num_heads, g, g) 门控矩阵
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression_bin, normalized_expr)  # (batch, g, d)
        M, y = self.gumbel_softmax(gene_emb)  # (batch, g, g), (batch, g, g, 3)
        for i, layer in enumerate(self.layers):
            if i >= self.mask_layer_start:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, M)
            else:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, None)

        if self.enable_mse and self.enable_ce:
            regression_output = self.regressor(expr_emb)
            regression_output = regression_output.squeeze(-1)
            logits = self.classifier(expr_emb)
            return logits, regression_output, y, gene_emb, expr_emb
        elif self.enable_mse:
            regression_output = self.regressor(expr_emb)
            regression_output = regression_output.squeeze(-1)
            return regression_output, y, gene_emb, expr_emb
        elif self.enable_ce:
            logits = self.classifier(expr_emb)
            return logits, y, gene_emb, expr_emb


class NewDeepSC(nn.Module):
    """
    使用新的手动实现注意力层的DeepSC模型
    """

    def __init__(
        self,
        embedding_dim,
        num_genes,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        fused=True,
        num_bins=8,
        alpha=0.1,
        mask_layer_start=None,
        enable_l0=True,
        enable_mse=True,
        enable_ce=True,
        num_layers_ffn=2,
    ):
        super().__init__()
        self.gene_embedding = GeneEmbedding(embedding_dim, num_genes)
        self.expr_embedding = ExpressionEmbedding(
            embedding_dim, num_bins=num_bins, alpha=alpha
        )
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                NewDeepSCTransformerBlock(
                    embedding_dim,
                    num_heads,
                    attn_dropout,
                    ffn_dropout,
                    fused,
                    num_layers_ffn,
                )
                for _ in range(num_layers)
            ]
        )
        self.gumbel_softmax = CategoricalGumbelSoftmax(embedding_dim)  # 默认参数
        self.mask_layer_start = (
            mask_layer_start if mask_layer_start is not None else len(self.layers) - 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),  # 升维
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),  # 降回原维
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_bins + 1),  # 输出类别
        )
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.enable_l0 = enable_l0
        self.enable_mse = enable_mse
        self.enable_ce = enable_ce
        # 初始化 classifier 内所有 Linear 层的权重和偏置
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        gene_ids,
        expression_bin,
        normalized_expr,
        return_encodings=False,
        return_mask_prob=True,
    ):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression_bin: (batch, g)  # 离散化的表达量
        normalized_expr: (batch, g)  # 归一化的表达量
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression_bin, normalized_expr)  # (batch, g, d)
        M, y = self.gumbel_softmax(gene_emb)  # (batch, g, g), (batch, g, g, 3)

        for i, layer in enumerate(self.layers):
            if i >= self.mask_layer_start:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, M)
            else:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, None)

        if self.enable_mse and self.enable_ce:
            regression_output = self.regressor(expr_emb)
            regression_output = regression_output.squeeze(-1)
            logits = self.classifier(expr_emb)
            return logits, regression_output, y, gene_emb, expr_emb
        elif self.enable_mse:
            regression_output = self.regressor(expr_emb)
            regression_output = regression_output.squeeze(-1)
            return regression_output, y, gene_emb, expr_emb
        elif self.enable_ce:
            logits = self.classifier(expr_emb)
            return logits, y, gene_emb, expr_emb
