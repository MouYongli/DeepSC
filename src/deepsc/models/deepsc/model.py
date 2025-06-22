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
        # 离散表达水平的嵌入矩阵 W_bin ∈ R^{d×N}
        self.bin_embedding = nn.Embedding(
            num_embeddings=num_bins + 3, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            expression: 表达量向量 x = [x_1, x_2, ..., x_g], shape: (batch_size, g)

        Returns:
            expr_embeddings: 表达量嵌入 E_expr ∈ R^{g×d}, shape: (batch_size, g, d)
        """
        expr_embeddings = self.bin_embedding(expression)
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
        self.ffn_gene = FeedForward(embedding_dim, dropout=ffn_dropout)
        self.ffn_expr = FeedForward(embedding_dim, dropout=ffn_dropout)

    def forward(self, gene_emb, expr_emb, M=None):
        # QKV: (batch, g, num_heads, head_dim)
        V_gene = self.gene_v(gene_emb)
        attn_gene = self.gene_attn(gene_emb, V_gene, M)
        h_gene = self.norm_gene1(gene_emb + attn_gene)
        ffn_gene = self.ffn_gene(h_gene)
        out_gene = self.norm_gene2(h_gene + ffn_gene)
        # Expression branch: 融合 gene 的 Q/K 参与注意力
        V_expr = self.expr_v(expr_emb)
        attn_expr = self.expr_attn(gene_emb, expr_emb, V_expr, M)
        h_expr = self.norm_expr1(expr_emb + attn_expr)
        ffn_expr = self.ffn_expr(h_expr)
        out_expr = self.norm_expr2(h_expr + ffn_expr)
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
        self.mask_layer_start = (
            mask_layer_start if mask_layer_start is not None else len(self.layers) - 1
        )
        self.classifier = nn.Linear(embedding_dim, num_bins)
        self.regressor = nn.Linear(embedding_dim, 1)
        self.enable_l0 = enable_l0
        self.enable_mse = enable_mse
        self.enable_ce = enable_ce
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self, gene_ids, expression, return_encodings=False, return_mask_prob=True
    ):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression: (batch, g)  # 表达量
        M: (batch, num_heads, g, g) 门控矩阵
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression)  # (batch, g, d)
        M, y = self.gumbel_softmax(gene_emb)  # (batch, g, g), (batch, g, g, 3)
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i >= self.mask_layer_start:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, M)
            else:
                gene_emb, expr_emb = layer(gene_emb, expr_emb, None)
        if return_encodings:
            return gene_emb, expr_emb
        # TODO: 是否需要去掉CLS token？ 我这里没有去掉，因为在它label里面是-100，所以被loss函数忽略了
        logits = self.classifier(expr_emb)
        if self.enable_ce and self.enable_mse and self.enable_l0:
            regression_output = self.regressor(expr_emb)
            regression_output = regression_output.squeeze(-1)
            return logits, regression_output, y
        elif self.enable_ce:
            return logits, y
