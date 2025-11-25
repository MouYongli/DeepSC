import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash Attention v2 support
try:
    from torch.nn.functional import scaled_dot_product_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: Flash Attention not available, falling back to standard attention")


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
    def __init__(self, embedding_dim: int, num_bins: int = 50, alpha: float = 0.3):
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
        self.continuous_projection = nn.Linear(1, embedding_dim, bias=True)

        # 初始化权重
        nn.init.xavier_uniform_(self.bin_embedding.weight)
        nn.init.xavier_uniform_(self.continuous_projection.weight)
        nn.init.zeros_(self.continuous_projection.bias)

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
        continuous_component = self.continuous_projection(normalized_expr.unsqueeze(-1))

        expr_embeddings = discrete_embeddings + continuous_component

        return expr_embeddings


class FlashAttentionLayer(nn.Module):
    """
    统一的 Flash Attention v2 注意力层

    接受 Q, K, V 的输入嵌入,计算多头注意力。

    Args:
        d: 嵌入维度
        num_heads: 注意力头数
        attn_dropout: 注意力dropout率
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

    def forward(self, Q_emb, K_emb=None, V_emb=None):
        """
        前向传播

        Args:
            Q_emb: Query嵌入, shape: (batch_size, seq_len, d)
            K_emb: Key嵌入, shape: (batch_size, seq_len, d)
                   如果为None,使用Q_emb (self-attention)
            V_emb: Value嵌入, shape: (batch_size, seq_len, d)
                   如果为None,使用K_emb

        Returns:
            output: 注意力输出, shape: (batch_size, seq_len, d)

        使用示例:
            # Self-attention: Q = K = V
            out = layer(x)

            # Cross-attention: Q != K = V
            out = layer(query, key_value)

            # 完全自定义: Q, K, V 都不同
            out = layer(q, k, v)
        """
        # 默认值处理
        if K_emb is None:
            K_emb = Q_emb
        if V_emb is None:
            V_emb = K_emb

        batch_size, seq_len, _ = Q_emb.shape

        # 计算 Q, K, V 投影
        Q = self.W_Q(Q_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_K(K_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_V(V_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 使用 Flash Attention v2
        if FLASH_ATTENTION_AVAILABLE:

            output = scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            # 应用 softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            A_bar = attn_weights

            # 计算输出
            output = torch.matmul(A_bar, V)

        # 转置并重塑
        output = output.transpose(
            1, 2
        ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d)  # (batch_size, seq_len, d)

        # 输出投影
        output = self.out_proj(output)

        return output


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


class MoERegressor(nn.Module):
    """
    Mixture of Experts (MoE) 回归器

    包含一个gate网络和三个expert网络：
    - Expert 1: 专门处理小值（低表达量）
    - Expert 2: 专门处理中等值（中等表达量）
    - Expert 3: 专门处理大值（高表达量）

    所有expert使用相同的网络结构，通过gate网络学习专门化
    """

    def __init__(
        self, embedding_dim, dropout=0.1, number_of_experts=3, gate_temperature=1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = number_of_experts
        self.gate_temperature = gate_temperature

        # Gate网络：决定每个expert的权重
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, self.num_experts),
        )

        # 创建三个相同结构的expert网络
        self.experts = nn.ModuleList(
            [RegressorExpert(embedding_dim, dropout) for _ in range(self.num_experts)]
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化所有网络的权重"""
        # 初始化gate网络
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # 初始化所有expert网络
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)
        Returns:
            output: (batch_size, seq_len)
            gate_weights: (batch_size, seq_len, num_experts)
        """
        # 1) gate logits -> softmax 权重
        gate_logits = self.gate(x)  # (B, L, E)
        gate_weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)

        # 2) 逐个 expert 前向
        expert_outputs = []
        for expert in self.experts:
            y = expert(x)  # (B, L, 1)
            expert_outputs.append(y)

        # 3) 堆叠 -> (B, L, E)
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # (B, L, E)

        # 4) 加权求和 -> (B, L)
        output = torch.sum(gate_weights * expert_outputs, dim=-1)

        return output, gate_weights


class RegressorExpert(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # ---- 第一子层 ----
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual  # 残差连接
        x = self.norm1(x)  # Post-Norm

        # ---- 第二子层 ----
        # 输出回归值，维度从 embedding_dim -> 1
        x = self.fc3(x)  # (B, L, 1)

        return x


import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, moe_cfg):
        super().__init__()
        self.dim = moe_cfg.dim
        self.topk = moe_cfg.n_activated_experts
        self.score_func = moe_cfg.score_func
        self.route_scale = moe_cfg.route_scale
        self.n_routed_experts = int(moe_cfg.n_routed_experts)
        self.proj = nn.Linear(self.dim, self.n_routed_experts, bias=True)

        # 用于统计专家使用情况
        self.register_buffer("expert_usage_count", torch.zeros(self.n_routed_experts))
        self.register_buffer("total_tokens", torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, dim]
        Returns:
            weights: [B, topk]
            indices: [B, topk]
        """
        # [B, n_experts]
        scores = self.proj(x)

        # softmax/sigmoid
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        indices = torch.topk(scores, self.topk, dim=-1)[1]  # [B, topk]
        weights = original_scores.gather(1, indices)  # [B, topk]

        # # 归一化
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * self.route_scale

        # 统计专家使用情况（仅在训练时）
        if self.training:
            batch_size = x.size(0)
            # 修正：每个token选择topk个专家，所以总的专家选择次数是 batch_size * topk
            self.total_tokens += batch_size * self.topk

            # 统计每个专家被选中的次数
            for expert_idx in range(self.n_routed_experts):
                count = (indices == expert_idx).sum().float()
                self.expert_usage_count[expert_idx] += count

        return weights.type_as(x), indices

    def get_expert_usage_stats(self):
        """
        获取专家使用统计信息

        Returns:
            dict: 包含各种统计指标的字典
        """
        if self.total_tokens == 0:
            return {
                "expert_usage_ratio": torch.zeros(self.n_routed_experts),
                "max_usage_ratio": 0.0,
                "min_usage_ratio": 0.0,
                "usage_variance": 0.0,
                "collapse_ratio": 0.0,
                "entropy": 0.0,
                "total_tokens": 0,
            }

        # 计算每个专家的使用比例
        usage_ratio = self.expert_usage_count / self.total_tokens

        # 计算统计指标
        max_usage = usage_ratio.max().item()
        min_usage = usage_ratio.min().item()
        usage_variance = usage_ratio.var().item()

        # 计算塌缩比例（最常用专家的使用比例）
        collapse_ratio = max_usage

        # 计算熵（衡量专家使用的均匀程度）
        # 避免log(0)的情况
        epsilon = 1e-10
        usage_ratio_safe = usage_ratio + epsilon
        entropy = -(usage_ratio_safe * torch.log(usage_ratio_safe)).sum().item()

        return {
            "expert_usage_ratio": usage_ratio,
            "max_usage_ratio": max_usage,
            "min_usage_ratio": min_usage,
            "usage_variance": usage_variance,
            "collapse_ratio": collapse_ratio,
            "entropy": entropy,
            "total_tokens": self.total_tokens.item(),
        }

    def reset_usage_stats(self):
        """重置使用统计"""
        self.expert_usage_count.zero_()
        self.total_tokens.zero_()

    def is_collapsed(self, threshold=0.8):
        """
        判断MoE是否塌缩

        Args:
            threshold: 塌缩阈值，如果某个专家使用比例超过此阈值则认为塌缩

        Returns:
            bool: 是否塌缩
        """
        stats = self.get_expert_usage_stats()
        return stats["collapse_ratio"] > threshold


class Expert(nn.Module):
    def __init__(self, moe_cfg, p_dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(moe_cfg.dim, moe_cfg.moe_inter_dim, bias=use_bias)
        self.w3 = nn.Linear(moe_cfg.dim, moe_cfg.moe_inter_dim, bias=use_bias)
        self.w2 = nn.Linear(moe_cfg.moe_inter_dim, moe_cfg.dim, bias=use_bias)
        self.dropout = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU / Gated-MLP
        h = F.silu(self.w1(x)) * self.w3(x)
        h = self.dropout(h)
        out = self.w2(h)
        return out


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(
        self,
        moe_cfg,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = moe_cfg.dim
        self.n_routed_experts = int(moe_cfg.n_routed_experts)
        self.n_activated_experts = int(moe_cfg.n_activated_experts)
        self.moe_inter_dim = int(moe_cfg.moe_inter_dim)
        self.n_shared_experts = int(moe_cfg.n_shared_experts)
        self.route_scale = float(moe_cfg.route_scale)
        self.moe_cfg = moe_cfg
        self.gate = Gate(self.moe_cfg)

        self.experts = nn.ModuleList(
            [Expert(self.moe_cfg) for i in range(self.n_routed_experts)]
        )
        self.shared_experts = MLP(self.dim, self.n_shared_experts * self.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)

        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)

    def get_expert_usage_stats(self):
        """获取专家使用统计信息"""
        return self.gate.get_expert_usage_stats()

    def reset_usage_stats(self):
        """重置使用统计"""
        self.gate.reset_usage_stats()

    def is_collapsed(self, threshold=0.8):
        """判断MoE是否塌缩"""
        return self.gate.is_collapsed(threshold)


class FlashDeepSCTransformerBlock(nn.Module):
    """
    使用 Flash Attention v2 的Transformer块
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        ffn_dropout,
        num_layers_ffn=2,
        moe_cfg=None,
        moe_layer=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.use_moe_ffn = bool(moe_layer)
        # 使用 Flash Attention 层
        self.gene_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.expr_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)

        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)

        # FFN
        # self.ffn_gene = FeedForward(
        #     embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        # )
        # self.ffn_expr = FeedForward(
        #     embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        # )
        self.dropout = nn.Dropout(ffn_dropout)
        self.ffn_gene = (
            MoE(moe_cfg)
            if self.use_moe_ffn
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )
        self.ffn_expr = (
            MoE(moe_cfg)
            if self.use_moe_ffn
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )

    def forward(self, gene_emb, expr_emb):
        """
        Args:
            gene_emb: 基因嵌入, shape: (batch_size, seq_len, embedding_dim)
            expr_emb: 表达嵌入, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_gene: 更新后的基因嵌入
            out_expr: 更新后的表达嵌入
        """
        # Gene self-attention
        x = self.norm_gene1(gene_emb)
        attn_gene = self.gene_attn(x)  # Self-attention: Q=K=V=x
        x = gene_emb + self.dropout(attn_gene)
        x_ln = self.norm_gene2(x)
        ffn_gene = self.ffn_gene(x_ln)
        out_gene = x + self.dropout(ffn_gene)

        # Expression cross-attention with gene
        y = self.norm_expr1(expr_emb)
        attn_expr = self.expr_attn(y, gene_emb)  # Cross-attention: Q=y, K=V=gene_emb
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_gene, out_expr


class FlashDeepSCTransformerCrossAttentionBlock(nn.Module):
    """
    使用 Flash Attention v2 的Transformer块
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        ffn_dropout,
        num_layers_ffn=2,
        moe_cfg=None,
        moe_layer=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.use_moe_ffn = bool(moe_layer)
        # 使用 Flash Attention 层
        self.gene_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.expr_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)

        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)

        # FFN
        # self.ffn_gene = FeedForward(
        #     embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        # )
        # self.ffn_expr = FeedForward(
        #     embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
        # )
        self.dropout = nn.Dropout(ffn_dropout)
        self.ffn_gene = (
            MoE(moe_cfg)
            if self.use_moe_ffn
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )
        self.ffn_expr = (
            MoE(moe_cfg)
            if self.use_moe_ffn
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )

    def forward(self, gene_emb, expr_emb):
        """
        Args:
            gene_emb: 基因嵌入, shape: (batch_size, seq_len, embedding_dim)
            expr_emb: 表达嵌入, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_gene: 更新后的基因嵌入
            out_expr: 更新后的表达嵌入
        """
        # Gene self-attention
        x = self.norm_gene1(gene_emb)
        attn_gene = self.gene_attn(x)  # Self-attention: Q=K=V=x
        x = gene_emb + self.dropout(attn_gene)
        x_ln = self.norm_gene2(x)
        ffn_gene = self.ffn_gene(x_ln)
        out_gene = x + self.dropout(ffn_gene)

        # Expression cross-attention with gene
        y = self.norm_expr1(expr_emb)
        attn_expr = self.expr_attn(y, gene_emb)  # Cross-attention: Q=y, K=V=gene_emb
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_gene, out_expr


class FlashDeepSCTransformerSelfAttentionBlock(nn.Module):
    """
    使用 Flash Attention v2 的Transformer块
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        attn_dropout,
        ffn_dropout,
        num_layers_ffn=2,
        moe_cfg=None,
        use_moe_in_layer=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.use_moe_in_layer = use_moe_in_layer
        self.expr_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.norm_expr1 = nn.LayerNorm(embedding_dim)
        self.norm_expr2 = nn.LayerNorm(embedding_dim)

        self.ffn_expr = (
            MoE(moe_cfg)
            if self.use_moe_in_layer
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, embedding):
        """
        Args:
            embedding: 基因或者表达嵌入, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_embedding: 更新后的嵌入
        """

        y = self.norm_expr1(embedding)
        attn_expr = self.expr_attn(y)  # Self-attention: Q=K=V=y
        y = embedding + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_expr


class DeepSC(nn.Module):
    """
    使用 Flash Attention v2 的DeepSC模型
    当前进行四个实验比较：
    1. ：先对基因嵌入进行N层self-attention，
            然后用这个得到的嵌入作为Q和K，用表达嵌入作为V进行N层cross attention之后，
            然后在进行M层expression的self-attention
    2. 基因嵌入和表达嵌入进行双流的cross attention
    用以上结果较好的结构作为下面两个实验的基础：
    3. KV来自基因嵌入，Q来自表达嵌入的单流attention
    4. Q来自基因嵌入，KV来自表达嵌入的单流attention

    """

    def __init__(
        self,
        embedding_dim,
        num_genes,
        num_layers=4,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        num_bins=8,
        alpha=0.1,
        mask_layer_start=None,
        enable_l0=True,
        enable_mse=True,
        enable_ce=True,
        num_layers_ffn=2,
        use_moe_regressor=True,
        number_of_experts=3,
        gene_embedding_participate_til_layer=3,
        moe=None,
        experiment=1,
    ):
        super().__init__()
        self.gene_embedding_participate_til_layer = gene_embedding_participate_til_layer
        self.gene_embedding = GeneEmbedding(embedding_dim, num_genes)
        self.expr_embedding = ExpressionEmbedding(
            embedding_dim, num_bins=num_bins, alpha=alpha
        )
        num_layers_expr = num_layers - gene_embedding_participate_til_layer
        self.num_heads = num_heads
        if experiment == 2:
            self.layers = nn.ModuleList()
            for i in range(gene_embedding_participate_til_layer):
                self.layers.append(
                    FlashDeepSCTransformerBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                    )
                )
            self.expression_layers = nn.ModuleList()
            for i in range(num_layers_expr):
                moe_layer = i >= (num_layers_expr - moe.n_moe_layers)
                self.expression_layers.append(
                    FlashDeepSCTransformerSelfAttentionBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                        use_moe_in_layer=moe_layer and moe.use_moe_ffn,
                    )
                )
        elif experiment == 1:
            self.gene_self_attention_layers = nn.ModuleList()
            for i in range(gene_embedding_participate_til_layer):
                self.gene_self_attention_layers.append(
                    FlashDeepSCTransformerSelfAttentionBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                        use_moe_in_layer=moe_layer and moe.use_moe_ffn,
                    )
                )
            self.cross_attention_layers = nn.ModuleList()
            for i in range(gene_embedding_participate_til_layer):
                self.cross_attention_layers.append(
                    FlashDeepSCTransformerBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                    )
                )

            self.expression_layers = nn.ModuleList()
            for i in range(num_layers_expr):
                moe_layer = i >= (num_layers_expr - moe.n_moe_layers)
                self.expression_layers.append(
                    FlashDeepSCTransformerSelfAttentionBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                        use_moe_in_layer=moe_layer and moe.use_moe_ffn,
                    )
                )

        # 如果你用的是 DictConfig
        # moe_cfg = MoECfg(**cfg.moe)
        # self.moe, self.n_local_experts = build_moe_from_cfg(moe_cfg)
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

        # 根据配置选择使用哪种regressor
        self.use_moe_regressor = use_moe_regressor
        if self.use_moe_regressor:
            self.regressor = MoERegressor(
                embedding_dim=embedding_dim,
                dropout=ffn_dropout,
                number_of_experts=number_of_experts,
                gate_temperature=1.0,
            )
        else:
            # 原来的simple regressor
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
        self.fused_emb_proj = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(
        self,
        gene_ids,
        expression_bin,
        normalized_expr,
        return_encodings=False,
        return_mask_prob=True,
        return_gate_weights=False,
    ):
        """
        gene_ids: (batch, g)  # 基因ID序列
        expression_bin: (batch, g)  # 离散化的表达量
        normalized_expr: (batch, g)  # 归一化的表达量
        return_gate_weights: 是否返回MoE的gate权重
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(expression_bin, normalized_expr)  # (batch, g, d)
        for i, layer in enumerate(self.layers):
            gene_emb, expr_emb = layer(gene_emb, expr_emb)
        for i, layer in enumerate(self.expression_layers):
            expr_emb = layer(expr_emb)
        final_emb = torch.cat([gene_emb, expr_emb], dim=-1)
        final_emb = self.fused_emb_proj(final_emb)  # (batch, g, d)
        if self.enable_mse and self.enable_ce:
            logits = self.classifier(final_emb)
            if return_gate_weights and self.use_moe_regressor:
                regression_output, gate_weights = self.get_regressor_output(final_emb)
                return logits, regression_output, gene_emb, expr_emb, gate_weights
            else:
                regression_output, _ = self.get_regressor_output(final_emb)
                return logits, regression_output, gene_emb, expr_emb
        elif self.enable_mse:
            if return_gate_weights and self.use_moe_regressor:
                regression_output, gate_weights = self.get_regressor_output(final_emb)
                return regression_output, gene_emb, expr_emb, gate_weights
            else:
                regression_output, _ = self.get_regressor_output(final_emb)
                return regression_output, gene_emb, expr_emb
        elif self.enable_ce:
            logits = self.classifier(final_emb)
            return logits, gene_emb, expr_emb

    def get_regressor_output(self, final_emb):
        """
        获取回归器输出

        Args:
            final_emb: 最终融合的嵌入特征

        Returns:
            regression_output: 回归预测结果
            gate_weights: MoE的gate权重（如果使用MoE）或None（如果使用简单回归器）
        """
        # 根据使用的regressor类型获取输出
        if self.use_moe_regressor:
            # MoE regressor返回两个值：regression_output和gate_weights
            regression_output, gate_weights = self.regressor(final_emb)
        else:
            # 原来的regressor只返回一个值
            regression_output = self.regressor(final_emb)
            regression_output = regression_output.squeeze(-1)  # 去掉最后一个维度
            gate_weights = None  # 原来的regressor没有gate_weights

        return regression_output, gate_weights

    def get_all_moe_stats(self):
        """
        获取模型中所有MoE层的统计信息

        Returns:
            dict: 包含所有MoE层统计信息的字典
        """
        moe_stats = {}

        # 检查transformer层中的MoE
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "ffn_gene") and isinstance(layer.ffn_gene, MoE):
                moe_stats[f"transformer_layer_{i}_gene_ffn"] = (
                    layer.ffn_gene.get_expert_usage_stats()
                )
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                moe_stats[f"transformer_layer_{i}_expr_ffn"] = (
                    layer.ffn_expr.get_expert_usage_stats()
                )

        # 检查expression层中的MoE
        for i, layer in enumerate(self.expression_layers):
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                moe_stats[f"expression_layer_{i}_expr_ffn"] = (
                    layer.ffn_expr.get_expert_usage_stats()
                )

        # 检查MoE regressor
        if self.use_moe_regressor and hasattr(self.regressor, "gate"):
            moe_stats["moe_regressor"] = {
                "expert_usage_ratio": (
                    self.regressor.gate_weights_history
                    if hasattr(self.regressor, "gate_weights_history")
                    else "Not available"
                )
            }

        return moe_stats

    def check_moe_collapse(self, threshold=0.8):
        """
        检查模型中是否有MoE层发生塌缩

        Args:
            threshold: 塌缩阈值

        Returns:
            dict: 包含塌缩检测结果的字典
        """
        collapse_results = {}
        moe_stats = self.get_all_moe_stats()

        for layer_name, stats in moe_stats.items():
            if isinstance(stats, dict) and "collapse_ratio" in stats:
                is_collapsed = stats["collapse_ratio"] > threshold
                collapse_results[layer_name] = {
                    "is_collapsed": is_collapsed,
                    "collapse_ratio": stats["collapse_ratio"],
                    "entropy": stats["entropy"],
                    "expert_usage_ratio": (
                        stats["expert_usage_ratio"].tolist()
                        if hasattr(stats["expert_usage_ratio"], "tolist")
                        else stats["expert_usage_ratio"]
                    ),
                }

        return collapse_results

    def reset_all_moe_stats(self):
        """重置所有MoE层的使用统计"""
        # 重置transformer层中的MoE
        for layer in self.layers:
            if hasattr(layer, "ffn_gene") and isinstance(layer.ffn_gene, MoE):
                layer.ffn_gene.reset_usage_stats()
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                layer.ffn_expr.reset_usage_stats()

        # 重置expression层中的MoE
        for layer in self.expression_layers:
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                layer.ffn_expr.reset_usage_stats()

    def print_moe_collapse_report(self, threshold=0.8):
        """
        打印MoE塌缩检测报告

        Args:
            threshold: 塌缩阈值
        """
        print(f"\n{'='*60}")
        print("MoE Collapse Detection Report")
        print(f"{'='*60}")
        print(f"Collapse Threshold: {threshold}")

        collapse_results = self.check_moe_collapse(threshold)

        if not collapse_results:
            print("No MoE layers found in the model.")
            return

        collapsed_layers = []
        healthy_layers = []

        for layer_name, result in collapse_results.items():
            if result["is_collapsed"]:
                collapsed_layers.append((layer_name, result))
            else:
                healthy_layers.append((layer_name, result))

        print(f"\nTotal MoE layers: {len(collapse_results)}")
        print(f"Collapsed layers: {len(collapsed_layers)}")
        print(f"Healthy layers: {len(healthy_layers)}")

        if collapsed_layers:
            print("\n🚨 COLLAPSED LAYERS:")
            for layer_name, result in collapsed_layers:
                print(f"  - {layer_name}:")
                print(f"    Collapse Ratio: {result['collapse_ratio']:.4f}")
                print(f"    Entropy: {result['entropy']:.4f}")
                print(
                    f"    Expert Usage: {[f'{x:.3f}' for x in result['expert_usage_ratio']]}"
                )

        if healthy_layers:
            print("\n✅ HEALTHY LAYERS:")
            for layer_name, result in healthy_layers:
                print(f"  - {layer_name}:")
                print(f"    Collapse Ratio: {result['collapse_ratio']:.4f}")
                print(f"    Entropy: {result['entropy']:.4f}")
                print(
                    f"    Expert Usage: {[f'{x:.3f}' for x in result['expert_usage_ratio']]}"
                )

        print(f"\n{'='*60}")

        # 返回是否有塌缩
        return len(collapsed_layers) > 0
