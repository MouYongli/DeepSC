import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsc.models.scgpt_pert import map_raw_id_to_vocab_id

# Flash Attention v2 support
try:
    from torch.nn.functional import scaled_dot_product_attention

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: Flash Attention not available, falling back to standard attention")


# TODO: The number of embedding rows is not just num_genes, but num_genes+1, because there is also <cls> token
class GeneEmbedding(nn.Module):
    """
    Gene Embedding branch: focuses on capturing semantic representations of genes

    Learn semantic representations of genes, including:
        - Functional similarity: functionally related genes are close in embedding space
        - Pathway relationships: genes in the same biological pathway have similar representations
        - Regulatory relationships: relationships between transcription factors and their target genes
    """

    # num_genes is the number of genes, including <cls> and <pad>
    def __init__(self, embedding_dim: int, num_genes: int):
        super(GeneEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.gene_embedding = nn.Embedding(
            num_embeddings=num_genes + 2, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_ids: gene ID sequence G = [g_1, g_2, ..., g_g], shape: (batch_size, g)

        Returns:
            gene_embeddings: gene embedding E_gene ∈ R^{g×d}, shape: (batch_size, g, d)
        """
        return self.gene_embedding(gene_ids)


# Needs modification, there's an issue: normalization and binning should be in data_collator, here only do embedding
# Also: need to properly do truncation, padding and mask in data_collator.
class ExpressionEmbedding(nn.Module):
    """
    Expression Embedding branch: focuses on capturing numerical features and contextual dependencies of expression values

    Considering the characteristics of scRNA-seq data, design hierarchical encoding strategy:
    1. Expression normalization and discretization
    2. Hierarchical expression embedding
    """

    # num_bins is the number of bins, including <cls>, <pad> and <mask>
    def __init__(self, embedding_dim: int, num_bins: int = 50, alpha: float = 0.3):
        """
        Args:
            embedding_dim: embedding dimension d
            num_bins: number of discretization bins N
            alpha: weight parameter to balance discrete and continuous features
        """
        super(ExpressionEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        # Can alpha be a learnable parameter
        self.alpha = alpha
        # Embedding matrix for discrete expression levels W_bin ∈ R^{d×N}
        self.bin_embedding = nn.Embedding(
            num_embeddings=num_bins + 3, embedding_dim=embedding_dim, padding_idx=0
        )
        # Projection vector for continuous values v_cont ∈ R^d
        self.continuous_projection = nn.Linear(1, embedding_dim, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.bin_embedding.weight)
        nn.init.xavier_uniform_(self.continuous_projection.weight)
        nn.init.zeros_(self.continuous_projection.bias)

    def forward(
        self, discrete_expression: torch.Tensor, normalized_expr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            expression: Expression vector x = [x_1, x_2, ..., x_g], shape: (batch_size, g)

        Returns:
            expr_embeddings: Expression embedding E_expr ∈ R^{g×d}, shape: (batch_size, g, d)
        """

        discrete_embeddings = self.bin_embedding(discrete_expression)
        continuous_component = self.continuous_projection(normalized_expr.unsqueeze(-1))

        expr_embeddings = discrete_embeddings + continuous_component

        return expr_embeddings


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class PerturbationEmbedding(nn.Module):
    """
    Perturbation Embedding branch: focuses on capturing perturbation information representation

    Learn perturbation information representation, only includes whether perturbed
    """

    def __init__(self, embedding_dim: int, num_genes: int):
        super(PerturbationEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.perturbation_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, perturbation_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perturbation_ids: Perturbation ID sequence P = [p_1, p_2, ..., p_p], shape: (batch_size, p)

        Returns:
            perturbation_embeddings: Perturbation embedding E_perturbation ∈ R^{g×d}, shape: (batch_size, g, d)
        """
        return self.perturbation_embedding(perturbation_ids)


class FlashAttentionLayer(nn.Module):
    """
    Unified Flash Attention v2 attention layer

    Accepts Q, K, V input embeddings and computes multi-head attention.

    Args:
        d: embedding dimension
        num_heads: Number of attention heads
        attn_dropout: Attention dropout rate
    """

    def __init__(self, d, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.d = d

        # Q, K, V projection matrices
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)

        # Output projection
        self.out_proj = nn.Linear(d, d)

        # Dropout
        self.dropout = nn.Dropout(attn_dropout)

        # Scale factor
        self.scale = self.head_dim**-0.5

    def forward(self, Q_emb, K_emb=None, V_emb=None):
        """
        Forward pass

        Args:
            Q_emb: Query embedding, shape: (batch_size, seq_len, d)
            K_emb: Key embedding, shape: (batch_size, seq_len, d)
                   If None, use Q_emb (self-attention)
            V_emb: Value embedding, shape: (batch_size, seq_len, d)
                   If None, use K_emb

        Returns:
            output: Attention output, shape: (batch_size, seq_len, d)

        Usage examples:
            # Self-attention: Q = K = V
            out = layer(x)

            # Cross-attention: Q != K = V
            out = layer(query, key_value)

            # Fully customized: Q, K, V are all different
            out = layer(q, k, v)
        """
        # Default value handling
        if K_emb is None:
            K_emb = Q_emb
        if V_emb is None:
            V_emb = K_emb

        batch_size, seq_len, _ = Q_emb.shape

        # Compute Q, K, V projections
        Q = self.W_Q(Q_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_K(K_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_V(V_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Use Flash Attention v2
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

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            A_bar = attn_weights

            # Compute output
            output = torch.matmul(A_bar, V)

        # Transpose and reshape
        output = output.transpose(
            1, 2
        ).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, seq_len, self.d)  # (batch_size, seq_len, d)

        # Output projection
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d, hidden_dim=None, dropout=0.1, num_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or d * 4
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create modules for each layer
        self.layers = nn.ModuleList()

        # First layer：d -> hidden_dim
        first_layer = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.layers.append(first_layer)

        # Middle layers：hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            middle_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.layers.append(middle_layer)

        # Last layer：hidden_dim -> d
        last_layer = nn.Sequential(nn.Linear(hidden_dim, d), nn.Dropout(dropout))
        self.layers.append(last_layer)

    def forward(self, x):
        if self.num_layers == 2:
            # Two-layer case: d -> hidden_dim -> d
            h = self.layers[0](x)  # d -> hidden_dim
            h = self.layers[1](h)  # hidden_dim -> d
            return x + h  # residual connection

        else:
            # Multi-layer case
            # First layer
            h = self.layers[0](x)  # d -> hidden_dim

            # Middle layers, each with residual connection
            for i in range(1, self.num_layers - 1):
                residual = h
                h = self.layers[i](h)  # hidden_dim -> hidden_dim
                h = h + residual  # residual connection

            # Last layer
            h = self.layers[-1](h)  # hidden_dim -> d
            return x + h  # final residual connection


class MoERegressor(nn.Module):
    """
    Mixture of Experts (MoE) Regressor

    Contains one gate network and three expert networks:
    - Expert 1: Specialized for handling small values (low expression)
    - Expert 2: Specialized for handling medium values (medium expression)
    - Expert 3: Specialized for handling large values (high expression)

    All experts use the same network structure, learning specialization through the gate network
    """

    def __init__(
        self, embedding_dim, dropout=0.1, number_of_experts=3, gate_temperature=1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = number_of_experts
        self.gate_temperature = gate_temperature

        # Gate network: determines weight of each expert
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, self.num_experts),
        )

        # Create three expert networks with the same structure
        self.experts = nn.ModuleList(
            [RegressorExpert(embedding_dim, dropout) for _ in range(self.num_experts)]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all networks"""
        # Initialize gate network
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize all expert networks
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
        # 1) gate logits -> softmax weights
        gate_logits = self.gate(x)  # (B, L, E)
        gate_weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)

        # 2) Forward pass through each expert
        expert_outputs = []
        for expert in self.experts:
            y = expert(x)  # (B, L, 1)
            expert_outputs.append(y)

        # 3) Stack -> (B, L, E)
        expert_outputs = torch.cat(expert_outputs, dim=-1)  # (B, L, E)

        # 4) Weighted sum -> (B, L)
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
        # ---- First sub-layer ----
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual  # residual connection
        x = self.norm1(x)  # Post-Norm

        # ---- Second sub-layer ----
        # Output regression value, dimension from embedding_dim -> 1
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

        # Used to track expert usage statistics
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

        # # Normalization
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * self.route_scale

        # Track expert usage statistics (only during training)
        if self.training:
            batch_size = x.size(0)
            # Correction: each token selects topk experts, so total expert selections = batch_size * topk
            self.total_tokens += batch_size * self.topk

            # Count how many times each expert is selected
            for expert_idx in range(self.n_routed_experts):
                count = (indices == expert_idx).sum().float()
                self.expert_usage_count[expert_idx] += count

        return weights.type_as(x), indices

    def get_expert_usage_stats(self):
        """
        Get expert usage statistics

        Returns:
            dict: Dictionary containing various statistical metrics
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

        # Calculate usage ratio for each expert
        usage_ratio = self.expert_usage_count / self.total_tokens

        # Calculate statistical metrics
        max_usage = usage_ratio.max().item()
        min_usage = usage_ratio.min().item()
        usage_variance = usage_ratio.var().item()

        # Calculate collapse ratio (usage ratio of most frequently used expert)
        collapse_ratio = max_usage

        # Calculate entropy (measure of uniformity of expert usage)
        # Avoid log(0)
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
        """Reset usage statistics"""
        self.expert_usage_count.zero_()
        self.total_tokens.zero_()

    def is_collapsed(self, threshold=0.8):
        """
        Determine if MoE has collapsed

        Args:
            threshold: Collapse threshold, considered collapsed if an expert's usage ratio exceeds this threshold

        Returns:
            bool: Whether collapsed
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
        """Get expert usage statistics"""
        return self.gate.get_expert_usage_stats()

    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.gate.reset_usage_stats()

    def is_collapsed(self, threshold=0.8):
        """Determine if MoE has collapsed"""
        return self.gate.is_collapsed(threshold)


class FlashDeepSCTransformerBlock(nn.Module):
    """
    Transformer block using Flash Attention v2
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
        cross_attention_architecture="A",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.use_moe_ffn = bool(moe_layer)
        # Use Flash Attention layer
        self.gene_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.expr_attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)

        # Norm
        self.norm_gene1 = nn.LayerNorm(embedding_dim)
        self.norm_gene2 = nn.LayerNorm(embedding_dim)
        self.norm_gene3 = nn.LayerNorm(embedding_dim)
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
        self.cross_attention_architecture = cross_attention_architecture

    def forward(self, gene_emb, expr_emb):
        """
        Args:
            gene_emb: Gene embedding, shape: (batch_size, seq_len, embedding_dim)
            expr_emb: Expression embedding, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_gene: Updated gene embedding
            out_expr: Updated expression embedding
        """
        # Gene self-attention
        x = self.norm_gene1(gene_emb)
        attn_gene = self.gene_attn(x)  # Self-attention: Q=K=V=x
        x = gene_emb + self.dropout(attn_gene)
        x_ln = self.norm_gene2(x)
        ffn_gene = self.ffn_gene(x_ln)
        out_gene = x + self.dropout(ffn_gene)

        out_gene_ln = self.norm_gene3(out_gene)
        # Expression cross-attention with gene
        y = self.norm_expr1(expr_emb)
        if self.cross_attention_architecture == "A":
            attn_expr = self.expr_attn(y, out_gene_ln, out_gene_ln)
        elif self.cross_attention_architecture == "B":
            attn_expr = self.expr_attn(out_gene_ln, out_gene_ln, y)
        elif self.cross_attention_architecture == "C":
            attn_expr = self.expr_attn(out_gene_ln, y, y)
        else:
            raise ValueError("Invalid cross_attention_architecture option.")
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm_expr2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_gene, out_expr


class FlashDeepSCTransformerCrossAttentionBlock(nn.Module):
    """
    Transformer block using Flash Attention v2
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
        cross_attention_architecture="A",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.use_moe_in_layer = use_moe_in_layer
        self.attn = FlashAttentionLayer(embedding_dim, num_heads, attn_dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.gene_norm = nn.LayerNorm(embedding_dim)
        self.ffn_expr = (
            MoE(moe_cfg)
            if self.use_moe_in_layer
            else FeedForward(
                embedding_dim, dropout=ffn_dropout, num_layers=num_layers_ffn
            )
        )
        self.dropout = nn.Dropout(ffn_dropout)
        self.cross_attention_architecture = cross_attention_architecture

    def forward(self, gene_emb, expr_emb):
        """
        Args:
            embedding: Gene or expression embedding, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_embedding: Updated embedding
        """
        gene_emb = self.gene_norm(gene_emb)
        y = self.norm1(expr_emb)
        if self.cross_attention_architecture == "A":
            attn_expr = self.attn(y, gene_emb, gene_emb)  # Self-attention: Q=K=V=y
        elif self.cross_attention_architecture == "B":
            attn_expr = self.attn(gene_emb, gene_emb, y)  # Self-attention: Q=K=V=y
        elif self.cross_attention_architecture == "C":
            attn_expr = self.attn(gene_emb, y, y)  # Self-attention: Q=K=V=y
        else:
            raise ValueError("Invalid cross_attention_architecture option.")
        y = expr_emb + self.dropout(attn_expr)
        y_ln = self.norm2(y)
        ffn_expr = self.ffn_expr(y_ln)
        out_expr = y + self.dropout(ffn_expr)

        return out_expr


class FlashDeepSCTransformerSelfAttentionBlock(nn.Module):
    """
    Transformer block using Flash Attention v2
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
            embedding: Gene or expression embedding, shape: (batch_size, seq_len, embedding_dim)
        Returns:
            out_embedding: Updated embedding
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
    DeepSC model using Flash Attention v2
    Currently comparing four experiments:
    1. First apply N layers of self-attention on gene embedding,
       then use the resulting embedding as Q and K, and expression embedding as V for N layers of cross attention,
       then perform M layers of expression self-attention
    2. Dual-stream cross attention between gene embedding and expression embedding
    Use the better structure from above as the basis for the following two experiments:
    3. Single-stream attention with KV from gene embedding, Q from expression embedding
    4. Single-stream attention with Q from gene embedding, KV from expression embedding

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
        enable_l0=True,
        enable_mse=True,
        enable_ce=True,
        num_layers_ffn=2,
        use_moe_regressor=True,
        number_of_experts=3,
        gene_embedding_participate_til_layer=3,
        moe=None,
        attention_stream=1,
        cross_attention_architecture="A",
    ):
        super().__init__()
        self.gene_embedding_participate_til_layer = gene_embedding_participate_til_layer
        self.gene_embedding = GeneEmbedding(embedding_dim, num_genes)
        self.expr_embedding = ContinuousValueEncoder(embedding_dim, dropout=0.1)
        self.pert_encoder = nn.Embedding(4, embedding_dim, padding_idx=0)
        self.cross_attention_architecture = cross_attention_architecture
        self.attention_stream = attention_stream
        num_layers_expr = num_layers - gene_embedding_participate_til_layer
        self.num_heads = num_heads
        if attention_stream == 2:
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
                        cross_attention_architecture=self.cross_attention_architecture,
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
        elif attention_stream == 1:
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
                    )
                )
            self.cross_attention_layers = nn.ModuleList()
            for i in range(gene_embedding_participate_til_layer):
                self.cross_attention_layers.append(
                    FlashDeepSCTransformerCrossAttentionBlock(
                        embedding_dim,
                        num_heads,
                        attn_dropout,
                        ffn_dropout,
                        num_layers_ffn,
                        moe,
                        cross_attention_architecture=self.cross_attention_architecture,
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

        # If you're using DictConfig
        # moe_cfg = MoECfg(**cfg.moe)
        # self.moe, self.n_local_experts = build_moe_from_cfg(moe_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),  # expand dimension
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(
                embedding_dim * 2, embedding_dim
            ),  # reduce back to original dimension
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_bins + 1),  # output classes
        )

        # Select which regressor to use based on configuration
        self.use_moe_regressor = use_moe_regressor
        if self.use_moe_regressor:
            self.regressor = MoERegressor(
                embedding_dim=embedding_dim,
                dropout=ffn_dropout,
                number_of_experts=number_of_experts,
                gate_temperature=1.0,
            )
        else:
            # Original simple regressor
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
        # Initialize weights and biases of all Linear layers in classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.fused_emb_proj = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(
        self,
        gene_ids,
        normalized_expr,
        input_pert_flags,
        return_encodings=False,
        return_mask_prob=True,
        return_gate_weights=False,
    ):
        """
        gene_ids: (batch, g)  # Gene ID sequence
        normalized_expr: (batch, g)  # Normalized expression values
        return_gate_weights: Whether to return MoE gate weights
        """
        gene_emb = self.gene_embedding(gene_ids)  # (batch, g, d)
        expr_emb = self.expr_embedding(normalized_expr)  # (batch, g, d)
        pert_emb = self.pert_encoder(input_pert_flags)  # (batch, g, d)
        # Is this reasonable? Directly adding???
        gene_emb = gene_emb + pert_emb
        expr_emb = expr_emb + pert_emb
        if self.attention_stream == 2:
            for i, layer in enumerate(self.layers):
                gene_emb, expr_emb = layer(gene_emb, expr_emb)
            for i, layer in enumerate(self.expression_layers):
                expr_emb = layer(expr_emb)
        elif self.attention_stream == 1:
            for i in range(self.gene_embedding_participate_til_layer):
                gene_emb = self.gene_self_attention_layers[i](gene_emb)
            for i in range(self.gene_embedding_participate_til_layer):
                expr_emb = self.cross_attention_layers[i](gene_emb, expr_emb)
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

    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
        genes=None,
    ):
        """
        Args:
            batch_data: a dictionary of input data with keys.
            include_zero_gene: whether to include zero genes.
            gene_ids: gene vocab ids.
            amp: whether to use automatic mixed precision.
            genes: list of gene names (required for new GEARS format).

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x

        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        # Construct pert_flags
        n_genes = ori_gene_values.size(1)
        if genes is None:
            raise ValueError("genes parameter is required for new GEARS format")
        name2col = {g: i for i, g in enumerate(genes)}
        pert_flags = torch.zeros(batch_size, n_genes, device=device, dtype=torch.long)
        for r, p in enumerate(batch_data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = name2col.get(g, -1)
                    if j != -1:
                        pert_flags[r, j] = 1

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # Mask out positions where mapped_input_gene_ids is 0 (unmapped genes)
            valid_gene_mask = mapped_input_gene_ids[0] != 0  # (seq_len,)
            input_values = input_values * valid_gene_mask.float().unsqueeze(
                0
            )  # broadcast to (batch_size, seq_len)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=True):
                output_diregression_output, gene_emb, expr_emb = self(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                )
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_diregression_output
        return pred_gene_values

    def get_regressor_output(self, final_emb):
        """
        Get regressor output

        Args:
            final_emb: Final fused embedding features

        Returns:
            regression_output: Regression prediction results
            gate_weights: MoE gate weights (if using MoE) or None (if using simple regressor)
        """
        # Get output based on regressor type
        if self.use_moe_regressor:
            # MoE regressor returns two values: regression_output and gate_weights
            regression_output, gate_weights = self.regressor(final_emb)
        else:
            # Original regressor only returns one value
            regression_output = self.regressor(final_emb)
            regression_output = regression_output.squeeze(-1)  # Remove last dimension
            gate_weights = None  # Original regressor has no gate_weights

        return regression_output, gate_weights

    def get_all_moe_stats(self):
        """
        Get statistics for all MoE layers in the model

        Returns:
            dict: Dictionary containing statistics for all MoE layers
        """
        moe_stats = {}

        # Check MoE in transformer layers
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "ffn_gene") and isinstance(layer.ffn_gene, MoE):
                moe_stats[f"transformer_layer_{i}_gene_ffn"] = (
                    layer.ffn_gene.get_expert_usage_stats()
                )
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                moe_stats[f"transformer_layer_{i}_expr_ffn"] = (
                    layer.ffn_expr.get_expert_usage_stats()
                )

        # Check MoE in expression layers
        for i, layer in enumerate(self.expression_layers):
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                moe_stats[f"expression_layer_{i}_expr_ffn"] = (
                    layer.ffn_expr.get_expert_usage_stats()
                )

        # Check MoE regressor
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
        Check if any MoE layers in the model have collapsed

        Args:
            threshold: Collapse threshold

        Returns:
            dict: Dictionary containing collapse detection results
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
        """Reset usage statistics for all MoE layers"""
        # Reset MoE in transformer layers
        for layer in self.layers:
            if hasattr(layer, "ffn_gene") and isinstance(layer.ffn_gene, MoE):
                layer.ffn_gene.reset_usage_stats()
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                layer.ffn_expr.reset_usage_stats()

        # Reset MoE in expression layers
        for layer in self.expression_layers:
            if hasattr(layer, "ffn_expr") and isinstance(layer.ffn_expr, MoE):
                layer.ffn_expr.reset_usage_stats()

    def print_moe_collapse_report(self, threshold=0.8):
        """
        Print MoE collapse detection report

        Args:
            threshold: Collapse threshold
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

        # Return whether there is any collapse
        return len(collapsed_layers) > 0
