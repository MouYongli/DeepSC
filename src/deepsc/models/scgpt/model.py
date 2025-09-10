"""
This file is only used for test hydra config. Not appropriate for pretraining.
"""

import warnings
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

try:
    from flash_attn.modules.mha import MHA

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

    # Create dummy classes to prevent errors
    class MHA:
        pass

    class MLP:
        pass


class FlashTransformerEncoderLayer(nn.Module):
    """Flash Attention Transformer Encoder Layer for efficient attention computation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        # Use flash attention if available, otherwise use standard multi-head attention
        if FLASH_ATTN_AVAILABLE and use_flash_attn:
            self.self_attn = MHA(
                d_model,
                num_heads=nhead,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
                causal=False,  # For encoder, we use bidirectional attention
            )
            self.use_flash_attn = True
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
            )
            self.use_flash_attn = False

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = getattr(F, activation)

        self.norm_first = norm_first

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            if self.use_flash_attn:
                src2, _ = self.self_attn(
                    self.norm1(src), key_padding_mask=src_key_padding_mask
                )
            else:
                src2, _ = self.self_attn(
                    self.norm1(src),
                    self.norm1(src),
                    self.norm1(src),
                    key_padding_mask=src_key_padding_mask,
                    attn_mask=src_mask,
                )
            src = src + self.dropout1(src2)
            src2 = self.linear2(
                self.dropout(self.activation(self.linear1(self.norm2(src))))
            )
            src = src + self.dropout2(src2)
        else:
            if self.use_flash_attn:
                src2, _ = self.self_attn(src, key_padding_mask=src_key_padding_mask)
            else:
                src2, _ = self.self_attn(
                    src,
                    src,
                    src,
                    key_padding_mask=src_key_padding_mask,
                    attn_mask=src_mask,
                )
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        return src


class GeneEmbedding(nn.Module):
    """Gene embedding layer with continuous and categorical embeddings."""

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        vocab: Optional[Dict] = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        explicit_zero_prob: bool = False,
    ):
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model
        self.input_emb_style = input_emb_style
        self.explicit_zero_prob = explicit_zero_prob

        # Gene embedding
        self.gene_encoder = nn.Embedding(ntoken, d_model, padding_idx=pad_value)

        # Expression value embedding
        if input_emb_style == "continuous":
            self.value_encoder = nn.Sequential(
                nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif input_emb_style == "binned":
            assert n_input_bins is not None
            self.value_encoder = nn.Embedding(n_input_bins, d_model)
        else:
            raise ValueError(f"input_emb_style {input_emb_style} is not supported")

        # Position embedding
        self.pos_encoder = nn.Embedding(ntoken, d_model)

        # Zero probability embedding if needed
        if explicit_zero_prob:
            self.zero_prob_encoder = nn.Linear(1, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        zero_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            gene_ids: [batch_size, seq_len]
            values: [batch_size, seq_len] or [batch_size, seq_len, 1]
            zero_probs: [batch_size, seq_len] optional zero probabilities
        """
        batch_size, seq_len = gene_ids.shape

        # Gene embeddings
        gene_embs = self.gene_encoder(gene_ids)  # [B, L, D]

        # Position embeddings
        positions = (
            torch.arange(seq_len, device=gene_ids.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_embs = self.pos_encoder(positions)  # [B, L, D]

        # Value embeddings
        if self.input_emb_style == "continuous":
            if values.dim() == 2:
                values = values.unsqueeze(-1)  # [B, L, 1]
            value_embs = self.value_encoder(values)  # [B, L, D]
        else:  # binned
            value_embs = self.value_encoder(values.long())  # [B, L, D]

        # Combine embeddings
        embeddings = gene_embs + value_embs + pos_embs

        # Add zero probability embeddings if needed
        if self.explicit_zero_prob and zero_probs is not None:
            zero_prob_embs = self.zero_prob_encoder(zero_probs.unsqueeze(-1))
            embeddings = embeddings + zero_prob_embs

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class CellTypeClassifier(nn.Module):
    """Multi-layer classifier for cell type prediction."""

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers_cls: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        layers = []
        input_dim = d_model

        for i in range(nlayers_cls):
            if i == nlayers_cls - 1:  # Last layer
                layers.append(nn.Linear(input_dim, n_cls))
            else:
                layers.extend(
                    [
                        nn.Linear(input_dim, d_model),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                input_dim = d_model

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TransformerModel(nn.Module):
    """
    scGPT: A Large-scale Generative Pre-trained Transformer for Single Cell Analysis

    This is a transformer-based model for single-cell genomics data analysis,
    supporting various tasks including cell type classification, gene expression
    modeling, and batch correction.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Optional[Dict] = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: bool = False,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ):
        super().__init__()

        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.n_cls = n_cls
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.cell_emb_style = cell_emb_style
        self.mvc_decoder_style = mvc_decoder_style
        self.ecs_threshold = ecs_threshold
        self.use_fast_transformer = use_fast_transformer
        self.pad_value = pad_value

        # Gene and expression embeddings
        self.encoder = GeneEmbedding(
            ntoken=ntoken,
            d_model=d_model,
            vocab=vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            input_emb_style=input_emb_style,
            n_input_bins=n_input_bins,
            explicit_zero_prob=explicit_zero_prob,
        )

        # CLS token for cell representation
        if cell_emb_style == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        if (
            use_fast_transformer
            and fast_transformer_backend == "flash"
            and FLASH_ATTN_AVAILABLE
        ):
            encoder_layers = nn.ModuleList(
                [
                    FlashTransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_hid,
                        dropout=dropout,
                        norm_first=pre_norm,
                        use_flash_attn=True,
                    )
                    for _ in range(nlayers)
                ]
            )
            self.transformer_encoder = nn.Sequential(*encoder_layers)
        elif use_fast_transformer:
            # If flash attention is not available but fast transformer is requested,
            # use our custom FlashTransformerEncoderLayer with standard attention
            encoder_layers = nn.ModuleList(
                [
                    FlashTransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_hid,
                        dropout=dropout,
                        norm_first=pre_norm,
                        use_flash_attn=False,
                    )
                    for _ in range(nlayers)
                ]
            )
            self.transformer_encoder = nn.Sequential(*encoder_layers)
            if not FLASH_ATTN_AVAILABLE:
                warnings.warn(
                    "Flash attention not available, using standard multi-head attention"
                )
                self.use_fast_transformer = False

        if not use_fast_transformer:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_hid,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=pre_norm,
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        # Task-specific heads
        if n_cls > 1:
            self.cls_classifier = CellTypeClassifier(
                d_model=d_model,
                n_cls=n_cls,
                nlayers_cls=nlayers_cls,
                dropout=dropout,
            )

        # Masked Value Prediction (MVP) head
        if do_mvc:
            if mvc_decoder_style == "inner product":
                self.mvc_decoder = nn.Linear(d_model, ntoken, bias=False)
            else:
                self.mvc_decoder = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, ntoken),
                )

        # Domain Adversarial Training head
        if do_dab and num_batch_labels:
            self.batch_classifier = CellTypeClassifier(
                d_model=d_model,
                n_cls=num_batch_labels,
                nlayers_cls=nlayers_cls,
                dropout=dropout,
            )

        # Batch normalization layers
        if domain_spec_batchnorm and num_batch_labels:
            self.dsbn_layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            str(i): nn.BatchNorm1d(d_model)
                            for i in range(num_batch_labels)
                        }
                    )
                    for _ in range(nlayers)
                ]
            )

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.1

        # Initialize embeddings
        if hasattr(self.encoder.gene_encoder, "weight"):
            self.encoder.gene_encoder.weight.data.uniform_(-initrange, initrange)
        if hasattr(self.encoder.pos_encoder, "weight"):
            self.encoder.pos_encoder.weight.data.uniform_(-initrange, initrange)

        # Initialize CLS token
        if hasattr(self, "cls_token"):
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _get_cell_emb_from_layer(
        self, layer_output: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract cell embeddings from layer output.

        Args:
            layer_output: [batch_size, seq_len, d_model]
            weights: optional attention weights [batch_size, seq_len]
        """
        if self.cell_emb_style == "cls":
            return layer_output[:, 0]  # First token is CLS token
        elif self.cell_emb_style == "avg-pool":
            if weights is not None:
                return torch.sum(layer_output * weights.unsqueeze(-1), dim=1)
            else:
                return torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                weights = torch.ones(layer_output.shape[:2], device=layer_output.device)
            normed_weights = F.softmax(weights, dim=-1)
            return torch.sum(layer_output * normed_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        batch_labels: Optional[torch.Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
        generative_training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gene_ids: [batch_size, seq_len] gene token ids
            values: [batch_size, seq_len] expression values
            src_key_padding_mask: [batch_size, seq_len] padding mask
            batch_labels: [batch_size] batch labels for domain adaptation
            CLS: whether to perform cell type classification
            CCE: whether to perform cell-cell embedding
            MVC: whether to perform masked value prediction
            ECS: whether to perform elastic cell similarity
            do_sample: whether to sample during generation
            generative_training: whether in generative training mode
        """
        batch_size, seq_len = gene_ids.shape
        device = gene_ids.device

        # Embedding
        embeddings = self.encoder(gene_ids, values)  # [B, L, D]

        # Add CLS token if needed
        if self.cell_emb_style == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # [B, L+1, D]

            # Update padding mask for CLS token
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
                src_key_padding_mask = torch.cat(
                    [cls_mask, src_key_padding_mask], dim=1
                )

        # Transformer encoding
        if self.use_fast_transformer:
            transformer_output = embeddings
            for layer in self.transformer_encoder:
                transformer_output = layer(
                    transformer_output, src_key_padding_mask=src_key_padding_mask
                )
        else:
            transformer_output = self.transformer_encoder(
                embeddings, src_key_padding_mask=src_key_padding_mask
            )

        output_dict = {}

        # Cell embedding for classification or other tasks
        if CLS or CCE:
            cell_emb = self._get_cell_emb_from_layer(transformer_output)
            output_dict["cell_emb"] = cell_emb

            if CLS and self.n_cls > 1:
                logits = self.cls_classifier(cell_emb)
                output_dict["cls_output"] = logits

        # Masked Value Prediction
        if MVC and self.do_mvc:
            if self.cell_emb_style == "cls":
                mvc_output = transformer_output[:, 1:]  # Remove CLS token
            else:
                mvc_output = transformer_output

            if self.mvc_decoder_style == "inner product":
                # Use gene embeddings for inner product decoding
                gene_embs = self.encoder.gene_encoder.weight  # [ntoken, d_model]
                mvc_logits = torch.matmul(mvc_output, gene_embs.T)  # [B, L, ntoken]
            else:
                mvc_logits = self.mvc_decoder(mvc_output)  # [B, L, ntoken]

            output_dict["mvc_output"] = mvc_logits

        # Domain Adversarial Training
        if self.do_dab and batch_labels is not None:
            cell_emb = self._get_cell_emb_from_layer(transformer_output)
            dab_output = self.batch_classifier(cell_emb)
            output_dict["dab_output"] = dab_output

        # Return all layer outputs for further processing
        output_dict["encoder_output"] = transformer_output

        return output_dict

    def encode_batch(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        batch_labels: Optional[torch.Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode a batch of cells and return cell embeddings.

        This method is optimized for inference and large-scale embedding extraction.
        """
        self.eval()

        if batch_size is None:
            batch_size = gene_ids.shape[0]

        with torch.no_grad():
            output_dict = self.forward(
                gene_ids=gene_ids,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels,
                CCE=True,
            )

            cell_embs = output_dict["cell_emb"]

            if output_to_cpu:
                cell_embs = cell_embs.cpu()

            if return_np:
                cell_embs = cell_embs.numpy()

        return cell_embs

    def generate(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        max_length: int,
        do_sample: bool = True,
        top_k: int = None,
        top_p: float = None,
        temperature: float = 1.0,
        pad_token_id: int = None,
    ) -> torch.Tensor:
        """
        Generate gene expressions for autoregressive generation.

        Args:
            gene_ids: [batch_size, seq_len] input gene ids
            values: [batch_size, seq_len] input expression values
            max_length: maximum generation length
            do_sample: whether to use sampling
            top_k: top-k sampling parameter
            top_p: top-p sampling parameter
            temperature: sampling temperature
            pad_token_id: padding token id
        """
        self.eval()
        batch_size, seq_len = gene_ids.shape
        device = gene_ids.device

        if pad_token_id is None:
            pad_token_id = self.pad_value

        # Start with input sequence
        generated_ids = gene_ids.clone()
        generated_values = values.clone()

        with torch.no_grad():
            for _ in range(max_length - seq_len):
                # Forward pass
                outputs = self.forward(
                    gene_ids=generated_ids,
                    values=generated_values,
                    MVC=True,
                )

                # Get predictions for next token
                next_token_logits = outputs["mvc_output"][:, -1, :]  # [B, ntoken]

                if do_sample:
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    if top_k is not None:
                        top_k = min(top_k, next_token_logits.size(-1))
                        values_to_remove = (
                            next_token_logits
                            < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        )
                        next_token_logits[values_to_remove] = float("-inf")

                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(
                            next_token_logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float("-inf")

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Generate corresponding expression value (could be learned or sampled)
                next_value = (
                    torch.randn(batch_size, 1, device=device) * 0.1
                )  # Placeholder

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                generated_values = torch.cat([generated_values, next_value], dim=1)

                # Check for early stopping
                if torch.all(next_token == pad_token_id):
                    break

        return generated_ids, generated_values
