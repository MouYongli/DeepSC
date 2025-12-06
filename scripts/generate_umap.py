#!/usr/bin/env python
"""
Generate UMAP visualization from pretrained DeepSC model.

Usage:
    python scripts/generate_umap.py \
        --checkpoint results/pretraining_1120/DeepSC_7_0.ckpt \
        --h5ad data/processed/baseline/scfoundation/zheng_merged.h5ad \
        --gene_map scripts/data/preprocessing/gene_map.csv \
        --output results/pretraining_1120/umap_zheng.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import umap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepsc.models.deepsc.model import DeepSC


class SimpleDataset(Dataset):
    """Simple dataset for inference without masking."""

    def __init__(
        self,
        h5ad_path: str,
        csv_path: str,
        var_name_col: str = None,
        obs_celltype_col: str = "cell_type_label",
    ):
        # 1) Read CSV for feature_name -> id mapping
        df_map = pd.read_csv(csv_path)
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) Read h5ad
        adata = sc.read_h5ad(h5ad_path)

        # Use var.index if var_name_col is not specified or not present
        if var_name_col is None or var_name_col not in adata.var.columns:
            var_names = adata.var.index.astype(str).values
        else:
            var_names = adata.var[var_name_col].astype(str).values

        # 3) Get cell type labels if available
        if obs_celltype_col in adata.obs.columns:
            celltype_cat = adata.obs[obs_celltype_col].astype("category")
            self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
            self.celltype_categories = list(celltype_cat.cat.categories)
            self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}
        else:
            self.celltype_ids = np.zeros(adata.shape[0], dtype=np.int64)
            self.celltype_categories = ["unknown"]
            self.id2type = {0: "unknown"}

        # 4) Match genes
        matched = []
        for j, nm in enumerate(var_names):
            _id = name2id.get(nm, None)
            if _id is not None:
                matched.append((_id, j))

        if len(matched) == 0:
            raise ValueError("No genes matched between h5ad and gene_map.csv")

        print(f"Matched {len(matched)} genes out of {len(var_names)}")

        # 5) Sort by id and extract expression matrix
        matched.sort(key=lambda x: x[0])
        self.sorted_ids = np.array([t[0] for t in matched], dtype=np.int64)
        cols_sorted = np.array([t[1] for t in matched], dtype=np.int64)

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        self.csr_matrix = X[:, cols_sorted]
        self.num_samples = self.csr_matrix.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        col_pos = row_coo.col.astype(np.int64)
        gene_ids_np = self.sorted_ids[col_pos]
        gene_indices = torch.from_numpy(gene_ids_np).long()
        expression_values = torch.from_numpy(row_coo.data.astype(np.float32))
        cell_type_id = int(self.celltype_ids[idx])
        return {
            "genes": gene_indices,
            "expressions": expression_values,
            "cell_type_id": torch.tensor(cell_type_id, dtype=torch.long),
        }


class InferenceCollator:
    """Collator for inference without masking."""

    def __init__(
        self,
        num_bins: int = 5,
        num_genes: int = 34683,
        max_length: int = 1024,
        pad_token_id: int = 0,
        pad_value: int = 0,
    ):
        self.num_bins = num_bins
        self.num_genes = num_genes
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.cls_token_id = num_genes + 1
        self.cls_value = num_bins + 1

    def __call__(self, examples):
        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = min(self.max_length, max_ori_len + 1)  # +1 for cls token

        padded_genes = []
        padded_discrete_expr = []
        padded_continuous_expr = []
        cell_type_ids = []

        for example in examples:
            genes = example["genes"] + 1  # gene_from_zero=True
            expressions = example["expressions"]

            # Discretize expression
            discrete_expr = self.discretize_expression(expressions)

            # Add cls token
            genes = torch.cat(
                [torch.tensor([self.cls_token_id], dtype=genes.dtype), genes]
            )
            discrete_expr = torch.cat(
                [
                    torch.tensor([self.cls_value], dtype=discrete_expr.dtype),
                    discrete_expr,
                ]
            )
            continuous_expr = torch.cat(
                [torch.tensor([self.cls_value], dtype=expressions.dtype), expressions]
            )

            # Truncate or pad
            if len(genes) > _max_length:
                # Keep top expressions (like _top_expr_or_pad)
                genes, discrete_expr, continuous_expr = self._top_expr_truncate(
                    genes, discrete_expr, continuous_expr, _max_length
                )
            elif len(genes) < _max_length:
                genes, discrete_expr, continuous_expr = self._pad(
                    genes, discrete_expr, continuous_expr, _max_length
                )

            padded_genes.append(genes)
            padded_discrete_expr.append(discrete_expr)
            padded_continuous_expr.append(continuous_expr)
            cell_type_ids.append(example["cell_type_id"])

        return {
            "gene": torch.stack(padded_genes, dim=0),
            "masked_discrete_expr": torch.stack(padded_discrete_expr, dim=0),
            "masked_continuous_expr": torch.stack(
                padded_continuous_expr, dim=0
            ).float(),
            "cell_type_id": torch.stack(cell_type_ids, dim=0),
        }

    def discretize_expression(self, normalized_expr):
        min_val = normalized_expr.min()
        max_val = normalized_expr.max()
        normalized_range = (normalized_expr - min_val) / (max_val - min_val + 1e-8)
        bin_indices = torch.floor(normalized_range * (self.num_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return bin_indices + 1

    def _top_expr_truncate(self, genes, discrete_expr, continuous_expr, max_length):
        # Keep first token (cls), sort rest by expression
        fixed_genes = genes[:1]
        fixed_discrete = discrete_expr[:1]
        fixed_continuous = continuous_expr[:1]

        rest_genes = genes[1:]
        rest_discrete = discrete_expr[1:]
        rest_continuous = continuous_expr[1:]

        sorted_indices = torch.argsort(rest_continuous, descending=True)
        needed = max_length - 1
        rest_genes = rest_genes[sorted_indices][:needed]
        rest_discrete = rest_discrete[sorted_indices][:needed]
        rest_continuous = rest_continuous[sorted_indices][:needed]

        return (
            torch.cat([fixed_genes, rest_genes]),
            torch.cat([fixed_discrete, rest_discrete]),
            torch.cat([fixed_continuous, rest_continuous]),
        )

    def _pad(self, genes, discrete_expr, continuous_expr, max_length):
        pad_len = max_length - len(genes)
        genes = torch.cat(
            [genes, torch.full((pad_len,), self.pad_token_id, dtype=genes.dtype)]
        )
        discrete_expr = torch.cat(
            [
                discrete_expr,
                torch.full((pad_len,), self.pad_value, dtype=discrete_expr.dtype),
            ]
        )
        continuous_expr = torch.cat(
            [
                continuous_expr,
                torch.full((pad_len,), self.pad_value, dtype=continuous_expr.dtype),
            ]
        )
        return genes, discrete_expr, continuous_expr


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_moe_ffn: bool = False,
    use_M_matrix: bool = False,
    alpha: float = 0.5,
) -> DeepSC:
    """Load pretrained DeepSC model."""
    # Model config (matching deepsc.yaml and experiment/default.yaml)
    from omegaconf import OmegaConf

    moe_cfg = OmegaConf.create(
        {
            "n_moe_layers": 4,
            "use_moe_ffn": use_moe_ffn,
            "dim": 256,
            "moe_inter_dim": 512,
            "n_routed_experts": 2,
            "n_activated_experts": 2,
            "n_shared_experts": 1,
            "score_func": "softmax",
            "route_scale": 1.0,
        }
    )

    model = DeepSC(
        embedding_dim=256,
        num_genes=34683,
        num_layers=10,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        fused=False,
        num_bins=5,
        alpha=alpha,
        mask_layer_start=2,
        enable_l0=True,
        enable_mse=True,
        enable_ce=True,
        num_layers_ffn=2,
        use_moe_regressor=True,
        use_M_matrix=use_M_matrix,
        gene_embedding_participate_til_layer=3,
        moe=moe_cfg,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove any prefix like "module." or "model."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_cell_embeddings(
    model: DeepSC,
    dataloader: DataLoader,
    device: torch.device,
    embedding_type: str = "cls",
) -> tuple:
    """Extract cell embeddings from model.

    Args:
        model: DeepSC model
        dataloader: DataLoader with cell data
        device: torch device
        embedding_type: "cls" for CLS token, "avg" for average pooling

    Returns:
        embeddings: numpy array of shape (num_cells, embedding_dim)
        cell_types: numpy array of cell type ids
    """
    all_embeddings = []
    all_cell_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            gene_ids = batch["gene"].to(device)
            expression_bin = batch["masked_discrete_expr"].to(device)
            normalized_expr = batch["masked_continuous_expr"].to(device)
            cell_type_ids = batch["cell_type_id"]

            # Forward pass
            outputs = model(
                gene_ids=gene_ids,
                expression_bin=expression_bin,
                normalized_expr=normalized_expr,
            )

            # outputs: logits, regression_output, y, gene_emb, expr_emb
            expr_emb = outputs[4]  # (batch, seq_len, embedding_dim)

            if embedding_type == "cls":
                # Use CLS token embedding
                cell_emb = expr_emb[:, 0, :]  # (batch, embedding_dim)
            else:
                # Use average pooling (excluding padding)
                # Create mask for non-padding positions
                mask = (gene_ids != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
                cell_emb = (expr_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            all_embeddings.append(cell_emb.cpu().numpy())
            all_cell_types.append(cell_type_ids.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    cell_types = np.concatenate(all_cell_types, axis=0)
    return embeddings, cell_types


def plot_umap(
    embeddings: np.ndarray,
    cell_types: np.ndarray,
    id2type: dict,
    output_path: str,
    title: str = "DeepSC Cell Embeddings UMAP",
):
    """Generate and save UMAP visualization."""
    print("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique cell types
    unique_types = np.unique(cell_types)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))

    for i, ct in enumerate(unique_types):
        mask = cell_types == ct
        label = id2type.get(ct, f"Type {ct}")
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i]],
            label=label,
            s=5,
            alpha=0.6,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    # Put legend outside the plot
    if len(unique_types) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"UMAP saved to {output_path}")

    # Also save without legend for cleaner view
    ax.get_legend().remove()
    output_path_no_legend = output_path.replace(".png", "_no_legend.png")
    plt.savefig(output_path_no_legend, dpi=150, bbox_inches="tight")
    print(f"UMAP (no legend) saved to {output_path_no_legend}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP from DeepSC model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--h5ad",
        type=str,
        required=True,
        help="Path to h5ad file",
    )
    parser.add_argument(
        "--gene_map",
        type=str,
        required=True,
        help="Path to gene_map.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="umap.png",
        help="Output path for UMAP plot",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="cls",
        choices=["cls", "avg"],
        help="Type of cell embedding: cls (CLS token) or avg (average pooling)",
    )
    parser.add_argument(
        "--celltype_col",
        type=str,
        default="cell_type_label",
        help="Column name for cell type in obs",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Maximum number of cells to use (for large datasets)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    # Model configuration arguments
    parser.add_argument(
        "--use_moe_ffn",
        action="store_true",
        help="Use MoE FFN layers (set for baseline model)",
    )
    parser.add_argument(
        "--use_M_matrix",
        action="store_true",
        help="Use M matrix (Gumbel-Softmax gate)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for expression embedding (0.3 for baseline, 0.5 for new)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.h5ad}...")
    dataset = SimpleDataset(
        h5ad_path=args.h5ad,
        csv_path=args.gene_map,
        obs_celltype_col=args.celltype_col,
    )

    # Subsample if needed
    if args.max_cells is not None and len(dataset) > args.max_cells:
        indices = np.random.choice(len(dataset), args.max_cells, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        # Update celltype_ids for subset
        original_celltype_ids = dataset.dataset.celltype_ids
        dataset.dataset.celltype_ids = original_celltype_ids[indices]
        print(f"Subsampled to {args.max_cells} cells")

    print(f"Dataset size: {len(dataset)} cells")

    # Create dataloader
    collator = InferenceCollator(
        num_bins=5,
        num_genes=34683,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print(
        f"Model config: use_moe_ffn={args.use_moe_ffn}, use_M_matrix={args.use_M_matrix}, alpha={args.alpha}"
    )
    model = load_model(
        args.checkpoint,
        device,
        use_moe_ffn=args.use_moe_ffn,
        use_M_matrix=args.use_M_matrix,
        alpha=args.alpha,
    )

    # Get embeddings
    embeddings, cell_types = get_cell_embeddings(
        model, dataloader, device, args.embedding_type
    )
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Get id2type mapping
    if hasattr(dataset, "id2type"):
        id2type = dataset.id2type
    elif hasattr(dataset, "dataset"):
        id2type = dataset.dataset.id2type
    else:
        id2type = {i: f"Type {i}" for i in np.unique(cell_types)}

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Plot UMAP
    plot_umap(
        embeddings,
        cell_types,
        id2type,
        args.output,
        title=f"DeepSC Cell Embeddings UMAP ({args.embedding_type})",
    )


if __name__ == "__main__":
    main()
