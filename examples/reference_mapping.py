"""
Reference Mapping Using Cell Embedding by Pretrained DeepSC Model

In this script, we demonstrate how to use the DeepSC model to embed cells and map them
to referenced embeddings. Then the meta labels of the reference cells, such as cell type
or disease conditions, can be propagated to the query cells. In this zero-shot settings,
no further training is needed.

We provide two modes of reference mapping:
1. Using a customized reference dataset with provided annotations
2. Using CellXGene atlas as reference (if index is available)

Usage:
    python examples/reference_mapping.py \
        --checkpoint results/pretraining_1120/DeepSC_7_0.ckpt \
        --ref_h5ad data/annotation_pancreas/demo_train.h5ad \
        --query_h5ad data/annotation_pancreas/demo_test.h5ad \
        --gene_map scripts/data/preprocessing/gene_map.csv \
        --cell_type_key cell_type
"""

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deepsc.models.deepsc.model import DeepSC

# Extra dependency for similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print(
        "To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss"
    )

warnings.filterwarnings("ignore", category=ResourceWarning)


# ============================================================================
# Dataset and Collator Classes
# ============================================================================


class ReferenceMappingDataset(Dataset):
    """Dataset for reference mapping inference without masking."""

    def __init__(
        self,
        h5ad_path: str = None,
        adata=None,
        csv_path: str = "",
        var_name_col: str = None,
        obs_celltype_col: str = "cell_type",
    ):
        """
        Args:
            h5ad_path: Path to h5ad file
            adata: AnnData object (alternative to h5ad_path)
            csv_path: Path to gene_map.csv with feature_name and id columns
            var_name_col: Column name for gene names in var (None uses index)
            obs_celltype_col: Column name for cell type in obs
        """
        # 1) Read CSV for feature_name -> id mapping
        df_map = pd.read_csv(csv_path)
        if not {"feature_name", "id"}.issubset(df_map.columns):
            raise ValueError("CSV must contain columns: feature_name, id")
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) Read h5ad
        if adata is not None:
            self.adata = adata
        elif h5ad_path is not None:
            self.adata = sc.read_h5ad(h5ad_path)
        else:
            raise ValueError("Either h5ad_path or adata must be provided")

        # Use var.index if var_name_col is not specified or not present
        if var_name_col is None or var_name_col not in self.adata.var.columns:
            var_names = self.adata.var.index.astype(str).values
        else:
            var_names = self.adata.var[var_name_col].astype(str).values

        # 3) Get cell type labels if available
        self.obs_celltype_col = obs_celltype_col
        if obs_celltype_col in self.adata.obs.columns:
            celltype_cat = self.adata.obs[obs_celltype_col].astype("category")
            self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
            self.celltype_categories = list(celltype_cat.cat.categories)
            self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}
            self.type2id = {name: i for i, name in enumerate(self.celltype_categories)}
            self.celltype_labels = self.adata.obs[obs_celltype_col].values
        else:
            self.celltype_ids = np.zeros(self.adata.shape[0], dtype=np.int64)
            self.celltype_categories = ["unknown"]
            self.id2type = {0: "unknown"}
            self.type2id = {"unknown": 0}
            self.celltype_labels = np.array(["unknown"] * self.adata.shape[0])

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

        X = self.adata.X
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


# ============================================================================
# Model Loading and Embedding Functions
# ============================================================================


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_moe_ffn: bool = False,
    use_M_matrix: bool = False,
    alpha: float = 0.5,
    num_bins: int = 5,
) -> DeepSC:
    """Load pretrained DeepSC model."""
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
        num_bins=num_bins,
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


def embed_data(
    model: DeepSC,
    dataset: ReferenceMappingDataset,
    device: torch.device,
    batch_size: int = 64,
    num_bins: int = 5,
    max_length: int = 1024,
    embedding_type: str = "cls",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed cells using DeepSC model.

    Args:
        model: DeepSC model
        dataset: ReferenceMappingDataset
        device: torch device
        batch_size: Batch size for inference
        num_bins: Number of bins for discretization
        max_length: Maximum sequence length
        embedding_type: "cls" for CLS token, "avg" for average pooling

    Returns:
        embeddings: numpy array of shape (num_cells, embedding_dim)
        cell_type_ids: numpy array of cell type ids
        cell_type_labels: numpy array of cell type string labels
    """
    collator = InferenceCollator(
        num_bins=num_bins,
        num_genes=34683,
        max_length=max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

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
                mask = (gene_ids != 0).float().unsqueeze(-1)
                cell_emb = (expr_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            all_embeddings.append(cell_emb.cpu().numpy())
            all_cell_types.append(cell_type_ids.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    cell_type_ids = np.concatenate(all_cell_types, axis=0)
    cell_type_labels = dataset.celltype_labels

    return embeddings, cell_type_ids, cell_type_labels


# ============================================================================
# Similarity Search Functions
# ============================================================================


def l2_sim(a, b):
    """Calculate L2 similarity (used when faiss is not installed)"""
    sims = -np.linalg.norm(a - b, axis=1)
    return sims


def get_similar_vectors(vector, ref, top_k=10):
    """Get similar vectors (used when faiss is not installed)"""
    sims = l2_sim(vector, ref)
    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


# ============================================================================
# Reference Mapping Functions
# ============================================================================


def run_customized_reference_mapping(
    checkpoint_path: str,
    ref_h5ad_path: str,
    query_h5ad_path: str,
    gene_map_path: str,
    cell_type_key: str = "cell_type",
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 1024,
    k: int = 10,
    use_moe_ffn: bool = False,
    use_M_matrix: bool = False,
    alpha: float = 0.5,
    num_bins: int = 5,
    embedding_type: str = "cls",
    visualize: bool = True,
    output_dir: str = None,
) -> float:
    """
    Run reference mapping using a customized reference dataset.

    Args:
        checkpoint_path: Path to DeepSC checkpoint
        ref_h5ad_path: Path to reference h5ad file
        query_h5ad_path: Path to query h5ad file
        gene_map_path: Path to gene_map.csv
        cell_type_key: Column name for cell type in obs
        device: Device to use
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        k: Number of neighbors for KNN
        use_moe_ffn: Use MoE FFN layers
        use_M_matrix: Use M matrix (Gumbel-Softmax gate)
        alpha: Alpha parameter for expression embedding
        num_bins: Number of bins for discretization
        embedding_type: "cls" or "avg"
        visualize: Whether to generate UMAP visualization
        output_dir: Directory to save outputs

    Returns:
        accuracy: Classification accuracy
    """
    print("=" * 60)
    print("Reference mapping using a customized reference dataset")
    print("=" * 60)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(
        checkpoint_path,
        device,
        use_moe_ffn=use_moe_ffn,
        use_M_matrix=use_M_matrix,
        alpha=alpha,
        num_bins=num_bins,
    )

    # Load and embed reference data
    print(f"Loading reference data from {ref_h5ad_path}...")
    ref_dataset = ReferenceMappingDataset(
        h5ad_path=ref_h5ad_path,
        csv_path=gene_map_path,
        obs_celltype_col=cell_type_key,
    )

    print("Embedding reference data...")
    ref_embeddings, ref_cell_type_ids, ref_cell_type_labels = embed_data(
        model, ref_dataset, device, batch_size, num_bins, max_length, embedding_type
    )
    print(f"Reference embeddings shape: {ref_embeddings.shape}")

    # Load and embed query data
    print(f"Loading query data from {query_h5ad_path}...")
    query_dataset = ReferenceMappingDataset(
        h5ad_path=query_h5ad_path,
        csv_path=gene_map_path,
        obs_celltype_col=cell_type_key,
    )

    print("Embedding query data...")
    query_embeddings, query_cell_type_ids, query_cell_type_labels = embed_data(
        model, query_dataset, device, batch_size, num_bins, max_length, embedding_type
    )
    print(f"Query embeddings shape: {query_embeddings.shape}")

    # Reference mapping using KNN
    print(f"Running reference mapping with k={k}...")

    if FAISS_AVAILABLE:
        # Use faiss for fast similarity search
        index = faiss.IndexFlatL2(ref_embeddings.shape[1])
        index.add(ref_embeddings.astype(np.float32))
        distances, labels = index.search(query_embeddings.astype(np.float32), k)
    else:
        labels = np.zeros((query_embeddings.shape[0], k), dtype=np.int64)
        for i in range(query_embeddings.shape[0]):
            idx, _ = get_similar_vectors(query_embeddings[i : i + 1], ref_embeddings, k)
            labels[i] = idx

    # Vote for predicted labels
    preds = []
    for k_idx in range(query_embeddings.shape[0]):
        idx = labels[k_idx]
        # Get the most common cell type among k neighbors
        neighbor_labels = ref_cell_type_labels[idx]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        pred = unique[np.argmax(counts)]
        preds.append(pred)

    preds = np.array(preds)
    gt = query_cell_type_labels

    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(gt, preds)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate per-class metrics
    print("\nPer-class metrics:")
    unique_classes = np.unique(np.concatenate([gt, preds]))
    for cls in unique_classes:
        cls_mask = gt == cls
        if cls_mask.sum() > 0:
            cls_acc = (preds[cls_mask] == cls).sum() / cls_mask.sum()
            print(f"  {cls}: {cls_acc:.4f} ({cls_mask.sum()} samples)")

    # Visualize if requested
    if visualize and output_dir is not None:
        try:
            import matplotlib.pyplot as plt
            import umap

            os.makedirs(output_dir, exist_ok=True)

            # Combine embeddings for joint UMAP
            combined_embeddings = np.concatenate(
                [ref_embeddings, query_embeddings], axis=0
            )
            combined_labels = np.concatenate(
                [
                    np.array(["ref_" + str(label) for label in ref_cell_type_labels]),
                    np.array(
                        ["query_" + str(label) for label in query_cell_type_labels]
                    ),
                ]
            )

            print("Computing UMAP...")
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=30, min_dist=0.3
            )
            embedding_2d = reducer.fit_transform(combined_embeddings)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            unique_labels = np.unique(combined_labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = combined_labels == label
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
            ax.set_title(f"DeepSC Reference Mapping UMAP (Accuracy: {accuracy:.4f})")

            if len(unique_labels) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)

            plt.tight_layout()
            output_path = os.path.join(output_dir, "reference_mapping_umap.png")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"UMAP saved to {output_path}")
            plt.close()

        except ImportError:
            print("umap-learn not installed, skipping visualization")

    return accuracy


def run_embed_data_task(
    checkpoint_path: str,
    h5ad_path: str,
    gene_map_path: str,
    output_path: str = None,
    cell_type_key: str = "cell_type",
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 1024,
    use_moe_ffn: bool = False,
    use_M_matrix: bool = False,
    alpha: float = 0.5,
    num_bins: int = 5,
    embedding_type: str = "cls",
) -> sc.AnnData:
    """
    Embed data and return as AnnData with embeddings in X.

    This is analogous to scgpt.tasks.embed_data

    Args:
        checkpoint_path: Path to DeepSC checkpoint
        h5ad_path: Path to h5ad file
        gene_map_path: Path to gene_map.csv
        output_path: Optional path to save output h5ad
        cell_type_key: Column name for cell type in obs
        device: Device to use
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        use_moe_ffn: Use MoE FFN layers
        use_M_matrix: Use M matrix
        alpha: Alpha parameter
        num_bins: Number of bins
        embedding_type: "cls" or "avg"

    Returns:
        adata: AnnData with embeddings in X
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        checkpoint_path,
        device,
        use_moe_ffn=use_moe_ffn,
        use_M_matrix=use_M_matrix,
        alpha=alpha,
        num_bins=num_bins,
    )

    # Load dataset
    dataset = ReferenceMappingDataset(
        h5ad_path=h5ad_path,
        csv_path=gene_map_path,
        obs_celltype_col=cell_type_key,
    )

    # Get embeddings
    embeddings, cell_type_ids, cell_type_labels = embed_data(
        model, dataset, device, batch_size, num_bins, max_length, embedding_type
    )

    # Create AnnData with embeddings
    adata = sc.AnnData(X=embeddings)
    adata.obs[cell_type_key] = cell_type_labels
    adata.obs["cell_type_id"] = cell_type_ids

    if output_path is not None:
        adata.write_h5ad(output_path)
        print(f"Saved embeddings to {output_path}")

    return adata


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Reference Mapping using DeepSC model")

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DeepSC checkpoint",
    )
    parser.add_argument(
        "--gene_map",
        type=str,
        required=True,
        help="Path to gene_map.csv",
    )

    # Data arguments
    parser.add_argument(
        "--ref_h5ad",
        type=str,
        default=None,
        help="Path to reference h5ad file",
    )
    parser.add_argument(
        "--query_h5ad",
        type=str,
        default=None,
        help="Path to query h5ad file",
    )
    parser.add_argument(
        "--h5ad",
        type=str,
        default=None,
        help="Path to h5ad file (for embed_data task)",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        default="cell_type",
        help="Column name for cell type in obs",
    )

    # Model configuration
    parser.add_argument(
        "--use_moe_ffn",
        action="store_true",
        help="Use MoE FFN layers",
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
        help="Alpha parameter for expression embedding",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Number of bins for discretization",
    )

    # Inference configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
        help="Type of cell embedding",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors for KNN",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--output_h5ad",
        type=str,
        default=None,
        help="Path to save embedded h5ad (for embed_data task)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate UMAP visualization",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="reference_mapping",
        choices=["reference_mapping", "embed_data"],
        help="Task to run",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DeepSC Reference Mapping / Embedding")
    print("=" * 60 + "\n")

    if args.task == "reference_mapping":
        if args.ref_h5ad is None or args.query_h5ad is None:
            raise ValueError(
                "--ref_h5ad and --query_h5ad are required for reference_mapping task"
            )

        accuracy = run_customized_reference_mapping(
            checkpoint_path=args.checkpoint,
            ref_h5ad_path=args.ref_h5ad,
            query_h5ad_path=args.query_h5ad,
            gene_map_path=args.gene_map,
            cell_type_key=args.cell_type_key,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            k=args.k,
            use_moe_ffn=args.use_moe_ffn,
            use_M_matrix=args.use_M_matrix,
            alpha=args.alpha,
            num_bins=args.num_bins,
            embedding_type=args.embedding_type,
            visualize=args.visualize,
            output_dir=args.output_dir,
        )
        print(f"\nReference mapping completed with accuracy: {accuracy:.4f}")

    elif args.task == "embed_data":
        if args.h5ad is None:
            raise ValueError("--h5ad is required for embed_data task")

        adata = run_embed_data_task(
            checkpoint_path=args.checkpoint,
            h5ad_path=args.h5ad,
            gene_map_path=args.gene_map,
            output_path=args.output_h5ad,
            cell_type_key=args.cell_type_key,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_moe_ffn=args.use_moe_ffn,
            use_M_matrix=args.use_M_matrix,
            alpha=args.alpha,
            num_bins=args.num_bins,
            embedding_type=args.embedding_type,
        )
        print(f"\nEmbedding completed. Shape: {adata.shape}")


if __name__ == "__main__":
    main()
