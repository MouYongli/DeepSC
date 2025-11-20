# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
# Dataset class for cell type annotation fine-tuning with scFoundation

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional, Tuple
import warnings
import sys
from load import main_gene_selection

class CellTypeDataset(Dataset):
    """
    Dataset for cell type annotation with scFoundation
    Properly aligns genes to scFoundation's 19,264 gene vocabulary

    Args:
        adata: AnnData object with gene expression data
        label_key: Key in adata.obs containing cell type labels
        gene_list_path: Path to scFoundation's gene vocabulary file
        split_key: Optional key in adata.obs for train/val/test split
        split_value: Value to filter by split_key (e.g., 'train', 'val', 'test')
    """

    def __init__(
        self,
        adata,
        label_key: str = 'cell_type',
        gene_list_path: str = './OS_scRNA_gene_index.19264.tsv',
        split_key: Optional[str] = None,
        split_value: Optional[str] = None,
        global_label_mapping: Optional[dict] = None,  # NEW: accept global label mapping
    ):
        super().__init__()

        # Filter by split if specified
        if split_key is not None and split_value is not None:
            if split_key not in adata.obs.columns:
                raise ValueError(f"Split key '{split_key}' not found in adata.obs")
            adata = adata[adata.obs[split_key] == split_value].copy()
            print(f"Filtered to split '{split_value}': {adata.n_obs} cells")

        self.adata = adata
        self.label_key = label_key

        # Load scFoundation's gene vocabulary (19,264 genes in fixed order)
        print(f"Loading gene vocabulary from {gene_list_path}")
        gene_list_df = pd.read_csv(gene_list_path, sep='\t', header=0)
        self.gene_list = list(gene_list_df['gene_name'])
        assert len(self.gene_list) == 19264, f"Expected 19264 genes, got {len(self.gene_list)}"
        print(f"Loaded {len(self.gene_list)} genes from vocabulary")

        # Get and validate cell type labels
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs. "
                           f"Available keys: {list(adata.obs.columns)}")

        self.labels = adata.obs[label_key].values

        # Use global label mapping if provided, otherwise create from current split
        if global_label_mapping is not None:
            # Use the provided global mapping
            self.label_to_idx = global_label_mapping
            self.unique_labels = sorted(global_label_mapping.keys())
            self.idx_to_label = {int(idx): int(label) if isinstance(label, (np.integer, int)) else str(label)
                                for label, idx in global_label_mapping.items()}
            self.num_classes = len(self.unique_labels)

            # Verify that all labels in this split exist in global mapping
            labels_in_split = set(self.labels)
            labels_not_in_mapping = labels_in_split - set(self.unique_labels)
            if labels_not_in_mapping:
                raise ValueError(f"Labels in this split not found in global mapping: {labels_not_in_mapping}")

            print(f"Using global label mapping with {self.num_classes} cell types")
            labels_in_this_split = sorted(list(set(self.labels)))
            print(f"This split contains {len(labels_in_this_split)} cell types: {labels_in_this_split}")
        else:
            # Create mapping from current split only (old behavior)
            self.unique_labels = sorted(list(set(self.labels)))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.idx_to_label = {int(idx): int(label) if isinstance(label, (np.integer, int)) else str(label)
                                for label, idx in self.label_to_idx.items()}
            self.num_classes = len(self.unique_labels)
            print(f"Found {self.num_classes} cell types: {self.unique_labels}")

        # Align genes to scFoundation's vocabulary using the official function
        self.aligned_data, self.missing_genes = self._align_genes_to_vocabulary()

        print(f"Dataset initialized: {len(self)} cells x {self.aligned_data.shape[1]} genes")

    def _align_genes_to_vocabulary(self) -> Tuple[np.ndarray, list]:
        """
        Align input genes to scFoundation's 19,264 gene vocabulary
        Uses the same logic as get_embedding.py
        """
        # Get gene names from adata.var
        try:
            # Try common column names for gene symbols
            if 'gene_name' in self.adata.var.columns:
                input_genes = self.adata.var['gene_name'].tolist()
            elif 'gene_symbol' in self.adata.var.columns:
                input_genes = self.adata.var['gene_symbol'].tolist()
            else:
                # Use var_names (index)
                input_genes = self.adata.var_names.tolist()
        except:
            input_genes = self.adata.var_names.tolist()

        print(f"Input data has {len(input_genes)} genes")

        # Convert to DataFrame (as expected by main_gene_selection)
        if hasattr(self.adata.X, 'toarray'):
            # Sparse matrix
            X_array = self.adata.X.toarray()
        else:
            # Dense matrix
            X_array = self.adata.X

        X_df = pd.DataFrame(
            X_array,
            index=self.adata.obs_names,
            columns=input_genes
        )

        # Use scFoundation's official gene selection function
        print("Aligning genes to scFoundation vocabulary...")
        X_aligned, to_fill_columns = main_gene_selection(X_df, self.gene_list)

        # Verify alignment
        assert X_aligned.shape[1] == 19264, \
            f"Alignment failed: expected 19264 genes, got {X_aligned.shape[1]}"
        assert list(X_aligned.columns) == self.gene_list, \
            "Gene order doesn't match vocabulary!"

        # Calculate correct matching statistics
        # to_fill_columns = genes in vocabulary but not in input data (need padding)
        # matched genes = genes in vocabulary that are also in input data
        matched_genes = len(self.gene_list) - len(to_fill_columns)
        match_rate = matched_genes / len(self.gene_list) * 100

        # Calculate how many input genes are not in vocabulary (will be discarded)
        discarded_genes = len(input_genes) - matched_genes

        print(f"Gene alignment complete:")
        print(f"  - Input genes: {len(input_genes)}")
        print(f"  - Matched: {matched_genes}/{len(self.gene_list)} genes ({match_rate:.2f}%)")
        print(f"  - Discarded (not in vocabulary): {discarded_genes} genes")
        print(f"  - Zero-padded (in vocabulary but not in data): {len(to_fill_columns)} genes")

        if match_rate < 50:
            warnings.warn(
                f"Low gene match rate ({match_rate:.2f}%). "
                f"This may affect model performance. "
                f"Check if gene names use the correct format (gene symbols like 'TP53', 'GAPDH')."
            )

        return X_aligned.values.astype(np.float32), to_fill_columns

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - x: (19264,) gene expression vector
                - label: integer label index
                - label_str: original label string
        """
        # Get aligned gene expression (19264 genes in correct order)
        x = torch.from_numpy(self.aligned_data[idx]).float()

        # Get label
        label_str = self.labels[idx]
        label_idx = self.label_to_idx[label_str]

        return {
            'x': x,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'label_str': label_str
        }


def create_data_splits(
    h5ad_path: str,
    label_key: str = 'cell_type',
    gene_list_path: str = './OS_scRNA_gene_index.19264.tsv',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_key: Optional[str] = None,
    random_seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Load h5ad file and create train/val/test dataloaders

    Args:
        h5ad_path: Path to h5ad file
        label_key: Key in adata.obs containing cell type labels
        gene_list_path: Path to scFoundation's gene vocabulary
        train_ratio: Proportion of training data (if creating new split)
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        split_key: If provided, use existing split column in adata.obs
        random_seed: Random seed for reproducibility
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader, num_classes, label_mapping
    """

    # Load data
    print(f"\n{'='*60}")
    print(f"Loading data from: {h5ad_path}")
    print(f"{'='*60}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes\n")

    # Check if split already exists
    if split_key is not None and split_key in adata.obs.columns:
        print(f"Using existing split from column '{split_key}'")
        split_values = adata.obs[split_key].unique()
        print(f"Split values found: {split_values}")

        # Check for standard split names
        has_train = 'train' in adata.obs[split_key].values
        has_val = 'val' in adata.obs[split_key].values or 'validation' in adata.obs[split_key].values
        has_test = 'test' in adata.obs[split_key].values

        if not (has_train and has_val and has_test):
            print(f"Warning: Standard split names (train/val/test) not all found")
            print(f"Creating new split instead...")
            split_key = None

    if split_key is None:
        # Create random split
        print(f"Creating random split: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        np.random.seed(random_seed)
        n_cells = adata.n_obs
        indices = np.random.permutation(n_cells)

        train_end = int(n_cells * train_ratio)
        val_end = train_end + int(n_cells * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Add split column to adata
        adata.obs['split'] = 'train'
        adata.obs.iloc[val_indices, adata.obs.columns.get_loc('split')] = 'val'
        adata.obs.iloc[test_indices, adata.obs.columns.get_loc('split')] = 'test'

        split_key = 'split'

    # Determine split values
    val_name = 'validation' if 'validation' in adata.obs[split_key].values else 'val'

    # Create GLOBAL label mapping BEFORE splitting
    # This ensures all splits use the same label-to-index mapping
    print(f"\n{'='*60}")
    print("Creating global label mapping...")
    print(f"{'='*60}\n")

    all_labels = adata.obs[label_key].values
    unique_labels = sorted(list(set(all_labels)))
    global_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"Global label mapping created for {len(unique_labels)} cell types:")
    for label, idx in sorted(global_label_mapping.items(), key=lambda x: x[1]):
        n_cells = np.sum(all_labels == label)
        print(f"  {idx}: {label} ({n_cells} cells)")

    # Create datasets
    print(f"\n{'='*60}")
    print("Creating datasets...")
    print(f"{'='*60}\n")

    train_dataset = CellTypeDataset(adata, label_key, gene_list_path, split_key, 'train',
                                    global_label_mapping=global_label_mapping)
    val_dataset = CellTypeDataset(adata, label_key, gene_list_path, split_key, val_name,
                                  global_label_mapping=global_label_mapping)
    test_dataset = CellTypeDataset(adata, label_key, gene_list_path, split_key, 'test',
                                   global_label_mapping=global_label_mapping)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm error with batch_size=1
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all validation data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all test data
    )

    num_classes = train_dataset.num_classes
    label_mapping = train_dataset.idx_to_label

    print(f"\n{'='*60}")
    print("Dataset Summary:")
    print(f"{'='*60}")
    print(f"Train: {len(train_dataset):>6} cells")
    print(f"Val:   {len(val_dataset):>6} cells")
    print(f"Test:  {len(test_dataset):>6} cells")
    print(f"{'='*60}")
    print(f"Total classes: {num_classes}")
    print(f"Class names: {list(label_mapping.values())}")
    print(f"{'='*60}\n")

    return train_loader, val_loader, test_loader, num_classes, label_mapping


if __name__ == '__main__':
    """
    Test the dataset loading

    Usage:
        python celltype_dataset.py --data your_data.h5ad --label_key cell_type
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test cell type dataset loading')
    parser.add_argument('--data', type=str, required=True, help='Path to h5ad file')
    parser.add_argument('--label_key', type=str, default='cell_type',
                       help='Key in adata.obs for cell type labels')
    parser.add_argument('--gene_list', type=str, default='./OS_scRNA_gene_index.19264.tsv',
                       help='Path to gene vocabulary file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')

    args = parser.parse_args()

    # Test dataset loading
    print("Testing dataset loading...\n")

    train_loader, val_loader, test_loader, num_classes, label_mapping = create_data_splits(
        h5ad_path=args.data,
        label_key=args.label_key,
        gene_list_path=args.gene_list,
        batch_size=args.batch_size,
        num_workers=0  # Use 0 for testing
    )

    # Test one batch
    print("Testing data loading from train set...")
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  x shape: {batch['x'].shape}")  # Should be (batch_size, 19264)
    print(f"  label shape: {batch['label'].shape}")  # Should be (batch_size,)
    print(f"  Non-zero genes per cell: {(batch['x'] > 0).sum(dim=1).float().mean():.1f}")
    print(f"  Label range: {batch['label'].min()}-{batch['label'].max()}")
    print(f"  Example labels: {[label_mapping[idx.item()] for idx in batch['label'][:3]]}")

    print("\n✅ Dataset loading test passed!")
