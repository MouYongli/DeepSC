"""
Cell Type Annotation (CTA) loss function utilities.

This module provides custom loss functions specifically designed for
cell type annotation tasks.
"""

import torch
import torch.nn as nn


def calculate_per_class_ce_loss(logits, labels, num_classes, ignore_index=-100):
    """
    Calculate cross-entropy loss for each class (cell type) separately.

    This function computes the CE loss for each cell type independently,
    which is useful for analyzing cell-type-specific performance and
    handling class imbalance in cell type annotation tasks.

    Args:
        logits: Model output logits, shape (batch, num_classes)
        labels: Ground truth cell type labels, shape (batch,)
        num_classes: Number of cell types
        ignore_index: Label value to ignore in loss calculation (default: -100)

    Returns:
        torch.Tensor: Per-class CE losses, shape (num_classes,)
            Each element is the mean CE loss for that cell type.
            Returns 0.0 for cell types with no samples in the batch.
    """
    ce_losses = []
    logits_flat = logits.reshape(-1, num_classes)
    labels_flat = labels.reshape(-1)

    for i in range(num_classes):
        # Only consider samples with label i and not ignore_index
        mask = (labels_flat == i) & (labels_flat != ignore_index)
        if mask.sum() == 0:
            # No samples for this cell type in this batch
            ce_losses.append(torch.tensor(0.0, device=logits.device))
            continue

        logits_i = logits_flat[mask]
        labels_i = labels_flat[mask]
        ce = nn.CrossEntropyLoss(reduction="mean")
        ce_loss = ce(logits_i, labels_i)
        ce_losses.append(ce_loss)

    if len(ce_losses) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(ce_losses)  # Shape: (num_classes,)


def calculate_mean_ce_loss(logits, labels, num_classes, ignore_index=-100):
    """
    Calculate mean cross-entropy loss across all cell types.

    This is a wrapper around calculate_per_class_ce_loss that returns
    the mean loss across all cell types. This provides a balanced loss
    where each cell type contributes equally regardless of sample count.

    Args:
        logits: Model output logits, shape (batch, num_classes)
        labels: Ground truth cell type labels, shape (batch,)
        num_classes: Number of cell types
        ignore_index: Label value to ignore in loss calculation (default: -100)

    Returns:
        torch.Tensor: Scalar mean CE loss averaged across all cell types
    """
    per_class_ce_loss = calculate_per_class_ce_loss(
        logits, labels, num_classes, ignore_index
    )
    mean_ce_loss = (
        per_class_ce_loss.mean()
        if per_class_ce_loss is not None
        else torch.tensor(0.0, device=logits.device)
    )
    return mean_ce_loss
