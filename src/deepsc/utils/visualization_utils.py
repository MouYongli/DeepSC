"""
Visualization utility functions for model evaluation.

This module provides utilities for generating evaluation plots and reports
for classification tasks.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def process_classification_metrics(y_true, y_pred, id2label):
    """
    Process evaluation data and compute classification metrics.

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        id2label: Dictionary mapping label IDs to label names

    Returns:
        dict: Dictionary containing processed metrics data including:
            - categories: List of category names
            - recalls: List of recall scores
            - precisions: List of precision scores
            - f1_scores: List of F1 scores
            - supports: List of support counts
            - proportions: List of class proportions
            - unique_labels: Array of unique label IDs
            - y_true: True labels
            - y_pred: Predicted labels
        None: If no valid categories are found
    """
    # Get all unique labels from id2label
    num_classes = len(id2label)
    eval_labels = np.arange(num_classes)
    target_names = [id2label[i] for i in eval_labels]

    # Compute classification metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=eval_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Extract metric data
    metrics_data = {
        "categories": target_names,
        "recalls": [report[name]["recall"] for name in target_names],
        "precisions": [report[name]["precision"] for name in target_names],
        "f1_scores": [report[name]["f1-score"] for name in target_names],
        "supports": [report[name]["support"] for name in target_names],
    }

    # Calculate class proportions
    total_samples = sum(metrics_data["supports"])
    metrics_data["proportions"] = [s / total_samples for s in metrics_data["supports"]]
    metrics_data["unique_labels"] = eval_labels
    metrics_data["y_true"] = y_true
    metrics_data["y_pred"] = y_pred

    return metrics_data


def plot_classification_metrics(
    y_true, y_pred, id2label, save_dir="./evaluation_plots", epoch=0
):
    """
    Plot evaluation charts for classification tasks.

    Generates:
    - Classification metrics chart (4 subplots: Recall, Precision, F1, Proportion)
    - Confusion matrix heatmap
    - CSV report with detailed metrics

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        id2label: Dictionary mapping label IDs to label names
        save_dir: Directory to save the plots (default: "./evaluation_plots")
        epoch: Current epoch number (for filename)
    """
    # Process evaluation data
    processed_data = process_classification_metrics(y_true, y_pred, id2label)
    if processed_data is None:
        print("Warning: No valid categories found for plotting")
        return

    # Unpack processed data
    categories = processed_data["categories"]
    recalls = processed_data["recalls"]
    precisions = processed_data["precisions"]
    f1_scores = processed_data["f1_scores"]
    supports = processed_data["supports"]
    proportions = processed_data["proportions"]
    unique_labels = processed_data["unique_labels"]
    y_true = processed_data["y_true"]
    y_pred = processed_data["y_pred"]

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Classification metrics details (4 subplots)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Classification Metrics by Cell Type - Epoch {epoch}",
        fontsize=16,
        fontweight="bold",
    )

    # Subplot 1: Recall
    bars1 = ax1.bar(range(len(categories)), recalls, color="skyblue", alpha=0.8)
    ax1.set_title("Recall by Cell Type")
    ax1.set_ylabel("Recall")
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Subplot 2: Precision
    bars2 = ax2.bar(range(len(categories)), precisions, color="lightcoral", alpha=0.8)
    ax2.set_title("Precision by Cell Type")
    ax2.set_ylabel("Precision")
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Subplot 3: F1 Score
    bars3 = ax3.bar(range(len(categories)), f1_scores, color="lightgreen", alpha=0.8)
    ax3.set_title("F1 Score by Cell Type")
    ax3.set_ylabel("F1 Score")
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha="right")
    ax3.set_ylim(0, 1)
    ax3.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Subplot 4: Class proportion
    bars4 = ax4.bar(range(len(categories)), proportions, color="gold", alpha=0.8)
    ax4.set_title("Class Proportion")
    ax4.set_ylabel("Proportion")
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha="right")
    ax4.set_ylim(0, max(proportions) * 1.1 if proportions else 1)
    ax4.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    metrics_file = os.path.join(save_dir, f"classification_metrics_epoch_{epoch}.png")
    plt.savefig(metrics_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Confusion matrix (only categories present in true labels)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(
        cm_normalized,
        index=categories[: cm.shape[0]],
        columns=categories[: cm.shape[1]],
    )

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"shrink": 0.8})
    plt.title(
        f"Confusion Matrix (Normalized) - Epoch {epoch}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    confusion_file = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(confusion_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Create and save metrics table
    metrics_df = pd.DataFrame(
        {
            "Cell Type": categories,
            "Recall": recalls,
            "Precision": precisions,
            "F1 Score": f1_scores,
            "Support": supports,
            "Proportion": proportions,
        }
    )
    csv_file = os.path.join(save_dir, f"classification_metrics_epoch_{epoch}.csv")
    # Format numeric columns to 2 decimal places
    metrics_df["Recall"] = metrics_df["Recall"].round(2)
    metrics_df["Precision"] = metrics_df["Precision"].round(2)
    metrics_df["F1 Score"] = metrics_df["F1 Score"].round(2)
    metrics_df["Proportion"] = metrics_df["Proportion"].round(2)
    metrics_df.to_csv(csv_file, index=False)

    print("Evaluation plots saved to:")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Confusion Matrix: {confusion_file}")
    print(f"  - CSV Report: {csv_file}")
