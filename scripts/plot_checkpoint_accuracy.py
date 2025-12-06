"""
Plot accuracy across checkpoints to visualize training dynamics
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from accuracy summary
checkpoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
accuracies = [
    0.6394,
    0.7003,
    0.6060,
    0.6337,
    0.5922,
    0.6069,
    0.6434,
    0.6147,
    0.6939,
    0.6991,
    0.6991,
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot accuracy line
ax.plot(
    checkpoints,
    accuracies,
    "o-",
    linewidth=2,
    markersize=8,
    color="#2E86AB",
    label="Reference Mapping Accuracy",
)

# Highlight best and worst
best_idx = np.argmax(accuracies)
worst_idx = np.argmin(accuracies)
ax.plot(
    checkpoints[best_idx],
    accuracies[best_idx],
    "g*",
    markersize=20,
    label=f"Best: Epoch {checkpoints[best_idx]} ({accuracies[best_idx]:.4f})",
)
ax.plot(
    checkpoints[worst_idx],
    accuracies[worst_idx],
    "r*",
    markersize=20,
    label=f"Worst: Epoch {checkpoints[worst_idx]} ({accuracies[worst_idx]:.4f})",
)

# Add horizontal line for mean
mean_acc = np.mean(accuracies)
ax.axhline(
    y=mean_acc, color="gray", linestyle="--", alpha=0.5, label=f"Mean: {mean_acc:.4f}"
)

# Styling
ax.set_xlabel("Epoch (Checkpoint)", fontsize=14, fontweight="bold")
ax.set_ylabel("Reference Mapping Accuracy", fontsize=14, fontweight="bold")
ax.set_title(
    "DeepSC Pre-training: Reference Mapping Accuracy Across Epochs",
    fontsize=16,
    fontweight="bold",
)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xticks(checkpoints)
ax.set_ylim([0.55, 0.75])
ax.legend(fontsize=11, loc="lower right")

# Add annotations for key observations
ax.annotate(
    "Sharp drop",
    xy=(3, accuracies[2]),
    xytext=(3.5, 0.58),
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    fontsize=10,
    color="red",
)

ax.annotate(
    "Recovery",
    xy=(9, accuracies[8]),
    xytext=(7.5, 0.72),
    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    fontsize=10,
    color="green",
)

plt.tight_layout()
plt.savefig(
    "/home/angli/DeepSC/results/reference_mapping_all_checkpoints/accuracy_trend.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Plot saved to: /home/angli/DeepSC/results/reference_mapping_all_checkpoints/accuracy_trend.png"
)

# Calculate statistics
print("\nStatistics:")
print(f"Mean accuracy: {mean_acc:.4f}")
print(f"Std deviation: {np.std(accuracies):.4f}")
print(f"Range: {np.max(accuracies) - np.min(accuracies):.4f}")
print(f"Best epoch: {checkpoints[best_idx]} (accuracy: {accuracies[best_idx]:.4f})")
print(f"Worst epoch: {checkpoints[worst_idx]} (accuracy: {accuracies[worst_idx]:.4f})")

# Check for monotonicity
increases = sum(
    [1 for i in range(1, len(accuracies)) if accuracies[i] > accuracies[i - 1]]
)
decreases = sum(
    [1 for i in range(1, len(accuracies)) if accuracies[i] < accuracies[i - 1]]
)
print("\nMonotonicity check:")
print(f"Number of increases: {increases}")
print(f"Number of decreases: {decreases}")
print(f"Number of no change: {len(accuracies) - 1 - increases - decreases}")
