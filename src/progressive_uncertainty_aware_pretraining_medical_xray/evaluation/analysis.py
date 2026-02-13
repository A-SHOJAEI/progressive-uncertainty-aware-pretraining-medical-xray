"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Training history dictionary with 'train_loss' and 'val_loss' keys.
        save_path: Optional path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rate' in history and history['learning_rate']:
        ax2.plot(epochs, history['learning_rate'], label='Learning Rate', color='green', marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")

    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        n_bins: Number of bins for calibration.
        save_path: Optional path to save the plot.
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_flat > bin_lower) & (y_pred_flat <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            avg_confidence = y_pred_flat[in_bin].mean()
            avg_accuracy = y_true_flat[in_bin].mean()
            bin_confidences.append(avg_confidence)
            bin_accuracies.append(avg_accuracy)
            bin_counts.append(bin_size)
        else:
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Calibration curve
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration', markersize=8, linewidth=2)

    # Add bin counts as bar widths
    ax.bar(
        bin_confidences,
        bin_accuracies,
        width=0.08,
        alpha=0.3,
        edgecolor='black',
        label='Bin Distribution'
    )

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")

    plt.close()


def plot_class_auroc_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot per-class AUROC comparison across different configurations.

    Args:
        metrics_dict: Dictionary mapping config names to their metrics.
        class_names: List of class names.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.8 / len(metrics_dict)

    for i, (config_name, metrics) in enumerate(metrics_dict.items()):
        auroc_values = [metrics.get(f'auroc_{class_name}', 0) for class_name in class_names]
        ax.bar(x + i * width, auroc_values, width, label=config_name)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Per-Class AUROC Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class AUROC comparison saved to {save_path}")

    plt.close()
