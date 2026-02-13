"""Evaluation metrics for multi-label classification."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logger = logging.getLogger(__name__)


def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        AUROC score.
    """
    try:
        auroc = roc_auc_score(y_true, y_pred, average=average)
        return float(auroc)
    except ValueError as e:
        logger.warning(f"Could not compute AUROC: {e}")
        return 0.0


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        n_bins: Number of bins for calibration.

    Returns:
        Expected calibration error.
    """
    # Flatten arrays for binary calibration per class
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(y_true_flat)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_flat > bin_lower) & (y_pred_flat <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            # Average confidence in bin
            avg_confidence = y_pred_flat[in_bin].mean()

            # Average accuracy in bin
            avg_accuracy = y_true_flat[in_bin].mean()

            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def compute_average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
) -> float:
    """Compute Average Precision Score.

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        Average precision score.
    """
    try:
        ap = average_precision_score(y_true, y_pred, average=average)
        return float(ap)
    except ValueError as e:
        logger.warning(f"Could not compute AP: {e}")
        return 0.0


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = 'macro',
) -> float:
    """Compute F1 score for multi-label classification.

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        threshold: Threshold for converting probabilities to binary predictions.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        F1 score.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred_binary, average=average, zero_division=0)
    return float(f1)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty_mask: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels (n_samples, n_classes).
        y_pred: Predicted probabilities (n_samples, n_classes).
        uncertainty_mask: Optional mask for uncertain labels (n_samples, n_classes).
        class_names: Optional list of class names for per-class metrics.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}

    # Overall metrics
    metrics['mean_auroc'] = compute_auroc(y_true, y_pred, average='macro')
    metrics['micro_auroc'] = compute_auroc(y_true, y_pred, average='micro')
    metrics['expected_calibration_error'] = compute_expected_calibration_error(y_true, y_pred)
    metrics['average_precision'] = compute_average_precision(y_true, y_pred, average='macro')
    metrics['f1_score'] = compute_f1_score(y_true, y_pred, threshold=0.5, average='macro')

    # Per-class AUROC
    num_classes = y_true.shape[1]
    class_names = class_names or [f'Class_{i}' for i in range(num_classes)]

    for i, class_name in enumerate(class_names):
        try:
            class_auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            metrics[f'auroc_{class_name}'] = float(class_auroc)
        except ValueError:
            metrics[f'auroc_{class_name}'] = 0.0

    # Metrics on uncertain labels if mask is provided
    if uncertainty_mask is not None:
        uncertain_indices = uncertainty_mask == 1.0

        if uncertain_indices.any():
            y_true_uncertain = y_true[uncertain_indices]
            y_pred_uncertain = y_pred[uncertain_indices]

            # Reshape for metrics computation
            n_uncertain = uncertain_indices.sum()
            y_true_uncertain_reshaped = y_true_uncertain.reshape(-1, 1) if y_true_uncertain.ndim == 1 else y_true_uncertain
            y_pred_uncertain_reshaped = y_pred_uncertain.reshape(-1, 1) if y_pred_uncertain.ndim == 1 else y_pred_uncertain

            # Ensure 2D arrays
            if y_true_uncertain_reshaped.ndim == 1:
                y_true_uncertain_reshaped = y_true_uncertain_reshaped.reshape(-1, 1)
            if y_pred_uncertain_reshaped.ndim == 1:
                y_pred_uncertain_reshaped = y_pred_uncertain_reshaped.reshape(-1, 1)

            try:
                metrics['uncertain_label_auroc'] = compute_auroc(
                    y_true_uncertain_reshaped, y_pred_uncertain_reshaped, average='macro'
                )
            except Exception as e:
                logger.warning(f"Could not compute uncertain label AUROC: {e}")
                metrics['uncertain_label_auroc'] = 0.0
        else:
            metrics['uncertain_label_auroc'] = 0.0

    return metrics


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect predictions from model on a dataset.

    Args:
        model: PyTorch model.
        dataloader: Data loader.
        device: Device to run inference on.

    Returns:
        Dictionary containing predictions, labels, and uncertainty masks.
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_uncertainty_masks = []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['labels']
        uncertainty_mask = batch['uncertainty_mask']

        # Forward pass
        outputs = model(images)
        predictions = outputs['probabilities'].cpu().numpy()

        all_predictions.append(predictions)
        all_labels.append(labels.numpy())
        all_uncertainty_masks.append(uncertainty_mask.numpy())

    return {
        'predictions': np.concatenate(all_predictions, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
        'uncertainty_masks': np.concatenate(all_uncertainty_masks, axis=0),
    }
