"""Custom loss functions, layers, and training components.

This module implements the novel uncertainty-aware curriculum learning approach.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UncertaintyAwareLoss(nn.Module):
    """Uncertainty-aware binary cross-entropy loss with curriculum weighting.

    Novel contribution: Dynamically weighs uncertain labels based on model confidence
    and curriculum progress, allowing the model to gradually learn from uncertain
    samples as training progresses.

    Args:
        curriculum_scheduler: Scheduler to determine current difficulty threshold.
        base_weight: Base weight for certain labels (default: 1.0).
        uncertain_weight_min: Minimum weight for uncertain labels (default: 0.1).
        uncertain_weight_max: Maximum weight for uncertain labels (default: 0.8).
        confidence_calibration: If True, use model confidence to modulate weights.
    """

    def __init__(
        self,
        curriculum_scheduler: Optional['CurriculumScheduler'] = None,
        base_weight: float = 1.0,
        uncertain_weight_min: float = 0.1,
        uncertain_weight_max: float = 0.8,
        confidence_calibration: bool = True,
    ):
        """Initialize uncertainty-aware loss."""
        super().__init__()
        self.curriculum_scheduler = curriculum_scheduler
        self.base_weight = base_weight
        self.uncertain_weight_min = uncertain_weight_min
        self.uncertain_weight_max = uncertain_weight_max
        self.confidence_calibration = confidence_calibration

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty_mask: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty-aware loss.

        Args:
            predictions: Model predictions (batch_size, num_classes).
            targets: Ground truth labels (batch_size, num_classes).
            uncertainty_mask: Binary mask for uncertain labels (batch_size, num_classes).
            confidence_scores: Confidence scores for each label (batch_size, num_classes).

        Returns:
            Dictionary containing total loss and loss components.
        """
        # Base BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        # Create sample weights based on uncertainty and curriculum
        certain_mask = 1.0 - uncertainty_mask
        sample_weights = torch.ones_like(bce_loss) * self.base_weight

        # Get curriculum progress if scheduler is available
        if self.curriculum_scheduler is not None:
            curriculum_weight = self.curriculum_scheduler.get_uncertainty_weight()
        else:
            curriculum_weight = self.uncertain_weight_max

        # Weight uncertain samples based on curriculum
        uncertain_weight = self.uncertain_weight_min + (
            curriculum_weight * (self.uncertain_weight_max - self.uncertain_weight_min)
        )

        # Apply confidence-based modulation for uncertain labels
        if self.confidence_calibration:
            # Higher confidence scores get higher weights
            # This is the meta-learning component: learn to trust confident uncertain labels
            confidence_modulated_weight = uncertain_weight * confidence_scores
            sample_weights = sample_weights * certain_mask + confidence_modulated_weight * uncertainty_mask
        else:
            sample_weights = sample_weights * certain_mask + uncertain_weight * uncertainty_mask

        # Compute weighted loss
        weighted_loss = bce_loss * sample_weights
        total_loss = weighted_loss.mean()

        # Compute separate losses for analysis
        certain_loss = (bce_loss * certain_mask).sum() / (certain_mask.sum() + 1e-8)
        uncertain_loss = (bce_loss * uncertainty_mask).sum() / (uncertainty_mask.sum() + 1e-8)

        return {
            'loss': total_loss,
            'certain_loss': certain_loss,
            'uncertain_loss': uncertain_loss,
            'curriculum_weight': torch.tensor(curriculum_weight),
        }


class CurriculumScheduler:
    """Curriculum scheduler for progressive uncertainty weighting.

    Gradually increases the weight of uncertain samples during training,
    implementing a curriculum learning strategy.

    Args:
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of epochs before starting to increase uncertainty weight.
        strategy: Weighting strategy ('linear', 'cosine', 'exponential').
    """

    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 5,
        strategy: str = 'cosine',
    ):
        """Initialize curriculum scheduler."""
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        self.current_epoch = 0

        logger.info(
            f"Initialized curriculum scheduler: {strategy} strategy, "
            f"{warmup_epochs} warmup epochs, {total_epochs} total epochs"
        )

    def step(self) -> None:
        """Advance curriculum by one epoch."""
        self.current_epoch += 1

    def get_uncertainty_weight(self) -> float:
        """Get current uncertainty weight based on curriculum progress.

        Returns:
            Weight value between 0.0 and 1.0.
        """
        if self.current_epoch < self.warmup_epochs:
            return 0.0

        progress = (self.current_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        progress = min(progress, 1.0)

        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'cosine':
            # Smooth cosine schedule
            return 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159265359))).item()
        elif self.strategy == 'exponential':
            # Exponential growth
            return 1.0 - torch.exp(-3.0 * torch.tensor(progress)).item()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class TemperatureScaling(nn.Module):
    """Temperature scaling layer for calibrating model predictions.

    Applies a learned temperature parameter to logits to improve calibration.

    Args:
        init_temperature: Initial temperature value (default: 1.5).
    """

    def __init__(self, init_temperature: float = 1.5):
        """Initialize temperature scaling."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Model logits (batch_size, num_classes).

        Returns:
            Temperature-scaled logits.
        """
        return logits / self.temperature


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (alternative loss component).

    Args:
        alpha: Weighting factor for class imbalance.
        gamma: Focusing parameter for modulating loss (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: Model predictions (batch_size, num_classes).
            targets: Ground truth labels (batch_size, num_classes).

        Returns:
            Focal loss value.
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        probs = torch.sigmoid(predictions)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()
