"""Training loop with curriculum learning and uncertainty weighting."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import CurriculumScheduler, UncertaintyAwareLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager with curriculum learning and early stopping.

    Args:
        model: PyTorch model to train.
        criterion: Loss function (UncertaintyAwareLoss recommended).
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        checkpoint_dir: Directory to save model checkpoints.
        mixed_precision: If True, use automatic mixed precision training.
        gradient_clip: Maximum gradient norm for clipping (None to disable).
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        checkpoint_dir: str = 'checkpoints',
        mixed_precision: bool = True,
        gradient_clip: Optional[float] = 1.0,
        early_stopping_patience: int = 10,
    ):
        """Initialize trainer."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_path = None

        logger.info(
            f"Trainer initialized - Device: {device}, "
            f"Mixed precision: {mixed_precision}, "
            f"Gradient clip: {gradient_clip}"
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_certain_loss = 0.0
        total_uncertain_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            uncertainty_mask = batch['uncertainty_mask'].to(self.device)
            confidence_scores = batch['confidence_scores'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(
                        outputs['logits'],
                        labels,
                        uncertainty_mask,
                        confidence_scores,
                    )
                    loss = loss_dict['loss']

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(
                    outputs['logits'],
                    labels,
                    uncertainty_mask,
                    confidence_scores,
                )
                loss = loss_dict['loss']

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_certain_loss += loss_dict['certain_loss'].item()
            total_uncertain_loss += loss_dict['uncertain_loss'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'certain_loss': total_certain_loss / num_batches,
                'uncertain_loss': total_uncertain_loss / num_batches,
            })

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_certain_loss = total_certain_loss / num_batches
        avg_uncertain_loss = total_uncertain_loss / num_batches

        return {
            'train_loss': avg_loss,
            'train_certain_loss': avg_certain_loss,
            'train_uncertain_loss': avg_uncertain_loss,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            uncertainty_mask = batch['uncertainty_mask'].to(self.device)
            confidence_scores = batch['confidence_scores'].to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss_dict = self.criterion(
                outputs['logits'],
                labels,
                uncertainty_mask,
                confidence_scores,
            )
            loss = loss_dict['loss']

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

        # Average metrics
        avg_loss = total_loss / num_batches

        return {'val_loss': avg_loss}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, list]:
        """Full training loop with curriculum learning and early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update curriculum if using UncertaintyAwareLoss
            if isinstance(self.criterion, UncertaintyAwareLoss):
                if self.criterion.curriculum_scheduler is not None:
                    self.criterion.curriculum_scheduler.step()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rate'].append(current_lr)

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Check for improvement
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0

                # Save best model
                self.best_model_path = self.checkpoint_dir / 'best_model.pth'
                self.save_checkpoint(self.best_model_path, epoch, val_metrics['val_loss'])
                logger.info(f"New best model saved: {self.best_model_path}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_metrics['val_loss'])

        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            val_loss: Validation loss.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
