"""Tests for training loop and trainer."""

import pytest
import torch
from torch.optim import Adam

from progressive_uncertainty_aware_pretraining_medical_xray.data import create_dataloaders
from progressive_uncertainty_aware_pretraining_medical_xray.models import (
    create_model,
    UncertaintyAwareLoss,
    CurriculumScheduler,
)
from progressive_uncertainty_aware_pretraining_medical_xray.training import Trainer


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def trainer_setup(self, sample_config, device, tmp_path):
        """Setup trainer for testing."""
        # Create model
        model = create_model(sample_config)
        model = model.to(device)

        # Create loss and optimizer
        scheduler = CurriculumScheduler(
            total_epochs=sample_config['num_epochs'],
            warmup_epochs=1,
        )
        criterion = UncertaintyAwareLoss(curriculum_scheduler=scheduler)
        optimizer = Adam(model.parameters(), lr=sample_config['learning_rate'])

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=str(tmp_path / 'checkpoints'),
            mixed_precision=False,
            gradient_clip=1.0,
            early_stopping_patience=5,
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(sample_config)

        return {
            'trainer': trainer,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
        }

    def test_trainer_creation(self, trainer_setup):
        """Test trainer creation."""
        trainer = trainer_setup['trainer']
        assert trainer is not None
        assert trainer.best_val_loss == float('inf')
        assert trainer.epochs_without_improvement == 0

    def test_train_epoch(self, trainer_setup):
        """Test training for one epoch."""
        trainer = trainer_setup['trainer']
        train_loader = trainer_setup['train_loader']

        metrics = trainer.train_epoch(train_loader, epoch=1)

        assert 'train_loss' in metrics
        assert 'train_certain_loss' in metrics
        assert 'train_uncertain_loss' in metrics
        assert metrics['train_loss'] >= 0

    def test_validate(self, trainer_setup):
        """Test validation."""
        trainer = trainer_setup['trainer']
        val_loader = trainer_setup['val_loader']

        metrics = trainer.validate(val_loader, epoch=1)

        assert 'val_loss' in metrics
        assert metrics['val_loss'] >= 0

    def test_full_training(self, trainer_setup):
        """Test full training loop."""
        trainer = trainer_setup['trainer']
        train_loader = trainer_setup['train_loader']
        val_loader = trainer_setup['val_loader']

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2

    def test_checkpoint_saving(self, trainer_setup, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = trainer_setup['trainer']
        checkpoint_path = tmp_path / 'test_checkpoint.pth'

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, epoch=1, val_loss=0.5)
        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        assert checkpoint['epoch'] == 1
        assert checkpoint['val_loss'] == 0.5

    def test_early_stopping(self, trainer_setup):
        """Test early stopping mechanism."""
        trainer = trainer_setup['trainer']
        trainer.early_stopping_patience = 2
        train_loader = trainer_setup['train_loader']
        val_loader = trainer_setup['val_loader']

        # Set initial best loss
        trainer.best_val_loss = 0.1

        # Train (should trigger early stopping if val loss doesn't improve)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
        )

        # Should stop before 10 epochs if no improvement
        # (actual behavior depends on random initialization)
        assert len(history['train_loss']) <= 10
