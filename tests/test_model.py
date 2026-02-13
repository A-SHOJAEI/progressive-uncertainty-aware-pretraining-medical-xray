"""Tests for model architecture and components."""

import pytest
import torch

from progressive_uncertainty_aware_pretraining_medical_xray.models import (
    ChestXrayClassifier,
    create_model,
    UncertaintyAwareLoss,
    CurriculumScheduler,
    TemperatureScaling,
)


class TestChestXrayClassifier:
    """Tests for ChestXrayClassifier."""

    def test_model_creation(self):
        """Test model creation."""
        model = ChestXrayClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False,
            dropout=0.3,
            use_temperature_scaling=True,
        )

        assert model is not None
        assert model.num_classes == 5
        assert model.use_temperature_scaling is True

    def test_forward_pass(self, device):
        """Test forward pass."""
        model = ChestXrayClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False,
        )
        model = model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224).to(device)

        # Forward pass
        output = model(x)

        assert 'logits' in output
        assert 'probabilities' in output
        assert output['logits'].shape == (batch_size, 5)
        assert output['probabilities'].shape == (batch_size, 5)

        # Check probabilities are in [0, 1]
        assert torch.all(output['probabilities'] >= 0)
        assert torch.all(output['probabilities'] <= 1)

    def test_return_features(self, device):
        """Test returning intermediate features."""
        model = ChestXrayClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False,
        )
        model = model.to(device)
        model.eval()

        x = torch.randn(4, 3, 224, 224).to(device)
        output = model(x, return_features=True)

        assert 'features' in output
        assert output['features'].shape[0] == 4

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing backbone."""
        model = ChestXrayClassifier(
            model_name='resnet18',
            num_classes=5,
            pretrained=False,
        )

        # Freeze
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = CurriculumScheduler(
            total_epochs=50,
            warmup_epochs=5,
            strategy='cosine',
        )

        assert scheduler.total_epochs == 50
        assert scheduler.warmup_epochs == 5
        assert scheduler.strategy == 'cosine'
        assert scheduler.current_epoch == 0

    def test_warmup_phase(self):
        """Test warmup phase returns 0 weight."""
        scheduler = CurriculumScheduler(
            total_epochs=50,
            warmup_epochs=5,
            strategy='linear',
        )

        # During warmup, weight should be 0
        for _ in range(5):
            weight = scheduler.get_uncertainty_weight()
            assert weight == 0.0
            scheduler.step()

    def test_linear_strategy(self):
        """Test linear curriculum strategy."""
        scheduler = CurriculumScheduler(
            total_epochs=10,
            warmup_epochs=0,
            strategy='linear',
        )

        weights = []
        for _ in range(10):
            weights.append(scheduler.get_uncertainty_weight())
            scheduler.step()

        # Weights should increase linearly
        assert weights[0] < weights[5] < weights[9]
        assert abs(weights[-1] - 1.0) < 0.1

    def test_cosine_strategy(self):
        """Test cosine curriculum strategy."""
        scheduler = CurriculumScheduler(
            total_epochs=10,
            warmup_epochs=0,
            strategy='cosine',
        )

        weights = []
        for _ in range(10):
            weights.append(scheduler.get_uncertainty_weight())
            scheduler.step()

        # Weights should increase and reach ~1.0 at the end
        assert weights[0] < weights[-1]
        assert abs(weights[-1] - 1.0) < 0.1


class TestUncertaintyAwareLoss:
    """Tests for UncertaintyAwareLoss."""

    def test_loss_creation(self):
        """Test loss function creation."""
        scheduler = CurriculumScheduler(total_epochs=50, warmup_epochs=5)
        loss_fn = UncertaintyAwareLoss(
            curriculum_scheduler=scheduler,
            base_weight=1.0,
            uncertain_weight_min=0.1,
            uncertain_weight_max=0.8,
            confidence_calibration=True,
        )

        assert loss_fn is not None

    def test_loss_computation(self, sample_batch):
        """Test loss computation."""
        scheduler = CurriculumScheduler(total_epochs=50, warmup_epochs=5)
        loss_fn = UncertaintyAwareLoss(curriculum_scheduler=scheduler)

        predictions = torch.randn(4, 5)
        targets = sample_batch['labels']
        uncertainty_mask = sample_batch['uncertainty_mask']
        confidence_scores = sample_batch['confidence_scores']

        loss_dict = loss_fn(predictions, targets, uncertainty_mask, confidence_scores)

        assert 'loss' in loss_dict
        assert 'certain_loss' in loss_dict
        assert 'uncertain_loss' in loss_dict
        assert 'curriculum_weight' in loss_dict

        # Loss should be a scalar
        assert loss_dict['loss'].dim() == 0
        assert loss_dict['loss'].item() >= 0

    def test_curriculum_effect(self, sample_batch):
        """Test that curriculum affects loss."""
        scheduler = CurriculumScheduler(total_epochs=10, warmup_epochs=0, strategy='linear')
        loss_fn = UncertaintyAwareLoss(curriculum_scheduler=scheduler)

        predictions = torch.randn(4, 5)
        targets = sample_batch['labels']
        uncertainty_mask = sample_batch['uncertainty_mask']
        confidence_scores = sample_batch['confidence_scores']

        # Loss at beginning of curriculum
        loss_dict_start = loss_fn(predictions, targets, uncertainty_mask, confidence_scores)

        # Advance curriculum
        for _ in range(9):
            scheduler.step()

        # Loss at end of curriculum
        loss_dict_end = loss_fn(predictions, targets, uncertainty_mask, confidence_scores)

        # Curriculum weight should have increased
        assert loss_dict_end['curriculum_weight'] > loss_dict_start['curriculum_weight']


class TestTemperatureScaling:
    """Tests for TemperatureScaling."""

    def test_temperature_scaling(self):
        """Test temperature scaling layer."""
        temp_scaling = TemperatureScaling(init_temperature=1.5)

        logits = torch.randn(4, 5)
        scaled_logits = temp_scaling(logits)

        assert scaled_logits.shape == logits.shape

        # Scaled logits should be smaller in magnitude (temperature > 1)
        assert torch.abs(scaled_logits).mean() < torch.abs(logits).mean()

    def test_temperature_parameter(self):
        """Test that temperature is a learnable parameter."""
        temp_scaling = TemperatureScaling(init_temperature=2.0)

        # Should have one parameter (temperature)
        params = list(temp_scaling.parameters())
        assert len(params) == 1
        assert params[0].requires_grad


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_model_from_config(self, sample_config):
        """Test creating model from config."""
        model = create_model(sample_config)

        assert model is not None
        assert model.num_classes == sample_config['num_classes']
        assert model.use_temperature_scaling == sample_config['use_temperature_scaling']
