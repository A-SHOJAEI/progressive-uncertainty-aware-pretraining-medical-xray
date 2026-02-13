"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np

from progressive_uncertainty_aware_pretraining_medical_xray.utils import set_seed


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="function")
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    set_seed(seed)
    return seed


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'seed': 42,
        'num_samples': 100,
        'num_classes': 5,
        'image_size': 224,
        'uncertain_ratio': 0.3,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'batch_size': 8,
        'num_workers': 0,
        'model_name': 'resnet18',
        'pretrained': False,
        'dropout': 0.3,
        'use_temperature_scaling': True,
        'num_epochs': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'mixed_precision': False,
        'gradient_clip': 1.0,
        'curriculum_strategy': 'linear',
        'curriculum_warmup_epochs': 1,
        'base_weight': 1.0,
        'uncertain_weight_min': 0.1,
        'uncertain_weight_max': 0.8,
        'confidence_calibration': True,
        'early_stopping_patience': 5,
    }


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    batch_size = 4
    num_classes = 5
    image_size = 224

    return {
        'image': torch.randn(batch_size, 3, image_size, image_size),
        'labels': torch.randint(0, 2, (batch_size, num_classes)).float(),
        'uncertainty_mask': torch.randint(0, 2, (batch_size, num_classes)).float(),
        'confidence_scores': torch.rand(batch_size, num_classes),
    }
