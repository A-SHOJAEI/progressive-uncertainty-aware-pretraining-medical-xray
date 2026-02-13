"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from progressive_uncertainty_aware_pretraining_medical_xray.data import (
    SyntheticChestXrayDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
)


class TestSyntheticChestXrayDataset:
    """Tests for SyntheticChestXrayDataset."""

    def test_dataset_creation(self, random_seed):
        """Test dataset creation."""
        dataset = SyntheticChestXrayDataset(
            num_samples=100,
            num_classes=5,
            image_size=(224, 224),
            uncertain_ratio=0.3,
            seed=random_seed,
        )

        assert len(dataset) == 100
        assert dataset.num_classes == 5
        assert dataset.uncertain_ratio == 0.3

    def test_dataset_getitem(self, random_seed):
        """Test dataset __getitem__ method."""
        dataset = SyntheticChestXrayDataset(
            num_samples=10,
            num_classes=5,
            seed=random_seed,
        )

        sample = dataset[0]

        assert 'image' in sample
        assert 'labels' in sample
        assert 'uncertainty_mask' in sample
        assert 'confidence_scores' in sample

        # Check shapes
        assert sample['image'].shape[0] == 3  # RGB channels
        assert sample['labels'].shape[0] == 5
        assert sample['uncertainty_mask'].shape[0] == 5
        assert sample['confidence_scores'].shape[0] == 5

    def test_uncertain_ratio(self, random_seed):
        """Test that uncertain ratio is approximately correct."""
        dataset = SyntheticChestXrayDataset(
            num_samples=1000,
            num_classes=5,
            uncertain_ratio=0.3,
            seed=random_seed,
        )

        # Count samples with at least one uncertain label
        samples_with_uncertainty = (dataset.uncertainty_mask.sum(axis=1) > 0).sum()
        actual_sample_ratio = samples_with_uncertainty / dataset.num_samples

        # Should be approximately 0.3 (allow some variance)
        assert 0.25 < actual_sample_ratio < 0.35

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        dataset1 = SyntheticChestXrayDataset(num_samples=10, seed=42)
        dataset2 = SyntheticChestXrayDataset(num_samples=10, seed=42)

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1['labels'], sample2['labels'])
        assert torch.allclose(sample1['uncertainty_mask'], sample2['uncertainty_mask'])


class TestDataTransforms:
    """Tests for data transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_train_transforms(image_size=224, advanced=True)
        assert transform is not None

        # Test on dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transform(image=dummy_image)

        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
        assert transformed['image'].shape == (3, 224, 224)

    def test_val_transforms(self):
        """Test validation transforms."""
        transform = get_val_transforms(image_size=224)
        assert transform is not None

        # Test on dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transform(image=dummy_image)

        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
        assert transformed['image'].shape == (3, 224, 224)


class TestDataLoaders:
    """Tests for dataloader creation."""

    def test_create_dataloaders(self, sample_config):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(sample_config)

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch size
        batch = next(iter(train_loader))
        assert batch['image'].shape[0] <= sample_config['batch_size']

    def test_dataloader_splits(self, sample_config):
        """Test that data is split correctly."""
        train_loader, val_loader, test_loader = create_dataloaders(sample_config)

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        total_size = train_size + val_size + test_size

        # Should sum to num_samples
        assert total_size == sample_config['num_samples']

        # Check approximate ratios
        assert abs(train_size / total_size - 0.7) < 0.05
        assert abs(val_size / total_size - 0.15) < 0.05
