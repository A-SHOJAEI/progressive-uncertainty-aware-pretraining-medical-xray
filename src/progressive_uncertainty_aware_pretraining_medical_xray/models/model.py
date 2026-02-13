"""Core model architecture for chest X-ray classification."""

import logging
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn

from .components import TemperatureScaling

logger = logging.getLogger(__name__)


class ChestXrayClassifier(nn.Module):
    """Multi-label chest X-ray classifier with ImageNet pretraining.

    Args:
        model_name: Name of the timm model to use as backbone.
        num_classes: Number of disease classes to predict.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout probability for classifier head.
        use_temperature_scaling: If True, add temperature scaling for calibration.
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_temperature_scaling: bool = True,
    ):
        """Initialize chest X-ray classifier."""
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_temperature_scaling = use_temperature_scaling

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Temperature scaling for calibration
        if use_temperature_scaling:
            self.temperature_scaling = TemperatureScaling(init_temperature=1.5)

        logger.info(
            f"Initialized {model_name} with {num_classes} classes, "
            f"pretrained={pretrained}, temperature_scaling={use_temperature_scaling}"
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images (batch_size, 3, height, width).
            return_features: If True, return intermediate features.

        Returns:
            Dictionary containing:
                - logits: Raw logits (batch_size, num_classes)
                - probabilities: Sigmoid probabilities (batch_size, num_classes)
                - features: Backbone features if return_features=True
        """
        # Extract features
        features = self.backbone(x)

        # Classification
        logits = self.classifier(features)

        # Apply temperature scaling if enabled
        if self.use_temperature_scaling:
            logits = self.temperature_scaling(logits)

        # Compute probabilities
        probabilities = torch.sigmoid(logits)

        output = {
            'logits': logits,
            'probabilities': probabilities,
        }

        if return_features:
            output['features'] = features

        return output

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict) -> ChestXrayClassifier:
    """Create model from configuration.

    Args:
        config: Configuration dictionary with model settings.

    Returns:
        Initialized ChestXrayClassifier model.
    """
    model_name = config.get('model_name', 'resnet50')
    num_classes = config.get('num_classes', 5)
    pretrained = config.get('pretrained', True)
    dropout = config.get('dropout', 0.3)
    use_temperature_scaling = config.get('use_temperature_scaling', True)

    model = ChestXrayClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        use_temperature_scaling=use_temperature_scaling,
    )

    logger.info(f"Model has {model.get_trainable_parameters():,} trainable parameters")

    return model
