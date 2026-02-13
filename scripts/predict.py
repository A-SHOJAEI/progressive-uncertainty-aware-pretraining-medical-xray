#!/usr/bin/env python
"""Inference script for making predictions on new chest X-ray images."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from progressive_uncertainty_aware_pretraining_medical_xray.data import (
    get_val_transforms,
    SyntheticChestXrayDataset,
)
from progressive_uncertainty_aware_pretraining_medical_xray.models import create_model
from progressive_uncertainty_aware_pretraining_medical_xray.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Make predictions on chest X-ray images')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to input image (if not specified, uses synthetic sample)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for positive prediction',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save prediction results (JSON format)',
    )
    return parser.parse_args()


def load_image(image_path: str, transform) -> torch.Tensor:
    """Load and preprocess image.

    Args:
        image_path: Path to image file.
        transform: Preprocessing transform.

    Returns:
        Preprocessed image tensor.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Apply transform
    transformed = transform(image=image_np)
    image_tensor = transformed['image']

    return image_tensor.unsqueeze(0)  # Add batch dimension


def generate_synthetic_sample(transform) -> torch.Tensor:
    """Generate a synthetic chest X-ray sample for demonstration.

    Args:
        transform: Preprocessing transform.

    Returns:
        Preprocessed synthetic image tensor.
    """
    # Create synthetic dataset with 1 sample
    dataset = SyntheticChestXrayDataset(
        num_samples=1,
        num_classes=5,
        image_size=(224, 224),
        uncertain_ratio=0.0,
        seed=12345,
    )

    # Get sample
    sample = dataset[0]
    image = sample['image']

    # If transform is provided and image needs preprocessing
    if transform is not None and image.max() <= 1.0:
        # Convert back to numpy for albumentations
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        transformed = transform(image=image_np)
        image = transformed['image']

    return image.unsqueeze(0)  # Add batch dimension


def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """Make prediction on a single image.

    Args:
        model: Trained model.
        image: Preprocessed image tensor.
        device: Device to run inference on.
        class_names: List of class names.
        threshold: Threshold for positive prediction.

    Returns:
        Dictionary containing predictions and probabilities.
    """
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        probabilities = outputs['probabilities'].cpu().numpy()[0]

    # Create predictions
    predictions = []
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        is_positive = prob >= threshold
        predictions.append({
            'class': class_name,
            'probability': float(prob),
            'prediction': 'Positive' if is_positive else 'Negative',
            'confidence': float(prob if is_positive else 1 - prob),
        })

    return {
        'predictions': predictions,
        'threshold': threshold,
    }


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create preprocessing transform
    image_size = config.get('image_size', 224)
    transform = get_val_transforms(image_size=image_size)

    # Load or generate image
    if args.image:
        logger.info(f"Loading image from {args.image}")
        image_tensor = load_image(args.image, transform)
    else:
        logger.info("Generating synthetic sample for demonstration")
        image_tensor = generate_synthetic_sample(transform)

    # Get class names
    class_names = SyntheticChestXrayDataset.PATHOLOGIES[:config.get('num_classes', 5)]

    # Make prediction
    logger.info("Making prediction...")
    result = predict(
        model=model,
        image=image_tensor,
        device=device,
        class_names=class_names,
        threshold=args.threshold,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Threshold: {result['threshold']:.2f}\n")

    for pred in result['predictions']:
        logger.info(f"{pred['class']:20s}: {pred['prediction']:8s} "
                   f"(Probability: {pred['probability']:.4f}, "
                   f"Confidence: {pred['confidence']:.4f})")
    logger.info("=" * 60 + "\n")

    # Save results if output path is specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    logger.info("Prediction completed successfully!")


if __name__ == '__main__':
    main()
