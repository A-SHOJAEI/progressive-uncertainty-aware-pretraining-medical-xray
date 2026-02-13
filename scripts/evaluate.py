#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from progressive_uncertainty_aware_pretraining_medical_xray.data import (
    create_dataloaders,
    get_val_transforms,
    SyntheticChestXrayDataset,
)
from progressive_uncertainty_aware_pretraining_medical_xray.models import create_model
from progressive_uncertainty_aware_pretraining_medical_xray.evaluation import (
    collect_predictions,
    compute_metrics,
    plot_calibration_curve,
)
from progressive_uncertainty_aware_pretraining_medical_xray.utils import load_config, set_seed

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
    parser = argparse.ArgumentParser(description='Evaluate trained chest X-ray classifier')
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
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on',
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
        default='results/evaluation_results.json',
        help='Path to save evaluation results',
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Create data transforms
    image_size = config.get('image_size', 224)
    val_transform = get_val_transforms(image_size=image_size)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        train_transform=val_transform,
        val_transform=val_transform,
    )

    # Select evaluation split
    eval_loader = test_loader if args.split == 'test' else val_loader
    logger.info(f"Evaluating on {args.split} split ({len(eval_loader.dataset)} samples)")

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
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Collect predictions
    logger.info("Collecting predictions...")
    predictions_dict = collect_predictions(model, eval_loader, device)

    y_pred = predictions_dict['predictions']
    y_true = predictions_dict['labels']
    uncertainty_masks = predictions_dict['uncertainty_masks']

    # Compute metrics
    logger.info("Computing metrics...")
    class_names = SyntheticChestXrayDataset.PATHOLOGIES[:config.get('num_classes', 5)]
    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty_mask=uncertainty_masks,
        class_names=class_names,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean AUROC:                    {metrics['mean_auroc']:.4f}")
    logger.info(f"Micro AUROC:                   {metrics['micro_auroc']:.4f}")
    logger.info(f"Expected Calibration Error:    {metrics['expected_calibration_error']:.4f}")
    logger.info(f"Average Precision:             {metrics['average_precision']:.4f}")
    logger.info(f"F1 Score:                      {metrics['f1_score']:.4f}")
    if 'uncertain_label_auroc' in metrics:
        logger.info(f"Uncertain Label AUROC:         {metrics['uncertain_label_auroc']:.4f}")

    logger.info("\nPer-Class AUROC:")
    for class_name in class_names:
        auroc = metrics.get(f'auroc_{class_name}', 0.0)
        logger.info(f"  {class_name:20s}: {auroc:.4f}")
    logger.info("=" * 60 + "\n")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'config': args.config,
        'checkpoint': str(checkpoint_path),
        'split': args.split,
        'num_samples': len(eval_loader.dataset),
        'metrics': metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Plot calibration curve
    calibration_plot_path = results_dir / f'calibration_curve_{args.split}.png'
    plot_calibration_curve(y_true, y_pred, save_path=str(calibration_plot_path))
    logger.info(f"Calibration curve saved to {calibration_plot_path}")

    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
