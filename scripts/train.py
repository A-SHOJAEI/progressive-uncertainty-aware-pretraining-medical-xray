#!/usr/bin/env python
"""Training script for progressive uncertainty-aware pretraining."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from progressive_uncertainty_aware_pretraining_medical_xray.data import (
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    SyntheticChestXrayDataset,
)
from progressive_uncertainty_aware_pretraining_medical_xray.models import (
    create_model,
    UncertaintyAwareLoss,
    CurriculumScheduler,
)
from progressive_uncertainty_aware_pretraining_medical_xray.training import Trainer
from progressive_uncertainty_aware_pretraining_medical_xray.utils import load_config, set_seed
from progressive_uncertainty_aware_pretraining_medical_xray.evaluation import plot_training_curves

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log'),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train progressive uncertainty-aware chest X-ray classifier'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified.',
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize MLflow (wrapped in try-except as server may not be available)
    mlflow_enabled = config.get('mlflow_tracking', False)
    if mlflow_enabled:
        try:
            import mlflow
            experiment_name = config.get('experiment_name', 'chest_xray_classification')
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params(config)
            logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            mlflow_enabled = False

    try:
        # Create data transforms
        image_size = config.get('image_size', 224)
        advanced_aug = config.get('advanced_augmentation', True)
        train_transform = get_train_transforms(image_size=image_size, advanced=advanced_aug)
        val_transform = get_val_transforms(image_size=image_size)

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            train_transform=train_transform,
            val_transform=val_transform,
        )

        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)

        # Create curriculum scheduler
        curriculum_strategy = config.get('curriculum_strategy', 'cosine')
        curriculum_warmup = config.get('curriculum_warmup_epochs', 5)
        num_epochs = config.get('num_epochs', 50)

        curriculum_scheduler = CurriculumScheduler(
            total_epochs=num_epochs,
            warmup_epochs=curriculum_warmup,
            strategy=curriculum_strategy,
        )

        # Create loss function
        criterion = UncertaintyAwareLoss(
            curriculum_scheduler=curriculum_scheduler,
            base_weight=config.get('base_weight', 1.0),
            uncertain_weight_min=config.get('uncertain_weight_min', 0.1),
            uncertain_weight_max=config.get('uncertain_weight_max', 0.8),
            confidence_calibration=config.get('confidence_calibration', True),
        )

        # Create optimizer
        learning_rate = config.get('learning_rate', 0.0001)
        weight_decay = config.get('weight_decay', 0.0001)

        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create learning rate scheduler
        scheduler_type = config.get('scheduler_type', 'cosine')
        min_lr = config.get('min_lr', 0.000001)

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=min_lr,
            )
        else:
            scheduler = None

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            mixed_precision=config.get('mixed_precision', True),
            gradient_clip=config.get('gradient_clip', 1.0),
            early_stopping_patience=config.get('early_stopping_patience', 10),
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(Path(args.resume))

        # Train model
        logger.info(f"Starting training for {num_epochs} epochs...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
        )

        # Save training history
        history_path = results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # Plot training curves
        plot_path = results_dir / 'training_curves.png'
        plot_training_curves(history, save_path=str(plot_path))
        logger.info(f"Training curves saved to {plot_path}")

        # Log final metrics to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_metric('final_train_loss', history['train_loss'][-1])
                mlflow.log_metric('final_val_loss', history['val_loss'][-1])
                mlflow.log_metric('best_val_loss', trainer.best_val_loss)
                mlflow.log_artifact(str(plot_path))
                mlflow.log_artifact(str(history_path))
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {trainer.best_model_path}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        # End MLflow run
        if mlflow_enabled:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == '__main__':
    main()
