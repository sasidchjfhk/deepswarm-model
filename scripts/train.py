#!/usr/bin/env python
# scripts/train.py
"""
Training script for Swarm IDS model.

Usage:
    python scripts/train.py --data-dir data/raw --epochs 100
"""

import argparse
import sys
from pathlib import Path
import torch
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import CICIDSDataLoader, TemporalSplitter, NetworkFeatureEngineer, LabelEncoder, create_dataloaders
from src.models import create_efficientnet_ids
from src.training import create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Swarm IDS model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing CICIDS2017 CSV files')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save models and outputs')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='b0',
                       choices=['b0', 'b1', 'b2', 'b3'],
                       help='EfficientNet model size')
    parser.add_argument('--num-features', type=int, default=50,
                       help='Number of features after selection')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle', 'plateau'],
                       help='LR scheduler')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Use Focal Loss instead of Cross Entropy')
    
    # System arguments
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SWARM IDS - TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # ========== PHASE 1: LOAD DATA ==========
    logger.info("\n[PHASE 1] Loading data...")
    
    loader = CICIDSDataLoader(data_dir)
    
    # Load all days (adjust based on available data)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dfs = {}
    
    for day in days:
        try:
            dfs[day] = loader.load_and_validate(day, validate_checksum=False)
        except Exception as e:
            logger.warning(f"Could not load {day}: {e}")
    
    if not dfs:
        logger.error("No data loaded! Please check data directory.")
        return
        
    # Split data temporally
    train_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    val_day = 'Friday'
    
    train_df, val_df, _ = TemporalSplitter.split_by_day(dfs, train_days, val_day)
    
    # ========== PHASE 2: FEATURE ENGINEERING ==========
    logger.info("\n[PHASE 2] Feature engineering...")
    
    # Separate features and labels
    label_col = 'Label'
    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    X_val = val_df.drop(columns=[label_col])
    y_val = val_df[label_col]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.encode(y_train)
    y_val_encoded = label_encoder.encode(y_val)
    
    # Feature engineering
    feature_engineer = NetworkFeatureEngineer(
        scaler_type='robust',
        feature_selection=True,
        k_best_features=args.num_features
    )
    
    X_train_processed = feature_engineer.fit_transform(X_train, y_train_encoded)
    X_val_processed = feature_engineer.transform(X_val)
    
    # Save feature engineer
    feature_engineer.save(output_dir / 'preprocessors')
    logger.info(f"✓ Saved feature engineering pipeline")
    
    # ========== PHASE 3: CREATE DATALOADERS ==========
    logger.info("\n[PHASE 3] Creating dataloaders...")
    
    train_loader, val_loader, class_weights = create_dataloaders(
        X_train_processed,
        y_train_encoded,
        X_val_processed,
        y_val_encoded,
        batch_size=args.batch_size,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    # ========== PHASE 4: CREATE MODEL ==========
    logger.info("\n[PHASE 4] Creating model...")
    
    num_features = X_train_processed.shape[1]
    num_classes = label_encoder.num_classes
    
    model = create_efficientnet_ids(
        num_features=num_features,
        num_classes=num_classes,
        model_size=args.model_size
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: EfficientNet-{args.model_size.upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ========== PHASE 5: CREATE TRAINER ==========
    logger.info("\n[PHASE 5] Creating trainer...")
    
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'epochs': args.epochs,
        'mixed_precision': not args.no_mixed_precision and torch.cuda.is_available(),
        'gradient_clip_val': args.gradient_clip,
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'experiment_name': f'swarm-ids-{args.model_size}',
        'use_mlflow': not args.no_mlflow,
        'focal_gamma': 2.0
    }
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        **config
    )
    
    # ========== PHASE 6: TRAIN ==========
    logger.info("\n[PHASE 6] Training...")
    logger.info(f"Training for {args.epochs} epochs")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Mixed Precision: {trainer.mixed_precision}")
    logger.info("")
    
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_every=10
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model saved to: {output_dir / 'checkpoints' / 'best_model.pth'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
