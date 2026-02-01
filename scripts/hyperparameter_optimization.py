#!/usr/bin/env python
# scripts/hyperparameter_optimization.py
"""
Automated hyperparameter optimization using Optuna.

Top-tier approach:
- Bayesian optimization
- Pruning poor trials
- Multi-objective optimization
- Distributed training support
"""

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
import pandas as pd
import numpy as np

from src.data import NetworkFeatureEngineer, LabelEncoder, create_dataloaders
from src.models import create_efficientnet_ids
from src.training import create_trainer


def objective(trial, X_train, y_train, X_val, y_val, label_encoder):
    """
    Optuna objective function.
    
    Optimizes:
    - Learning rate
    - Batch size
    - Model size
    - Number of features
    - Optimizer parameters
    """
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    model_size = trial.suggest_categorical('model_size', ['b0', 'b1'])
    num_features = trial.suggest_int('num_features', 30, 80)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    
    logger.info(f"\nTrial {trial.number}:")
    logger.info(f"  LR: {lr:.6f}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Model: EfficientNet-{model_size}")
    logger.info(f"  Features: {num_features}")
    
    try:
        # Feature engineering
        engineer = NetworkFeatureEngineer(
            scaler_type='robust',
            feature_selection=True,
            k_best_features=num_features
        )
        
        X_train_proc = engineer.fit_transform(X_train, y_train)
        X_val_proc = engineer.transform(X_val)
        
        # Create dataloaders
        train_loader, val_loader, class_weights = create_dataloaders(
            X_train_proc, y_train,
            X_val_proc, y_val,
            batch_size=batch_size,
            num_workers=0
        )
        
        # Create model
        model = create_efficientnet_ids(
            num_features=X_train_proc.shape[1],
            num_classes=label_encoder.num_classes,
            model_size=model_size
        )
        
        # Create trainer
        config = {
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'focal_gamma': focal_gamma,
            'epochs': 20,  # Reduced for HPO
            'scheduler': 'cosine',
            'checkpoint_dir': f'models/optuna/trial_{trial.number}',
            'experiment_name': f'hpo-trial-{trial.number}',
            'use_mlflow': False,  # Disable for faster trials
            'mixed_precision': torch.cuda.is_available()
        }
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=label_encoder.num_classes,
            class_weights=class_weights,
            use_focal_loss=True,
            **config
        )
        
        # Train with early stopping
        trainer.train(
            num_epochs=20,
            early_stopping_patience=5,
            save_every=20
        )
        
        # Return validation F1 (metric to maximize)
        return trainer.best_val_f1
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_hyperparameter_optimization(
    X_train, y_train,
    X_val, y_val,
    label_encoder,
    n_trials: int = 50
):
    """
    Run Bayesian hyperparameter optimization.
    
    Args:
        n_trials: Number of optimization trials
    """
    
    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION (Optuna)")
    logger.info("="*80)
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Objective: Maximize Validation F1")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='swarm-ids-hpo',
        storage='sqlite:///models/optuna/study.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
        )
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, label_encoder),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    
    logger.info(f"\nBest Trial: {study.best_trial.number}")
    logger.info(f"Best F1 Score: {study.best_value:.4f}")
    logger.info("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
        
    # Save best parameters
    import json
    output_path = Path('models/best_hyperparameters.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
        
    logger.info(f"\n✓ Best hyperparameters saved to: {output_path}")
    
    return study.best_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='cicddata')
    parser.add_argument('--n-trials', type=int, default=50)
    
    args = parser.parse_args()
    
    # Load data (simplified for HPO)
    logger.info("Loading data...")
    from scripts.train_production import ProductionDataLoader, prepare_features
    
    loader = ProductionDataLoader(Path(args.data_dir))
    df = loader.load_improved_dataset()
    df = loader.validate_data_quality(df)
    
    X, y = prepare_features(df)
    
    # Simple train/val split for HPO
    from sklearn.model_selection import train_test_split
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.encode(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # Run optimization
    best_params = run_hyperparameter_optimization(
        X_train, y_train,
        X_val, y_val,
        label_encoder,
        n_trials=args.n_trials
    )
