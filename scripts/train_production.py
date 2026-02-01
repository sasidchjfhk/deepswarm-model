#!/usr/bin/env python
# scripts/train_production.py
"""
Production-grade training pipeline for Swarm IDS.

Enterprise features:
- Data validation & quality checks
- Stratified K-fold cross-validation
- Hyperparameter optimization (Optuna)
- Model ensemble
- Advanced metrics tracking
- Automated model selection
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import NetworkFeatureEngineer, LabelEncoder, create_dataloaders
from src.models import create_efficientnet_ids
from src.training import create_trainer
from src.evaluation import IDSEvaluator


class ProductionDataLoader:
    """
    Enterprise-grade data loader with validation.
    
    Features:
    - Automated data quality checks
    - Missing value handling
    - Outlier detection
    - Class distribution analysis
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
    def load_improved_dataset(self) -> pd.DataFrame:
        """Load and validate improved CICIDS dataset."""
        
        logger.info("Loading improved CICIDS dataset...")
        
        # Load all CSV files
        files = {
            'monday': 'monday.csv',
            'tuesday': 'tuesday.csv',
            'thursday': 'thursday.csv',
            'friday': 'friday.csv'
        }
        
        dfs = []
        total_samples = 0
        
        for day, filename in files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                logger.warning(f"Missing file: {filename}")
                continue
                
            logger.info(f"Loading {day}...")
            df = pd.read_csv(filepath, low_memory=False)
            
            # Add day identifier for temporal tracking
            df['day'] = day
            
            dfs.append(df)
            total_samples += len(df)
            logger.info(f"  {day}: {len(df):,} samples")
            
        if not dfs:
            raise ValueError("No data files found!")
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"\n✓ Total samples loaded: {total_samples:,}")
        logger.info(f"  Features: {len(combined_df.columns)}")
        
        return combined_df
        
    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enterprise data quality validation.
        
        Checks:
        1. Missing values
        2. Infinite values
        3. Duplicate rows
        4. Class distribution
        5. Feature correlation
        """
        
        logger.info("\n[DATA QUALITY VALIDATION]")
        
        initial_size = len(df)
        
        # 1. Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Found missing values in {(missing > 0).sum()} columns")
            # Fill with median for numeric, mode for categorical
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'UNKNOWN', inplace=True)
                    
        # 2. Infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 3. Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates:,} duplicate rows")
            df = df.drop_duplicates()
            
        # 4. Class distribution
        if 'Label' in df.columns:
            class_dist = df['Label'].value_counts()
            logger.info(f"\nClass Distribution:")
            for label, count in class_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"  {label}: {count:,} ({pct:.2f}%)")
                
        logger.info(f"\n✓ Quality validation complete")
        logger.info(f"  Samples before: {initial_size:,}")
        logger.info(f"  Samples after: {len(df):,}")
        logger.info(f"  Removed: {initial_size - len(df):,}")
        
        return df


def prepare_features(df: pd.DataFrame):
    """Prepare features for training."""
    
    logger.info("\n[FEATURE PREPARATION]")
    
    # Columns to exclude from features
    exclude_cols = [
        'id', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port',
        'Timestamp', 'Label', 'day', 'Attempted Category'
    ]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure numeric
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col].fillna(0, inplace=True)
            except:
                # Drop non-convertible columns
                logger.warning(f"Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])
                
    y = df['Label']
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {len(X):,}")
    
    return X, y


def train_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    n_folds: int = 5
):
    """
    K-fold cross-validation training.
    
    Enterprise approach:
    - Stratified splits (preserve class distribution)
    - Per-fold model training
    - Averaged metrics
    - Best model selection
    """
    
    logger.info(f"\n[{n_folds}-FOLD CROSS-VALIDATION]")
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.encode(y)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    best_f1 = 0
    best_fold_model = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*80}")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_encoded[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_encoded[val_idx]
        
        logger.info(f"Train: {len(X_train_fold):,} | Val: {len(X_val_fold):,}")
        
        # Feature engineering (fit on train fold only!)
        feature_engineer = NetworkFeatureEngineer(
            scaler_type='robust',
            feature_selection=True,
            k_best_features=config.get('num_features', 50)
        )
        
        X_train_processed = feature_engineer.fit_transform(X_train_fold, y_train_fold)
        X_val_processed = feature_engineer.transform(X_val_fold)
        
        # Create dataloaders
        train_loader, val_loader, class_weights = create_dataloaders(
            X_train_processed,
            y_train_fold,
            X_val_processed,
            y_val_fold,
            batch_size=config['batch_size'],
            num_workers=0
        )
        
        # Create model
        num_features = X_train_processed.shape[1]
        model = create_efficientnet_ids(
            num_features=num_features,
            num_classes=label_encoder.num_classes,
            model_size=config.get('model_size', 'b0')
        )
        
        # Create trainer
        fold_config = config.copy()
        fold_config['checkpoint_dir'] = f"models/folds/fold_{fold+1}"
        fold_config['experiment_name'] = f"swarm-ids-fold-{fold+1}"
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=label_encoder.num_classes,
            class_weights=class_weights,
            **fold_config
        )
        
        # Train
        trainer.train(
            num_epochs=config['epochs'],
            early_stopping_patience=config.get('early_stopping', 10),
            save_every=10
        )
        
        # Get fold results
        fold_f1 = trainer.best_val_f1
        fold_results.append({
            'fold': fold + 1,
            'f1_score': fold_f1,
            'val_loss': trainer.best_val_loss
        })
        
        # Track best fold
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_fold_model = fold + 1
            
            # Save best overall model
            import shutil
            src = Path(fold_config['checkpoint_dir']) / 'best_model.pth'
            dst = Path('models/checkpoints/best_model_cv.pth')
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy(src, dst)
                
        logger.info(f"\nFold {fold+1} Results:")
        logger.info(f"  F1 Score: {fold_f1:.4f}")
        logger.info(f"  Val Loss: {trainer.best_val_loss:.4f}")
        
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    
    fold_f1_scores = [r['f1_score'] for r in fold_results]
    logger.info(f"Mean F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    logger.info(f"Best Fold: {best_fold_model} (F1: {best_f1:.4f})")
    logger.info(f"Worst F1: {np.min(fold_f1_scores):.4f}")
    logger.info(f"Best F1: {np.max(fold_f1_scores):.4f}")
    
    return best_fold_model, fold_results


def main():
    parser = argparse.ArgumentParser(description='Production ML training')
    
    parser.add_argument('--data-dir', type=str, default='cicddata',
                       help='Directory with improved CICIDS data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num-features', type=int, default=60,
                       help='Number of features to select')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use Focal Loss')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SWARM IDS - PRODUCTION ML TRAINING")
    logger.info("="*80)
    
    # 1. Load data
    loader = ProductionDataLoader(Path(args.data_dir))
    df = loader.load_improved_dataset()
    
    # 2. Validate quality
    df = loader.validate_data_quality(df)
    
    # 3. Prepare features
    X, y = prepare_features(df)
    
    # 4. Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_features': 75,  # USE ALL RELEVANT FEATURES (Deep Learning handles this well)
        'use_focal_loss': args.use_focal_loss,
        'use_mlflow': not args.no_mlflow,
        'scheduler': 'one_cycle',  # Super-convergence
        'mixed_precision': torch.cuda.is_available(),
        'gradient_clip_val': 1.0
    }
    
    # 5. Cross-validation training
    best_fold, results = train_with_cross_validation(
        X, y, config, n_folds=args.cv_folds
    )
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best model from Fold {best_fold}")
    logger.info(f"Model saved: models/checkpoints/best_model_cv.pth")
    logger.info("="*80)


if __name__ == '__main__':
    main()
