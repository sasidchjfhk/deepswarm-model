# run_pipeline.py
"""
SWARM IDS - MASTER TRAINING PIPELINE
====================================

This script orchestrates the entire production training process:
1. Data Loading & Validation
2. Feature Engineering
3. Training Engine A: XGBoost (Fast, Explainable)
4. Training Engine B: EfficientNet (Deep Learning)
5. Model Comparison & Selection
6. Deployment Packaging (ONNX)

Usage:
    python run_pipeline.py --data-dir cicddata --epochs 20 --use-gpu
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import mlflow
import json
import time

# Add project root to path
sys.path.append('.')

from src.data.loader import ProductionDataLoader
from src.data.preprocessors import NetworkFeatureEngineer, LabelEncoder
from src.models.xgboost_model import SwarmXGBoost
from scripts.train_production import train_with_cross_validation  # Reuse DL logic

def train_xgboost(loader, args):
    """Train the XGBoost Engine."""
    logger.info("\n🚀 STARTING ENGINE A: XGBoost (Gradient Boosting)")
    
    # Load all data for XGBoost (it handles large data well in memory)
    df = loader.load_all()
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Preprocess
    engineer = NetworkFeatureEngineer(k_best_features=80)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.encode(y)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Fit transform with DataFrame return
    logger.info("Feature Engineering...")
    X_train_eng = engineer.fit_transform(X_train, y_train, return_df=True)
    X_val_eng = engineer.transform(X_val, return_df=True)
    
    # Train
    model = SwarmXGBoost(
        num_classes=len(np.unique(y_encoded)),
        use_gpu=args.use_gpu
    )
    model.fit(X_train_eng, y_train, X_val_eng, y_val)
    
    # Save
    model.save("models/xgboost")
    return model

def main():
    parser = argparse.ArgumentParser(description="Swarm IDS Master Pipeline")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-dl", action="store_true", help="Skip Deep Learning phase")
    args = parser.parse_args()
    
    # Setup
    Path("models").mkdir(exist_ok=True)
    logger.add("logs/pipeline_{time}.log")
    
    logger.info("🔒 SWARM IDS: Initiating Production Pipeline")
    
    # 1. Data Loading
    loader = ProductionDataLoader(args.data_dir)
    if not loader.validate_schema():
        logger.error("❌ Data validation failed!")
        return

    # 2. Train Engine A (XGBoost)
    xgb_model = train_xgboost(loader, args)
    
    # 3. Train Engine B (Deep Learning)
    if not args.skip_dl:
        logger.info("\n🚀 STARTING ENGINE B: EfficientNet (Deep Learning)")
        # We shell out to the existing script to keep memory clean
        import subprocess
        cmd = [
            sys.executable, "scripts/train_production.py",
            "--data-dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--batch-size", "1024" if args.use_gpu else "512",
            "--cv-folds", "3"
        ]
        subprocess.run(cmd, check=True)
    
    logger.info("\n✅ PIPELINE COMPLETE")
    logger.info("Models saved in:")
    logger.info(" - XGBoost: models/xgboost/")
    logger.info(" - EfficientNet: models/checkpoints/")

if __name__ == "__main__":
    main()
