# scripts/train_xgboost.py
"""
Train Production XGBoost Model for Swarm IDS.

Command:
    python scripts/train_xgboost.py --data-dir cicddata --use-gpu
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

# Add project root to path
sys.path.append('.')

from src.data.loader import ProductionDataLoader
from src.data.preprocessors import NetworkFeatureEngineer, LabelEncoder
from src.models.xgboost_model import SwarmXGBoost

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Swarm IDS")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CICIDS dataset")
    parser.add_argument("--output-dir", type=str, default="models/xgboost", help="Save path")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()
    
    # 1. Load Data
    loader = ProductionDataLoader(args.data_dir)
    df = loader.load_all()
    
    # 2. Preprocessing
    logger.info("🛠️ Preparing data...")
    feature_engineer = NetworkFeatureEngineer(num_features=80) # Use all useful features
    label_encoder = LabelEncoder()
    
    # 3. Train/Test Split (Stratified 80/20)
    # Note: In production we use cross-validation, here we do a simple split 
    # to demonstrate the XGBoost pipeline quickly.
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Encode Labels
    y_encoded = label_encoder.encode(y)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Fit Feature Engineering (on Train only)
    logger.info("🔧 Fitting feature engineer...")
    X_train_eng = feature_engineer.fit_transform(X_train, y_train)
    X_val_eng = feature_engineer.transform(X_val)
    
    # Convert back to DataFrame to keep feature names (XGBoost likes this)
    feature_names = feature_engineer.selected_features
    X_train_df = pd.DataFrame(X_train_eng, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_eng, columns=feature_names)
    
    # 4. Train XGBoost
    num_classes = len(np.unique(y_encoded))
    model = SwarmXGBoost(
        num_classes=num_classes, 
        use_gpu=args.use_gpu,
        params={'max_depth': 10, 'learning_rate': 0.05} # Fine-tuned for IDS
    )
    
    model.fit(X_train_df, y_train, X_val_df, y_val)
    
    # 5. Evaluate
    logger.info("📊 Evaluating...")
    preds = model.predict(X_val_df)
    f1 = f1_score(y_val, preds, average='macro')
    
    logger.info(f"🏆 Validation F1 Score: {f1:.4f}")
    
    # detailed report
    report = classification_report(
        y_val, preds, 
        target_names=[label_encoder.idx_to_label[i] for i in range(num_classes)],
        digits=4
    )
    print("\n" + report)
    
    # 6. Save
    model.save(args.output_dir)
    logger.info(f"✅ Production Model Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
