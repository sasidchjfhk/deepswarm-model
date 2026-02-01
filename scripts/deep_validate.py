# scripts/deep_validate.py
"""
SWARM IDS - DEEP VALIDATION & AUDIT
==================================
Performs per-class analysis, confusion matrix generation, 
and Benign False Positive Rate (FPR) calculation.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger

# Add project root to path
sys.path.append('.')

from src.data.loader import ProductionDataLoader
from src.data.preprocessors import NetworkFeatureEngineer, LabelEncoder

def deep_audit(data_dir: str, model_path: str):
    logger.info("Starting Deep Production Validation...")
    
    # 1. Load Model & Components
    if not Path(model_path).exists():
        logger.error("Model not found!")
        return
        
    wrapper = joblib.load(model_path)
    loader = ProductionDataLoader(data_dir)
    
    # 2. Load Evaluation Data (Combine ALL to ensure all classes are present)
    logger.info("Loading Complete Test Matrix (Monday-Friday)...")
    df_full = loader.load_all()
    X_raw = df_full.drop(columns=['Label'])
    y_str = df_full['Label']
    
    # 3. Process Labels
    label_encoder = LabelEncoder()
    y_true = label_encoder.encode(y_str)
    
    # 4. Feature Engineering
    engineer = NetworkFeatureEngineer(feature_selection=False)
    X_processed = engineer.fit_transform(X_raw, y_true, return_df=True)
    
    # 5. Predict
    logger.info("Generating Predictions for All Attack Types...")
    y_pred = wrapper.predict(X_processed)
    
    # 6. Detailed Report
    target_names = [label_encoder.idx_to_label[i] for i in range(len(np.unique(y_true)))]
    
    print("\n" + "="*60)
    print("      PER-CLASS PERFORMANCE REPORT (GLOBAL)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # 7. BENIGN False Positive Rate
    benign_idx = -1
    for idx, name in label_encoder.idx_to_label.items():
        if name.upper() == "BENIGN":
            benign_idx = idx
            break
            
    if benign_idx != -1:
        benign_mask = (y_true == benign_idx)
        total_benign = np.sum(benign_mask)
        false_positives = np.sum((y_true == benign_idx) & (y_pred != benign_idx))
        fpr = (false_positives / total_benign) * 100
        
        print(f"\n[SECURITY] BENIGN FALSE POSITIVE RATE (FPR): {fpr:.4f}%")
        print(f"   (False Alarms: {false_positives:,} out of {total_benign:,} clean packets)")

    # 8. Feature Importance Visualization
    logger.info("Generating Feature Importance Plot...")
    importance = wrapper.model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Security Features (Swarm XGBoost)")
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X_processed.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('logs/feature_importance.png')
    logger.info("Saved plot to logs/feature_importance.png")

if __name__ == "__main__":
    deep_audit("cicddata", "models/xgboost/swarm_xgboost_v1.pkl")
