# scripts/cross_validate.py
"""
SWARM IDS - CROSS-DATASET VALIDATION
====================================
Validates the CICIDS2017-trained model on the UNSW-NB15 dataset.
This proves generalization across different network environments.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from loguru import logger
from sklearn.metrics import classification_report

# Add project root to path
sys.path.append('.')

from src.data.preprocessors import LabelEncoder

# UNSW-NB15 Sample URL (Kaggle or Github mirror)
UNSW_SAMPLE_URL = "https://raw.githubusercontent.com/697966754/UNSW-NB15/master/UNSW_NB15_training-set.csv"

def download_unsw_sample(dest_path: Path):
    """Download a small sample of UNSW-NB15 for validation."""
    if dest_path.exists():
        logger.info(f"UNSW sample already exists at {dest_path}")
        return True
    
    logger.info("Downloading UNSW-NB15 sample (approx 10MB)...")
    try:
        response = requests.get(UNSW_SAMPLE_URL, timeout=30)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        logger.success("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download UNSW-NB15: {e}")
        return False

def map_unsw_to_cicids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW-NB15 features to CICIDS-like features expected by the model.
    Note: This is an approximation since network features differ across datasets.
    """
    logger.info("Mapping UNSW-NB15 features to behavioral standard...")
    
    # Mapping logic: Translate UNSW concepts to CICIDS concepts
    mapping = {
        'dur': 'Flow Duration',
        'spkts': 'Total Fwd Packets',
        'dpkts': 'Total Backward Packets',
        'sbytes': 'Total Length of Fwd Packets',
        'dbytes': 'Total Length of Bwd Packets',
        'rate': 'Flow Packets/s',
        'smean': 'Fwd Packet Length Mean',
        'dmean': 'Bwd Packet Length Mean',
        'sinpkt': 'Fwd IAT Mean',
        'dinpkt': 'Bwd IAT Mean',
        # Flag-like estimation (approximation)
        'tcprtt': 'tcp_rtt_proxy'
    }
    
    # Rename
    df_mapped = df.rename(columns=mapping)
    
    # Synthetic synthesis of missing features if necessary
    # (Just enough to satisfy the model's feature requirement)
    df_mapped['Average Packet Size'] = (df_mapped['Total Length of Fwd Packets'] + df_mapped['Total Length of Bwd Packets']) / \
                                       (df_mapped['Total Fwd Packets'] + df_mapped['Total Backward Packets'] + 1e-9)
    
    # Create flag ratios (Approximation based on service types)
    df_mapped['syn_ratio'] = 0.0
    df_mapped['rst_ratio'] = 0.0
    df_mapped.loc[df['proto'] == 'tcp', 'syn_ratio'] = 0.1 # Heuristic for general TCP
    
    return df_mapped

def run_cross_validation(model_path: str):
    data_dir = Path("data/unsw")
    data_dir.mkdir(parents=True, exist_ok=True)
    unsw_file = data_dir / "unsw_sample.csv"
    
    if not download_unsw_sample(unsw_file):
        logger.error("Required data missing. Execution aborted.")
        return

    # 1. Load Model
    logger.info(f"Loading Production Model: {model_path}")
    wrapper = joblib.load(model_path)
    
    # 2. Load and Map UNSW Data
    df = pd.read_csv(unsw_file)
    logger.info(f"Loaded {len(df):,} samples from UNSW-NB15")
    
    # Map labels to binary (Benign vs Attack) for easier generalization proof
    # In UNSW, 'label' is 0 for Benign, 1 for Attack
    y_true_binary = df['label'].values
    
    df_mapped = map_unsw_to_cicids(df)
    
    # 3. Predict
    # We need to ensure the columns match the model's training features
    # The SwarmXGBoost wrapper expects a DataFrame with specific columns
    logger.info("Predicting on Cross-Dataset...")
    
    # Use the model's feature engineering logic (it handles missing columns by ignoring or 0-filling)
    # We use the raw mapped data
    try:
        from src.data.preprocessors import NetworkFeatureEngineer
        # We simulate the transformation to satisfy the XGBoost feature names
        engineer = NetworkFeatureEngineer(feature_selection=False)
        # We need a subset of features that actually overlap
        X_processed = engineer.fit_transform(df_mapped, y_true_binary, return_df=True)
        
        # Ensure feature order matches original model
        model_features = wrapper.feature_names
        # Reindex to match model features, filling missing with 0
        X_final = X_processed.reindex(columns=model_features, fill_value=0)
        
        preds = wrapper.predict(X_final)
        
        # 4. Results (Binary Detection)
        # Map our multiclass predictions back to binary (0 = Benign, >0 = Attack)
        preds_binary = (preds > 0).astype(int)
        
        print("\n" + "="*60)
        print("      CROSS-DATASET VALIDATION: UNSW-NB15")
        print("="*60)
        print(classification_report(y_true_binary, preds_binary, target_names=["BENIGN", "ATTACK"]))
        
        accuracy = np.mean(preds_binary == y_true_binary) * 100
        logger.success(f"Final Generalization Accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cross_validation("models/xgboost/swarm_xgboost_v1.pkl")
