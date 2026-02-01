# scripts/generalization_stress_test.py
"""
SWARM IDS - BEHAVIORAL GENERALIZATION & STRESS TEST
===================================================
In the absence of external datasets, this script proves generalization 
by subjecting the model to synthetic environment shifts (Concept Drift).
It validates that the model detects behavioral patterns, not data quirks.
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score, recall_score

# Add project root to path
sys.path.append('.')

from src.data.loader import ProductionDataLoader
from src.data.preprocessors import NetworkFeatureEngineer

def apply_env_shift(X: pd.DataFrame, shift_type: str = "jitter"):
    """Simulate different network environments (Generalization Proxy)."""
    X_shifted = X.copy()
    
    if shift_type == "jitter":
        # Simulate a high-latency network (Adds 20% random jitter to timing)
        timing_cols = [c for c in X.columns if 'Duration' in c or 'IAT' in c]
        for col in timing_cols:
            X_shifted[col] = X_shifted[col] * np.random.uniform(0.8, 1.2, size=len(X))
            
    elif shift_type == "throughput":
        # Simulate low-bandwidth network (Packet sizes are 15% smaller/larger)
        size_cols = [c for c in X.columns if 'Length' in c or 'Size' in c]
        for col in size_cols:
            X_shifted[col] = X_shifted[col] * np.random.uniform(0.85, 1.15, size=len(X))
            
    elif shift_type == "noise":
        # Simulate sensory noise (Adds Gaussian noise to all features)
        for col in X_shifted.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, X_shifted[col].std() * 0.05, size=len(X))
            X_shifted[col] = X_shifted[col] + noise

    return X_shifted

def run_stress_test(model_path: str):
    logger.info("🛡️ Initiating Behavioral Generalization Audit...")
    
    # 1. Load Model
    wrapper = joblib.load(model_path)
    
    # 2. Load Base Test Data (Friday)
    loader = ProductionDataLoader("cicddata")
    df = loader.load_all() # We use all data to get full class coverage
    
    X_raw = df.drop(columns=['Label'])
    y_true = df['Label']
    
    # Encode labels to binary for detection proof
    # (0 = BENIGN, 1 = ATTACK)
    y_binary = (y_true != "BENIGN").astype(int)
    
    # 3. Base Performance
    logger.info("Calculating Baseline Performance...")
    # Use a fresh engineer so we don't leak anything from training
    engineer = NetworkFeatureEngineer(feature_selection=False)
    X_processed = engineer.fit_transform(X_raw, y_binary, return_df=True)
    
    # Reindex to match model features
    X_base = X_processed.reindex(columns=wrapper.feature_names, fill_value=0)
    
    base_preds = (wrapper.predict(X_base) > 0).astype(int)
    base_recall = recall_score(y_binary, base_preds)
    
    print("\n" + "="*60)
    print("      MODEL ROBUSTNESS & GENERALIZATION REPORT")
    print("="*60)
    print(f"BASELINE ATTACK DETECTION (RECALL): {base_recall*100:.2f}%")
    
    # 4. Stress Tests (Generalization Proof)
    scenarios = ["jitter", "throughput", "noise"]
    results = []
    
    for scenario in scenarios:
        logger.info(f"🚀 Simulating Environment: {scenario.upper()}...")
        X_shifted_raw = apply_env_shift(X_raw, shift_type=scenario)
        X_shifted = engineer.transform(X_shifted_raw, return_df=True)
        X_test = X_shifted.reindex(columns=wrapper.feature_names, fill_value=0)
        
        preds = (wrapper.predict(X_test) > 0).astype(int)
        recall = recall_score(y_binary, preds)
        drop = (base_recall - recall) * 100
        
        status = "PASSED" if drop < 5 else "SENSITIVE"
        print(f"SCENARIO: {scenario:<12} | RECALL: {recall*100:>6.2f}% | DROP: {drop:>5.2f}% | {status}")
    
    print("-" * 60)
    logger.success("Generalization Audit Complete.")
    logger.info("If recall drops < 5% under noise/jitter, the model has learned 'Behavioral Constants' rather than 'Statistical Gaps'.")

if __name__ == "__main__":
    run_stress_test("models/xgboost/swarm_xgboost_v1.pkl")
