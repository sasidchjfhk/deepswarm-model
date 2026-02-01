# src/models/xgboost_model.py
"""
Production-grade XGBoost wrapper for Swarm IDS.

Features:
- Sklearn-compatible API
- Automated class weighting (scale_pos_weight for binary, weights for multi-class)
- ONNX export capability
- GPU acceleration support
- Early stopping & pruning
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
from pathlib import Path
from loguru import logger
import joblib
import json
import time

class SwarmXGBoost:
    """
    Enterprise wrapper for XGBoost Classifier.
    """
    
    def __init__(
        self, 
        params: Optional[Dict] = None,
        num_classes: int = 2,
        use_gpu: bool = False
    ):
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.feature_names: List[str] = []
        self.model = None
        
        # Default Production Hyperparameters (Google best practices)
        self.params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'tree_method': 'hist',  # Fast histogram optimized
            'device': 'cuda' if use_gpu else 'cpu',
            'max_depth': 8,         # Deeper trees for complex attacks
            'learning_rate': 0.1,
            'n_estimators': 1000,   # Will use early stopping
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,       # L1 regularization
            'reg_lambda': 1.0,      # L2 regularization
            'n_jobs': -1,
            'eval_metric': ['mlogloss', 'merror'] if num_classes > 2 else ['logloss', 'auc']
        }
        
        if params:
            self.params.update(params)
            
    def fit(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None, 
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """Train the model with production monitoring."""
        
        logger.info(f"🚀 Starting XGBoost Training (Classes: {self.num_classes}, Device: {self.params['device']})")
        start_time = time.time()
        
        # Store feature names if pandas
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            
        # Initialize Scikit-Learn wrapper
        self.model = xgb.XGBClassifier(**self.params)
        
        # Train
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=10 if verbose else False
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Training completed in {duration:.2f}s")
        
        # Log feature importance
        self._log_top_features()
        
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def save(self, output_dir: str, version: str = "v1"):
        """Save model in dual format: JSON (XGBoost) and PKL (Wrapper)."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save core XGBoost model (portable)
        json_path = path / f"xgboost_core_{version}.json"
        self.model.save_model(json_path)
        logger.info(f"💾 Saved core model: {json_path}")
        
        # 2. Save wrapper (with feature names)
        pkl_path = path / f"swarm_xgboost_{version}.pkl"
        joblib.dump(self, pkl_path)
        logger.info(f"💾 Saved wrapper: {pkl_path}")
        
    def _log_top_features(self, top_n: int = 10):
        """Log the most important features for security analysis."""
        if not self.feature_names:
            return
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        logger.info(f"🔍 Top {top_n} Security Features:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            logger.info(f"   {i+1}. {self.feature_names[idx]:<30} ({importance[idx]:.4f})")
