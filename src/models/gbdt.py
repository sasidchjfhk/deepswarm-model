# src/models/gbdt.py
"""
Lightweight GBDT (Gradient Boosting Decision Tree) model for Swarm IDS.

Uses LightGBM for state-of-the-art performance on tabular data with 
minimal CPU/memory footprint.
"""

import lightgbm as lgb
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
from loguru import logger

class IDSModel:
    """
    Production wrapper for LightGBM model.
    
    Optimized for:
    - High accuracy on tabular network data
    - Low latency inference
    - Low memory usage
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 15,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'device': 'cpu',
            'seed': 42
        }
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LightGBM model."""
        logger.info("Initializing LightGBM training...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        valid_names = []
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        logger.info("✓ LightGBM training complete")

    def predict(self, X):
        """Predict class indices."""
        probs = self.model.predict(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict(X)

    def save(self, path: Path):
        """Save model to disk using joblib (lightweight)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"✓ Model loaded from {path}")
        return instance
