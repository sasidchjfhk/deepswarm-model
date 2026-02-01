# src/deployment/serialization.py
"""
Lightweight model serialization and inference.
"""

import joblib
from pathlib import Path
from loguru import logger
import numpy as np
import time

class ModelInference:
    """
    Lightweight inference wrapper using Joblib.
    """
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        logger.info(f"Loading lightweight model from {model_path}...")
        self.model = joblib.load(model_path)
        logger.info("✓ Model loaded successfully")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run classification."""
        # LightGBM models loaded via joblib are the Booster objects
        # or sklearn wrappers. If it's a Booster, we use .predict().
        probs = self.model.predict(X)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            return probs.argmax(axis=1)
        return (probs > 0.5).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities."""
        return self.model.predict(X)
