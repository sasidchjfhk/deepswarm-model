# src/training/metrics.py
"""
Metrics calculation for model evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        average: Averaging method for multi-class
        
    Returns:
        metrics: Dictionary of metric values
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics.
    
    Returns:
        per_class_metrics: Dict mapping class name to metrics
    """
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        
    return per_class
