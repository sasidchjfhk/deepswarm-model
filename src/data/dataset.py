# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Callable, Tuple
from loguru import logger

class IDSDataset(Dataset):
    """
    PyTorch Dataset for intrusion detection.
    
    Features:
    - Memory-efficient (lazy loading option)
    - Augmentation support
    - Class weights for imbalanced data
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        """
        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            transform: Optional transform function
            augment: Apply augmentation (training only)
        """
        # Convert to float32 for PyTorch
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        
        # Apply augmentation (only for training)
        X = self.X[idx]
        y = self.y[idx]
        
        # SAFETY: Replace NaN/Inf with 0.0 to prevent training crash
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.transform:
            X = self.transform(X)
            
        return x, y
        
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data augmentation for network features.
        
        Techniques:
        - Gaussian noise injection
        - Feature dropout
        """
        
        # Gaussian noise (small magnitude)
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
        # Random feature dropout (10% of features)
        if torch.rand(1) < 0.2:
            mask = torch.rand(x.shape) > 0.1
            x = x * mask
            
        return x
        
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset.
        
        Returns:
            weights: [num_classes] tensor
        """
        
        # Count samples per class
        unique, counts = torch.unique(self.y, return_counts=True)
        
        # Map counts to all classes (some might be missing in this split)
        # We need to know total classes for safety, but here we estimate from what we have
        # Or better, this returns weights for classes present.
        
        # Inverse frequency weighting
        total = len(self.y)
        # Avoid division by zero
        weights = total / (len(unique) * counts.float())
        
        # Normalize
        weights = weights / weights.sum() * len(unique)
        
        # Create a full weight tensor if we knew num_classes, but for now 
        # this returns weights aligned with 'unique' classes found.
        # In a real trainer, we'd map these to the full class index.
        # For simplicity here, we assume standard class indices.
        
        full_weights = torch.ones(15) # 15 classes default
        full_weights[unique] = weights
        
        logger.info(f"Class weights computed for {len(unique)} classes present")
        
        return full_weights


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0, # Default to 0 for Windows compatibility in dev
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create training and validation dataloaders.
    
    Best practices:
    - Shuffle training data
    - DON'T shuffle validation data (for reproducibility)
    """
    
    # Create datasets
    train_dataset = IDSDataset(X_train, y_train, augment=True)
    val_dataset = IDSDataset(X_val, y_val, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # CRITICAL for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,  # NEVER shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(
        f"Created dataloaders: "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches"
    )
    
    return train_loader, val_loader, train_dataset.get_class_weights()
