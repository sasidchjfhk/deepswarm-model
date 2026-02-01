# src/training/losses.py
"""
Advanced loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    From: https://arxiv.org/abs/1708.02002
    
    Focal Loss = -α(1-p)^γ log(p)
    
    Benefits:
    - Down-weights easy examples
    - Focuses on hard misclassified examples
    - Better for highly imbalanced datasets
    """
    
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (default: 2.0)
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch] class indices
            
        Returns:
            loss: scalar or [batch]
        """
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Combine
        loss = focal_term * ce_loss
        
        # Add Label Smoothing (Hybrid approach)
        # Smooths the target distribution to prevent overfitting on noisy labels
        smoothing = 0.1
        smooth_loss = -inputs.mean(dim=1)  # Uniform distribution prior
        loss = (1 - smoothing) * loss + smoothing * smooth_loss
        
        # Apply alpha weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Prevents overconfidence and can improve generalization.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        
        # True labels (1 - smoothing)
        nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Uniform distribution over other classes (smoothing / num_classes)
        smooth_loss = -log_probs.mean(dim=1)
        
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()
