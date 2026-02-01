# src/models/efficientnet.py
"""
EfficientNet-based model for network intrusion detection.

Adapted from: https://github.com/lukemelas/EfficientNet-PyTorch
Optimized for tabular data and production deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class Swish(nn.Module):
    """Swish activation function (used in EfficientNet)."""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Adaptively recalibrates channel-wise feature responses.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        # x: [B, C, L]
        
        # Global pooling
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Scale
        y = y.view(b, c, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck block.
    
    Core building block of EfficientNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.has_expansion = (expand_ratio != 1)
        
        if self.has_expansion:
            self.expand_conv = nn.Conv1d(
                in_channels, expanded_channels, 1, bias=False
            )
            self.expand_bn = nn.BatchNorm1d(expanded_channels)
            
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            expanded_channels, expanded_channels,
            kernel_size, stride, padding=kernel_size//2,
            groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm1d(expanded_channels)
        
        # Squeeze-and-Excitation
        self.has_se = (se_ratio is not None) and (se_ratio > 0)
        if self.has_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_block = SEBlock(expanded_channels, reduction=se_channels)
            
        # Output phase
        self.project_conv = nn.Conv1d(
            expanded_channels, out_channels, 1, bias=False
        )
        self.project_bn = nn.BatchNorm1d(out_channels)
        
        # Activation
        self.swish = Swish()
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.has_expansion:
            x = self.swish(self.expand_bn(self.expand_conv(x)))
            
        # Depthwise
        x = self.swish(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze-Excitation
        if self.has_se:
            x = self.se_block(x)
            
        # Project
        x = self.project_bn(self.project_conv(x))
        
        # Skip connection with drop connect
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.training and self.drop_connect_rate > 0:
                x = self._drop_connect(x, self.drop_connect_rate)
            x = x + identity
            
        return x
        
    def _drop_connect(self, x, drop_rate):
        """Stochastic depth (drop connect)."""
        if not self.training:
            return x
            
        keep_prob = 1 - drop_rate
        batch_size = x.shape[0]
        
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, 1, 1],
            dtype=x.dtype,
            device=x.device
        )
        binary_tensor = torch.floor(random_tensor)
        
        return x / keep_prob * binary_tensor


class EfficientNetIDS(nn.Module):
    """
    EfficientNet adapted for network intrusion detection.
    
    Architecture:
    - Stem: Initial projection layer
    - Blocks: Stacked MBConv blocks
    - Head: Classification head with dropout
    
    Input: [batch, features] → reshaped to [batch, channels, length]
    Output: [batch, num_classes]
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int = 15,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2
    ):
        """
        Args:
            num_features: Input feature dimension
            num_classes: Number of attack classes
            width_mult: Width multiplier (channels)
            depth_mult: Depth multiplier (layers)
            dropout_rate: Dropout rate in classification head
            drop_connect_rate: Drop connect rate in blocks
        """
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Reshape features to pseudo-1D-image: [B, C, L]
        # Treat features as "channels" with length 1, then expand
        self.feature_reshape = nn.Linear(num_features, 64)
        
        # Stem
        stem_channels = self._round_channels(32, width_mult)
        self.stem_conv = nn.Conv1d(1, stem_channels, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(stem_channels)
        self.swish = Swish()
        
        # Building blocks
        block_configs = [
            # (channels, layers, kernel, stride, expand_ratio)
            (16, 1, 3, 1, 1),
            (24, 2, 3, 2, 6),
            (40, 2, 5, 2, 6),
            (80, 3, 3, 2, 6),
            (112, 3, 5, 1, 6),
            (192, 4, 5, 2, 6),
            (320, 1, 3, 1, 6),
        ]
        
        self.blocks = nn.ModuleList([])
        in_channels = stem_channels
        total_blocks = sum([cfg[1] for cfg in block_configs])
        block_idx = 0
        
        for channels, num_layers, kernel, stride, expand in block_configs:
            out_channels = self._round_channels(channels, width_mult)
            num_layers = self._round_repeats(num_layers, depth_mult)
            
            for i in range(num_layers):
                # Calculate drop connect rate (increases with depth)
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand,
                        drop_connect_rate=drop_rate
                    )
                )
                
                block_idx += 1
                
            in_channels = out_channels
            
        # Head
        head_channels = self._round_channels(1280, width_mult)
        self.head_conv = nn.Conv1d(in_channels, head_channels, 1, bias=False)
        self.head_bn = nn.BatchNorm1d(head_channels)
        
        # Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_features]
            
        Returns:
            logits: [batch, num_classes]
        """
        
        # Reshape to [B, 1, F] for 1D conv
        x = self.feature_reshape(x)  # [B, 64]
        x = x.unsqueeze(1)  # [B, 1, 64]
        
        # Stem
        x = self.swish(self.stem_bn(self.stem_conv(x)))
        
        # Blocks
        for block in self.blocks:
            x = block(x)
            
        # Head
        x = self.swish(self.head_bn(self.head_conv(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
        
    @staticmethod
    def _round_channels(channels: int, width_mult: float) -> int:
        """Round channels to nearest multiple of 8."""
        channels *= width_mult
        new_channels = max(8, int(channels + 4) // 8 * 8)
        if new_channels < 0.9 * channels:
            new_channels += 8
        return int(new_channels)
        
    @staticmethod
    def _round_repeats(repeats: int, depth_mult: float) -> int:
        """Round number of layer repeats."""
        return int(math.ceil(depth_mult * repeats))
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def create_efficientnet_ids(
    num_features: int,
    num_classes: int = 15,
    model_size: str = "b0"
) -> EfficientNetIDS:
    """
    Create EfficientNet-IDS model with predefined sizes.
    
    Args:
        num_features: Input feature dimension
        num_classes: Number of classes
        model_size: One of ["b0", "b1", "b2", "b3"]
        
    Returns:
        model: EfficientNet-IDS instance
    """
    
    configs = {
        "b0": {"width": 1.0, "depth": 1.0, "dropout": 0.2},
        "b1": {"width": 1.0, "depth": 1.1, "dropout": 0.2},
        "b2": {"width": 1.1, "depth": 1.2, "dropout": 0.3},
        "b3": {"width": 1.2, "depth": 1.4, "dropout": 0.3},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
        
    cfg = configs[model_size]
    
    return EfficientNetIDS(
        num_features=num_features,
        num_classes=num_classes,
        width_mult=cfg["width"],
        depth_mult=cfg["depth"],
        dropout_rate=cfg["dropout"]
    )
