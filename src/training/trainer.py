# src/training/trainer.py
"""
Production-grade training loop with all best practices.

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Experiment tracking (MLflow)
- Gradient clipping
- Advanced optimizers (AdamW with weight decay)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import time

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - experiment tracking disabled")

from .metrics import calculate_metrics


class IDSTrainer:
    """
    Production trainer for IDS models.
    
    Implements:
    - Automatic mixed precision (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Model checkpointing
    - MLflow experiment tracking
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir: Path = Path("models/checkpoints"),
        experiment_name: str = "swarm-ids",
        use_mlflow: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        # Mixed precision
        self.mixed_precision = mixed_precision and device == "cuda"
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.global_step = 0
        
        # MLflow
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                    
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                        
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
            # Metrics
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for inputs, targets in tqdm(self.val_loader, desc="Validation"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_preds),
            average='weighted'
        )
        
        metrics['val_loss'] = val_loss
        
        return metrics
        
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_every: int = 10
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs
        """
        
        if self.use_mlflow:
            with mlflow.start_run():
                self._train_loop(num_epochs, early_stopping_patience, save_every)
        else:
            self._train_loop(num_epochs, early_stopping_patience, save_every)
            
    def _train_loop(self, num_epochs, early_stopping_patience, save_every):
        """Internal training loop."""
        
        # Log hyperparameters
        if self.use_mlflow:
            mlflow.log_params({
                'model': self.model.__class__.__name__,
                'optimizer': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.train_loader.batch_size,
                'epochs': num_epochs,
                'device': self.device,
                'mixed_precision': self.mixed_precision,
                'gradient_clip': self.gradient_clip_val,
                'gradient_accumulation_steps': self.gradient_accumulation_steps
            })
            
        patience_counter = 0
        train_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
                    
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            metrics['epoch_time'] = time.time() - epoch_start
            
            # Log metrics
            if self.use_mlflow:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=epoch)
                    
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_f1={val_metrics['f1_score']:.4f}, "
                f"lr={metrics['learning_rate']:.6f}"
            )
            
            # Model checkpointing
            is_best_f1 = val_metrics['f1_score'] > self.best_val_f1
            is_best_loss = val_metrics['val_loss'] < self.best_val_loss
            
            if is_best_f1:
                self.best_val_f1 = val_metrics['f1_score']
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', metrics)
                logger.info(f"✓ New best model saved (F1: {self.best_val_f1:.4f})")
            else:
                patience_counter += 1
                
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', metrics)
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience: {early_stopping_patience})"
                )
                break
                
        # Training complete
        total_time = time.time() - train_start_time
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)
        
        # Log final model
        if self.use_mlflow:
            try:
                mlflow.pytorch.log_model(self.model, "model")
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")
            
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"✓ Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer(
    model,
    train_loader,
    val_loader,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    use_focal_loss: bool = True,
    **config
):
    """Factory function to create configured trainer."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Loss function
    if use_focal_loss:
        from .losses import FocalLoss
        criterion = FocalLoss(
            alpha=class_weights.to(device) if class_weights is not None else None,
            gamma=config.get('focal_gamma', 2.0)
        )
        logger.info("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        logger.info("Using Cross Entropy Loss")
    
    # Optimizer - AdamW with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        betas=config.get('betas', (0.9, 0.999))
    )
    
    # Learning rate scheduler
    scheduler_type = config.get('scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        logger.info("Using Cosine Annealing LR")
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('learning_rate', 1e-3),
            steps_per_epoch=len(train_loader),
            epochs=config.get('epochs', 100)
        )
        logger.info("Using OneCycle LR")
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        logger.info("Using ReduceLROnPlateau")
    else:
        scheduler = None
        logger.info("No LR scheduler")
        
    # Create trainer
    trainer = IDSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=config.get('mixed_precision', True),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        checkpoint_dir=Path(config.get('checkpoint_dir', 'models/checkpoints')),
        experiment_name=config.get('experiment_name', 'swarm-ids'),
        use_mlflow=config.get('use_mlflow', True)
    )
    
    return trainer
