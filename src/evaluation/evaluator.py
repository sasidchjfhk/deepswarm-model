# src/evaluation/evaluator.py
"""
Production-grade model evaluation with comprehensive metrics.

Follows ML best practices from Google, Meta, and industry leaders.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import json


class IDSEvaluator:
    """
    Comprehensive model evaluator for IDS.
    
    Metrics:
    - Overall: Accuracy, F1, Precision, Recall
    - Per-class: All metrics breakdown
    - ROC curves and AUC (multi-class)
    - Confusion matrix
    - False positive analysis
    - Attack detection rate
    """
    
    def __init__(
        self,
        class_names: List[str],
        output_dir: Path = Path("evaluation_results")
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader,
        device: str = "cuda",
        dataset_name: str = "test"
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Returns:
            metrics: Dictionary with all evaluation metrics
        """
        
        model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Collect predictions
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate all metrics
        metrics = self._calculate_metrics(
            all_targets,
            all_preds,
            all_probs,
            dataset_name
        )
        
        # Generate visualizations
        self._generate_visualizations(
            all_targets,
            all_preds,
            all_probs,
            dataset_name
        )
        
        # Save detailed report
        self._save_report(metrics, dataset_name)
        
        return metrics
        
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        dataset_name: str
    ) -> Dict:
        """Calculate all evaluation metrics."""
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Weighted average (account for class imbalance)
        metrics['precision_weighted'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[0]
        metrics['recall_weighted'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[1]
        metrics['f1_weighted'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]
        
        # Macro average (equal weight to all classes)
        metrics['precision_macro'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )[0]
        metrics['recall_macro'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )[1]
        metrics['f1_macro'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )[2]
        
        # Per-class breakdown
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            
        # False Positive Rate (critical for IDS)
        cm = confusion_matrix(y_true, y_pred)
        
        # FPR = FP / (FP + TN)
        fpr_per_class = {}
        for i in range(self.num_classes):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_per_class[self.class_names[i]] = float(fpr)
            
        metrics['false_positive_rate'] = fpr_per_class
        
        # Overall FPR (benign incorrectly classified as attack)
        benign_idx = 0  # Assuming BENIGN is index 0
        benign_total = cm[benign_idx, :].sum()
        benign_correct = cm[benign_idx, benign_idx]
        benign_false_alarms = benign_total - benign_correct
        metrics['benign_false_positive_rate'] = float(
            benign_false_alarms / benign_total if benign_total > 0 else 0
        )
        
        # Attack Detection Rate
        attack_indices = [i for i in range(self.num_classes) if i != benign_idx]
        attack_total = cm[attack_indices, :].sum()
        attack_detected = cm[attack_indices, :][:, attack_indices].sum()
        metrics['attack_detection_rate'] = float(
            attack_detected / attack_total if attack_total > 0 else 0
        )
        
        # ROC AUC (multi-class, one-vs-rest)
        try:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            # Macro AUC
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true_bin, y_prob, average='macro', multi_class='ovr'
            )
            
            # Weighted AUC
            metrics['roc_auc_weighted'] = roc_auc_score(
                y_true_bin, y_prob, average='weighted', multi_class='ovr'
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
            
        logger.info(f"\n{dataset_name.upper()} RESULTS:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  Benign FPR: {metrics['benign_false_positive_rate']:.4f}")
        logger.info(f"  Attack Detection Rate: {metrics['attack_detection_rate']:.4f}")
        
        return metrics
        
    def _generate_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        dataset_name: str
    ):
        """Generate evaluation visualizations."""
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, dataset_name)
        
        # 2. ROC Curves
        self._plot_roc_curves(y_true, y_prob, dataset_name)
        
        # 3. Per-class Performance
        self._plot_per_class_metrics(y_true, y_pred, dataset_name)
        
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str
    ):
        """Plot confusion matrix."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by true class
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax1
        )
        ax1.set_title(f'Confusion Matrix - {dataset_name} (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax2
        )
        ax2.set_title(f'Confusion Matrix - {dataset_name} (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'confusion_matrix_{dataset_name}.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"✓ Saved confusion matrix: confusion_matrix_{dataset_name}.png")
        
    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        dataset_name: str
    ):
        """Plot ROC curves for all classes."""
        
        try:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))
            
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(
                        fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})'
                    )
                except:
                    continue
                    
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curves - {dataset_name}', fontsize=14)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'roc_curves_{dataset_name}.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            
            logger.info(f"✓ Saved ROC curves: roc_curves_{dataset_name}.png")
        except Exception as e:
            logger.warning(f"Could not plot ROC curves: {e}")
        
    def _plot_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str
    ):
        """Plot per-class metrics breakdown."""
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Attack Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance - {dataset_name}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'per_class_metrics_{dataset_name}.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"✓ Saved per-class metrics: per_class_metrics_{dataset_name}.png")
        
    def _save_report(self, metrics: Dict, dataset_name: str):
        """Save detailed evaluation report."""
        
        # JSON format
        report_path = self.output_dir / f'evaluation_report_{dataset_name}.json'
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"✓ Saved JSON report: {report_path}")
        
        # Human-readable text format
        text_report_path = self.output_dir / f'evaluation_report_{dataset_name}.txt'
        with open(text_report_path, 'w') as f:
            f.write(f"EVALUATION REPORT - {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1 (weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"  Precision (weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"  Recall (weighted): {metrics['recall_weighted']:.4f}\n")
            f.write(f"  ROC AUC (macro): {metrics.get('roc_auc_macro', 0):.4f}\n\n")
            
            f.write("IDS-SPECIFIC METRICS:\n")
            f.write(f"  Benign False Positive Rate: {metrics['benign_false_positive_rate']:.4f}\n")
            f.write(f"  Attack Detection Rate: {metrics['attack_detection_rate']:.4f}\n\n")
            
            f.write("PER-CLASS BREAKDOWN:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}\n")
            f.write("-" * 80 + "\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(
                    f"{class_name:<30} "
                    f"{class_metrics['precision']:<12.4f} "
                    f"{class_metrics['recall']:<12.4f} "
                    f"{class_metrics['f1_score']:<12.4f} "
                    f"{class_metrics['support']:<12}\n"
                )
                
        logger.info(f"✓ Saved text report: {text_report_path}")


def cross_dataset_validation(
    model: torch.nn.Module,
    datasets: Dict[str, torch.utils.data.DataLoader],
    class_names: List[str],
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Validate model across multiple datasets.
    
    Critical for production: Model must generalize beyond training data.
    """
    
    evaluator = IDSEvaluator(class_names)
    
    results = []
    
    for dataset_name, dataloader in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating on: {dataset_name}")
        logger.info(f"{'='*80}")
        
        metrics = evaluator.evaluate_model(
            model, dataloader, device, dataset_name
        )
        
        results.append({
            'dataset': dataset_name,
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'benign_fpr': metrics['benign_false_positive_rate'],
            'attack_detection_rate': metrics['attack_detection_rate'],
            'roc_auc': metrics.get('roc_auc_macro', 0)
        })
        
    results_df = pd.DataFrame(results)
    
    # Save comparison
    results_df.to_csv('evaluation_results/cross_dataset_comparison.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("CROSS-DATASET VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    
    return results_df
