#!/usr/bin/env python
# scripts/evaluate.py
"""
Evaluation script for Swarm IDS model.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --data-dir data/raw
"""

import argparse
import sys
from pathlib import Path
import torch
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import CICIDSDataLoader, NetworkFeatureEngineer, LabelEncoder, create_dataloaders
from src.models import create_efficientnet_ids
from src.evaluation import IDSEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Swarm IDS model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing test data')
    parser.add_argument('--preprocessor-dir', type=str, default='models/preprocessors',
                       help='Directory containing saved preprocessors')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--test-day', type=str, default='Friday',
                       help='Day to use for testing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    preprocessor_dir = Path(args.preprocessor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("SWARM IDS - EVALUATION PIPELINE")
    logger.info("=" * 80)
    
    # ========== PHASE 1: LOAD DATA ==========
    logger.info("\n[PHASE 1] Loading test data...")
    
    loader = CICIDSDataLoader(data_dir)
    test_df = loader.load_and_validate(args.test_day, validate_checksum=False)
    
    # ========== PHASE 2: LOAD PREPROCESSORS ==========
    logger.info("\n[PHASE 2] Loading preprocessors...")
    
    feature_engineer = NetworkFeatureEngineer.load(preprocessor_dir)
    label_encoder = LabelEncoder()
    
    # Process data
    label_col = 'Label'
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]
    
    y_test_encoded = label_encoder.encode(y_test)
    X_test_processed = feature_engineer.transform(X_test)
    
    logger.info(f"Test samples: {len(X_test_processed):,}")
    logger.info(f"Features: {X_test_processed.shape[1]}")
    
    # ========== PHASE 3: CREATE DATALOADER ==========
    logger.info("\n[PHASE 3] Creating test dataloader...")
    
    from torch.utils.data import DataLoader
    from src.data import IDSDataset
    
    test_dataset = IDSDataset(X_test_processed, y_test_encoded, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # ========== PHASE 4: LOAD MODEL ==========
    logger.info("\n[PHASE 4] Loading model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    num_features = X_test_processed.shape[1]
    num_classes = label_encoder.num_classes
    
    model = create_efficientnet_ids(
        num_features=num_features,
        num_classes=num_classes,
        model_size='b0'  # Adjust if needed
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Best validation F1: {checkpoint.get('best_val_f1', 0):.4f}")
    
    # ========== PHASE 5: EVALUATE ==========
    logger.info("\n[PHASE 5] Evaluating model...")
    
    class_names = list(label_encoder.label_to_idx.keys())
    evaluator = IDSEvaluator(class_names, output_dir=output_dir)
    
    metrics = evaluator.evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        dataset_name=args.test_day
    )
    
    # ========== RESULTS ==========
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    logger.info(f"Benign FPR: {metrics['benign_false_positive_rate']:.4f}")
    logger.info(f"Attack Detection Rate: {metrics['attack_detection_rate']:.4f}")
    logger.info(f"ROC AUC (macro): {metrics.get('roc_auc_macro', 0):.4f}")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
