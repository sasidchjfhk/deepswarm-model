#!/usr/bin/env python
# scripts/export_onnx.py
"""
Export trained model to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth --output models/onnx/model.onnx
"""

import argparse
import sys
from pathlib import Path
import torch
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_efficientnet_ids
from src.deployment import ONNXExporter


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='models/onnx/model.onnx',
                       help='Output path for ONNX model')
    parser.add_argument('--num-features', type=int, default=50,
                       help='Number of input features')
    parser.add_argument('--num-classes', type=int, default=15,
                       help='Number of output classes')
    parser.add_argument('--model-size', type=str, default='b0',
                       choices=['b0', 'b1', 'b2', 'b3'],
                       help='Model size')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    
    logger.info("=" * 80)
    logger.info("SWARM IDS - ONNX EXPORT")
    logger.info("=" * 80)
    
    # ========== PHASE 1: LOAD MODEL ==========
    logger.info("\n[PHASE 1] Loading PyTorch model...")
    
    # Create model
    model = create_efficientnet_ids(
        num_features=args.num_features,
        num_classes=args.num_classes,
        model_size=args.model_size
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # ========== PHASE 2: EXPORT TO ONNX ==========
    logger.info("\n[PHASE 2] Exporting to ONNX...")
    
    exporter = ONNXExporter(model, input_shape=(args.num_features,))
    
    onnx_path = exporter.export(
        output_path=output_path,
        opset_version=14,
        verify=not args.no_verify
    )
    
    # ========== RESULTS ==========
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE!")
    logger.info(f"ONNX model saved to: {onnx_path}")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Test the ONNX model with sample data")
    logger.info("  2. Deploy with model serving (see scripts/serve.py)")
    logger.info("  3. Monitor inference latency (<20ms target)")


if __name__ == '__main__':
    main()
