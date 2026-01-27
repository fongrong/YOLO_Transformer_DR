"""
YOLO-Transformer Training Script

This script trains a hybrid YOLO-Transformer model for diabetic retinopathy
detection by replacing the YOLO backbone with either ViT or CvT.

Usage:
    python train.py --backbone cvt --data path/to/data.yml --epochs 200

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import argparse
import torch
from ultralytics import YOLO

# Import custom backbones
import sys
sys.path.append('..')
from models.cvt_backbone import CvTBackbone
from models.vit_backbone import ViTBackbone, ViTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLO with Transformer backbone for DR detection'
    )
    parser.add_argument(
        '--backbone', 
        type=str, 
        default='cvt',
        choices=['cvt', 'vit'],
        help='Backbone architecture: cvt (CvT-Tiny) or vit (ViT-Base)'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov9t.pt',
        help='Base YOLO model weights (yolov9t.pt, yolo11n.pt, etc.)'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.00005,
        help='Weight decay for regularization'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Training device (cuda or cpu)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Experiment name'
    )
    return parser.parse_args()


def create_backbone(backbone_type, device):
    """
    Create the specified backbone architecture.
    
    Args:
        backbone_type (str): 'cvt' or 'vit'
        device (torch.device): Target device
    
    Returns:
        nn.Module: Backbone model
    """
    if backbone_type == 'cvt':
        print("Loading CvT-Tiny backbone...")
        backbone = CvTBackbone(pretrained=True)
    elif backbone_type == 'vit':
        print("Loading ViT-Base backbone...")
        cfg = ViTConfig()
        backbone = ViTBackbone(cfg)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    return backbone.to(device)


def train_yolo_transformer(args):
    """
    Main training function for YOLO-Transformer hybrid model.
    
    Args:
        args: Parsed command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load base YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model).to(device)

    # 2. Replace backbone with Transformer
    backbone = create_backbone(args.backbone, device)
    model.model.model[0] = backbone
    print(f"Backbone replaced with {args.backbone.upper()}")

    # 3. Start training
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"  Backbone: {args.backbone.upper()}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Learning rate: {args.lr}")
    print("="*50 + "\n")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        optimizer="AdamW",
        lr0=args.lr,
        weight_decay=args.weight_decay,
        val=True,
        device=args.device,
        lrf=0.01,                    # Final learning rate factor
        cos_lr=True,                 # Cosine learning rate scheduler
        warmup_epochs=20,            # Warmup epochs
        patience=0,                  # Disable early stopping
        amp=True,                    # Automatic Mixed Precision
        label_smoothing=0.1,         # Label smoothing for regularization
        save_period=5,               # Save checkpoint every N epochs
        iou=0.6,                     # IoU threshold
        project=args.project,
        name=args.name
    )

    print("\nTraining completed!")
    print(f"Results saved to: {args.project}/{args.name}")


if __name__ == "__main__":
    args = parse_args()
    train_yolo_transformer(args)
