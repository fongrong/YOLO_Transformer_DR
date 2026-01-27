"""
K-Fold Cross-Validation with Hyperparameter Search

This script performs K-fold cross-validation combined with grid search
for hyperparameter optimization of the YOLO-Transformer model.

Usage:
    python kfold_cv.py --data path/to/data.yml --folds 5

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import argparse
import torch
from ultralytics import YOLO
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime

import sys
sys.path.append('..')
from models.vit_backbone import ViTBackbone, ViTConfig
from models.cvt_backbone import CvTBackbone


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='K-Fold Cross-Validation for YOLO-Transformer'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML configuration file'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='vit',
        choices=['cvt', 'vit'],
        help='Backbone architecture'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov9t.pt',
        help='Base YOLO model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Epochs per fold (reduced for efficiency)'
    )
    parser.add_argument(
        '--final-epochs',
        type=int,
        default=100,
        help='Epochs for final training with best params'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Training device'
    )
    return parser.parse_args()


def create_backbone(backbone_type, device):
    """Create backbone model based on type."""
    if backbone_type == 'cvt':
        return CvTBackbone(pretrained=True).to(device)
    elif backbone_type == 'vit':
        cfg = ViTConfig()
        return ViTBackbone(cfg).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")


def run_kfold_cv(args):
    """
    Run K-Fold cross-validation with hyperparameter grid search.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict: Best hyperparameters and corresponding score
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define hyperparameter search space
    param_grid = {
        'lr0': [0.001, 0.005, 0.01],           # Learning rate
        'batch_size': [16, 32],                 # Batch size
        'weight_decay': [0.0001, 0.0005, 0.001] # Weight decay
    }

    # Generate all parameter combinations
    param_list = list(ParameterGrid(param_grid))
    print(f"\nTotal parameter combinations: {len(param_list)}")
    print(f"Total training runs: {len(param_list) * args.folds}")

    # Track best results
    best_score = -1
    best_params = None
    all_results = []

    # K-Fold Cross-Validation
    for param_idx, params in enumerate(param_list):
        print(f"\n{'='*60}")
        print(f"Parameter Set {param_idx + 1}/{len(param_list)}")
        print(f"Parameters: {params}")
        print('='*60)
        
        fold_scores = []

        for fold in range(args.folds):
            print(f"\n  Fold {fold + 1}/{args.folds}")
            
            # Reload model with fresh backbone for each fold
            model = YOLO(args.yolo_model).to(device)
            backbone = create_backbone(args.backbone, device)
            model.model.model[0] = backbone

            # Train for this fold
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=640,
                batch=params['batch_size'],
                optimizer="AdamW",
                lr0=params['lr0'],
                weight_decay=params['weight_decay'],
                val=True,
                device=args.device,
                verbose=False
            )

            # Evaluate and get mAP@50
            results = model.val()
            map50 = results.box.map50
            fold_scores.append(map50)
            print(f"    mAP@50: {map50:.4f}")

        # Calculate average score across folds
        avg_score = sum(fold_scores) / len(fold_scores)
        std_score = (sum((s - avg_score)**2 for s in fold_scores) / len(fold_scores))**0.5
        
        print(f"\n  Average mAP@50: {avg_score:.4f} ± {std_score:.4f}")
        
        # Store results
        result = {
            'params': params,
            'avg_score': avg_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        }
        all_results.append(result)

        # Update best parameters
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            print("  *** New best score! ***")

    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"\nBest Parameters: {best_params}")
    print(f"Best mAP@50: {best_score:.4f}")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"cv_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return best_params, best_score


def train_final_model(args, best_params):
    """
    Train final model with best hyperparameters.
    
    Args:
        args: Command line arguments
        best_params: Best hyperparameters from CV
    """
    print("\n" + "="*60)
    print("FINAL TRAINING WITH BEST PARAMETERS")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model with best parameters
    model = YOLO(args.yolo_model).to(device)
    backbone = create_backbone(args.backbone, device)
    model.model.model[0] = backbone

    # Full training
    model.train(
        data=args.data,
        epochs=args.final_epochs,
        imgsz=640,
        batch=best_params['batch_size'],
        optimizer="AdamW",
        lr0=best_params['lr0'],
        weight_decay=best_params['weight_decay'],
        val=True,
        device=args.device,
        cos_lr=True,
        warmup_epochs=10,
        amp=True,
        label_smoothing=0.1
    )

    print("\nFinal training completed!")


if __name__ == "__main__":
    args = parse_args()
    
    # Run cross-validation
    best_params, best_score = run_kfold_cv(args)
    
    # Train final model with best parameters
    train_final_model(args, best_params)
