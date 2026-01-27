"""
Evaluation Utilities for Diabetic Retinopathy Detection

This module provides functions for:
1. Converting YOLO predictions to CSV format
2. Computing evaluation metrics (accuracy, precision, recall, F1, QWK)
3. Generating confusion matrices and visualizations

Usage:
    python evaluate.py --pred-dir path/to/labels --true-csv path/to/labels.csv

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score,
    recall_score, 
    f1_score, 
    cohen_kappa_score,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO-Transformer predictions'
    )
    parser.add_argument(
        '--pred-dir',
        type=str,
        required=True,
        help='Directory containing YOLO prediction .txt files'
    )
    parser.add_argument(
        '--true-csv',
        type=str,
        required=True,
        help='CSV file with ground truth labels (columns: id_code, diagnosis)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory for saving evaluation results'
    )
    parser.add_argument(
        '--class-thresholds',
        type=str,
        default='0.01,0.01,0.01,0.01,0.01',
        help='Confidence thresholds per class (comma-separated)'
    )
    parser.add_argument(
        '--boost-weights',
        type=str,
        default='1.0,0.5,2.0,1.5,2.0',
        help='Confidence boost weights per class (comma-separated)'
    )
    return parser.parse_args()


def convert_yolo_to_csv(label_dir, output_csv):
    """
    Convert YOLO prediction .txt files to CSV format.
    
    YOLO output format: class_id x_center y_center width height confidence
    
    Args:
        label_dir (str): Directory containing .txt prediction files
        output_csv (str): Output CSV file path
    
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    records = []
    
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
            
        image_id = filename.replace(".txt", "")
        filepath = os.path.join(label_dir, filename)
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = int(parts[0])
                    confidence = float(parts[-1])
                    records.append({
                        'id_code': image_id,
                        'predicted_diagnosis': cls_id,
                        'confidence': confidence
                    })
    
    pred_df = pd.DataFrame(records)
    pred_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  Unique images: {pred_df['id_code'].nunique()}")
    
    return pred_df


def filter_predictions(pred_df, class_thresholds, boost_weights):
    """
    Filter predictions by confidence threshold and apply class-specific boosting.
    
    Args:
        pred_df (pd.DataFrame): Raw predictions
        class_thresholds (dict): Confidence threshold per class
        boost_weights (dict): Confidence boost weight per class
    
    Returns:
        pd.DataFrame: Filtered predictions with top-1 per image
    """
    # Apply confidence boost
    pred_df['adjusted_conf'] = pred_df.apply(
        lambda row: row['confidence'] * boost_weights.get(row['predicted_diagnosis'], 1.0),
        axis=1
    )
    
    # Filter by class-specific thresholds
    filtered_df = pred_df[
        pred_df.apply(
            lambda row: row['confidence'] >= class_thresholds.get(row['predicted_diagnosis'], 0),
            axis=1
        )
    ].copy()
    
    # Keep only top-1 prediction per image (highest adjusted confidence)
    top1_df = (filtered_df
               .sort_values(by=['id_code', 'adjusted_conf'], ascending=[True, False])
               .groupby('id_code')
               .first()
               .reset_index())
    
    return top1_df


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of class labels
    
    Returns:
        dict: Dictionary of computed metrics
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'qwk': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
    }
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    return metrics


def plot_confusion_matrix(cm, labels, output_path, title="Confusion Matrix"):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        output_path: Path to save the figure
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={'size': 12}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def print_metrics_report(metrics, class_names):
    """
    Print formatted metrics report.
    
    Args:
        metrics (dict): Computed metrics
        class_names (list): Names for each class
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS REPORT")
    print("="*60)
    
    print("\n📊 Overall Metrics:")
    print("-"*40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"  QWK:       {metrics['qwk']:.4f}")
    
    print("\n📊 Per-Class Metrics:")
    print("-"*40)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*50)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} "
              f"{metrics['f1_per_class'][i]:>10.4f}")
    
    print("\n📊 Confusion Matrix:")
    print("-"*40)
    print(f"  Total samples: {metrics['confusion_matrix'].sum()}")
    print(metrics['confusion_matrix'])
    print("="*60)


def evaluate(args):
    """
    Main evaluation function.
    
    Args:
        args: Parsed command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse thresholds and weights
    class_thresholds = {i: float(t) for i, t in enumerate(args.class_thresholds.split(','))}
    boost_weights = {i: float(w) for i, w in enumerate(args.boost_weights.split(','))}
    
    print("Configuration:")
    print(f"  Class thresholds: {class_thresholds}")
    print(f"  Boost weights: {boost_weights}")
    
    # Convert YOLO predictions to CSV
    pred_csv = os.path.join(args.output_dir, "predictions.csv")
    pred_df = convert_yolo_to_csv(args.pred_dir, pred_csv)
    
    # Filter and get top-1 predictions
    top1_df = filter_predictions(pred_df, class_thresholds, boost_weights)
    
    # Load ground truth
    true_df = pd.read_csv(args.true_csv)
    
    # Merge predictions with ground truth
    merged = pd.merge(true_df, top1_df, on='id_code', how='inner')
    print(f"\nMatched samples: {len(merged)}")
    
    # Get labels
    y_true = merged['diagnosis']
    y_pred = merged['predicted_diagnosis'].astype(int)
    labels = [0, 1, 2, 3, 4]
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, labels)
    
    # Print report
    print_metrics_report(metrics, class_names)
    
    # Save confusion matrix plot
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'QWK'],
        'Value': [
            metrics['accuracy'],
            metrics['precision_weighted'],
            metrics['recall_weighted'],
            metrics['f1_weighted'],
            metrics['qwk']
        ]
    })
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
