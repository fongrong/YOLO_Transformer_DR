"""
YOLO-Transformer Inference Script

This script runs inference on fundus images using a trained YOLO-Transformer
model for diabetic retinopathy detection and grading.

Usage:
    python inference.py --weights path/to/best.pt --source path/to/images

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import argparse
import time
import os
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained YOLO-Transformer model'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (best.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to test images (folder or single image)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Inference image size'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.2,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.3,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results to text files'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in text files'
    )
    parser.add_argument(
        '--save-img',
        action='store_true',
        help='Save images with predictions'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/predict',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Experiment name for this inference run'
    )
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on the specified source images.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        tuple: (results, inference_time)
    """
    # Load trained model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Count images
    if os.path.isdir(args.source):
        n_images = len([f for f in os.listdir(args.source) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {n_images} images in {args.source}")
    else:
        n_images = 1
        print(f"Processing single image: {args.source}")

    # Start timing
    start_time = time.time()

    # Run prediction
    results = model.predict(
        source=args.source,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        save=args.save_img,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        hide_conf=False
    )

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    print(f"  Images processed: {n_images}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per image: {total_time/n_images:.3f} seconds")
    print(f"  Throughput: {n_images/total_time:.1f} images/second")
    print(f"  Results saved to: {args.project}/{args.name}")
    print("="*50)

    return results, total_time


def print_detection_summary(results):
    """
    Print summary of detections from results.
    
    Args:
        results: YOLO prediction results
    """
    class_names = {
        0: "No DR",
        1: "Mild",
        2: "Moderate", 
        3: "Severe",
        4: "Proliferative DR"
    }
    
    print("\nDETECTION SUMMARY BY CLASS:")
    print("-" * 40)
    
    class_counts = {i: 0 for i in range(5)}
    total_detections = 0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                if cls_id in class_counts:
                    class_counts[cls_id] += 1
                    total_detections += 1
    
    for cls_id, count in class_counts.items():
        cls_name = class_names.get(cls_id, f"Class {cls_id}")
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {cls_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n  Total detections: {total_detections}")


if __name__ == "__main__":
    args = parse_args()
    
    # Run inference
    results, inference_time = run_inference(args)
    
    # Print detection summary
    print_detection_summary(results)
