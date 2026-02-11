# Hybrid YOLO-Transformer for Diabetic Retinopathy Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid deep learning framework combining **YOLO object detection** with **Vision Transformer (ViT/CvT) classification heads** for automated diabetic retinopathy (DR) grading from fundus images.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [K-Fold Cross-Validation](#k-fold-cross-validation)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🔬 Overview

Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide. This repository implements a novel hybrid architecture that combines:

- **YOLO Detection Backbones**: YOLOv9t or YOLO11n for efficient feature extraction
- **Transformer Classification Heads**: ViT-Base or CvT-21 for global context modeling
- **Multiple Preprocessing Strategies**: Original RGB, Green Channel, Gaussian Filtering

The framework achieves **99.26% binary classification accuracy** and **83.06% five-class grading accuracy** on the APTOS 2019 dataset.

## ✨ Key Features

- 🔄 **Modular Architecture**: Easy swap between YOLO backbones and Transformer heads
- 📊 **K-Fold Cross-Validation**: Built-in hyperparameter optimization
- 🚀 **Mixed Precision Training**: Automatic FP16/FP32 for faster training
- 📈 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, QWK metrics
- 🎯 **Cost-Sensitive Loss**: Handles class imbalance in DR severity levels
- 📦 **Easy Deployment**: Simple inference pipeline for clinical use

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone repository
git clone https://github.com/fongrong/yolo-transformer-dr.git
cd yolo-transformer-dr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
yolo-transformer-dr/
├── models/
│   ├── __init__.py
│   ├── cvt_backbone.py      # CvT-Tiny backbone implementation
│   └── vit_backbone.py      # ViT-Base backbone with lesion attention
├── scripts/
│   ├── train.py             # Main training script
│   ├── kfold_cv.py          # K-fold cross-validation
│   └── inference.py         # Run predictions on new images
├── utils/
│   └── evaluate.py          # Evaluation metrics and visualization
├── configs/
│   └── default.yml          # Default hyperparameters
├── data/
│   └── aptos2019.yml        # Dataset configuration example
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Usage

### Training

Train a hybrid YOLO-Transformer model:

```bash
# Train with CvT backbone (recommended)
python scripts/train.py \
    --backbone cvt \
    --yolo-model yolov9t.pt \
    --data data/aptos2019.yml \
    --epochs 200 \
    --batch-size 32 \
    --lr 0.001

# Train with ViT backbone
python scripts/train.py \
    --backbone vit \
    --yolo-model yolo11n.pt \
    --data data/aptos2019.yml \
    --epochs 200
```

**Key Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--backbone` | Backbone type: `cvt` or `vit` | `cvt` |
| `--yolo-model` | Base YOLO weights | `yolov9t.pt` |
| `--data` | Dataset YAML path | Required |
| `--epochs` | Training epochs | 200 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--img-size` | Input image size | 640 |

### K-Fold Cross-Validation

Run K-fold CV with hyperparameter search:

```bash
python scripts/kfold_cv.py \
    --data data/aptos2019.yml \
    --folds 5 \
    --backbone cvt \
    --epochs 50 \
    --final-epochs 100
```

This will:
1. Test multiple hyperparameter combinations
2. Report average mAP@50 across folds
3. Train final model with best parameters

### Inference

Run predictions on new fundus images:

```bash
python scripts/inference.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/test_images \
    --conf 0.2 \
    --save-txt \
    --save-conf
```

**Output:**
- Prediction labels in `runs/predict/exp/labels/`
- Visualization images (if `--save-img` specified)
- Inference time statistics

### Evaluation

Compute evaluation metrics:

```bash
python utils/evaluate.py \
    --pred-dir runs/predict/exp/labels \
    --true-csv data/test_labels.csv \
    --output-dir evaluation_results
```

**Outputs:**
- Confusion matrix visualization
- Per-class precision, recall, F1
- Overall accuracy, weighted F1, QWK

## 🏗️ Model Architecture

### Hybrid Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────┐
│ Fundus      │     │ Preprocessing    │     │ YOLO Backbone   │     │ Trans- │
│ Image       │ --> │ (O/G/F)          │ --> │ (v9t/11n)       │ --> │ former │ --> DR Grade
│ 640×640     │     │                  │     │                  │     │ (ViT/  │     (0-4)
└─────────────┘     └──────────────────┘     └──────────────────┘     │ CvT)   │
                                                                       └────────┘
```

### Backbone Options

| Backbone | Parameters | Inference Time | Best For |
|----------|------------|----------------|----------|
| CvT-Tiny | 32M | 20.4s | 5-class grading |
| ViT-Base | 86M | 37.2s | Binary screening |

### Preprocessing Strategies

1. **Original (O)**: Raw RGB images, 640×640
2. **Green Channel (G)**: Enhanced contrast with γ=1.2, median filter
3. **Gaussian Filter (F)**: Noise reduction with σ=1.5, 5×5 kernel

## 📊 Results

### Performance on APTOS 2019 Dataset

| Configuration | Binary Acc. | 5-Class Acc. | QWK | F1 Score |
|--------------|-------------|--------------|-----|----------|
| Y9-ViT-F | 97.47% | **83.06%** | 0.900 | 0.831 |
| Y9-CvT-F | 97.47% | 83.06% | 0.900 | 0.831 |
| Y11-ViT-O | **99.26%** | 82.02% | 0.985 | 0.993 |
| Y11-CvT-O | 99.26% | 82.02% | 0.985 | 0.993 |

### Key Findings

- **Binary Classification**: YOLO11n + ViT with original images achieves 99.26% accuracy
- **5-Class Grading**: YOLOv9t + ViT/CvT with Gaussian filtering achieves 83.06% accuracy
- **Preprocessing Matters**: Significant detector×preprocessing interaction (F=1080.12, p<0.0001)

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{tsai2025hybrid,
  title={Hybrid YOLO-Transformer architectures for automated diabetic retinopathy grading: 
         systematic evaluation of backbones and preprocessing},
  author={Tsai, I-Shiuan and Cheng, Bor-Wen and Yang, Feng-Jung},
  journal={npj Digital Public Health},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [timm](https://github.com/huggingface/pytorch-image-models) for Vision Transformer models
- [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) for the dataset

## 📧 Contact

For questions or collaboration:
- Feng-Jung Yang, MD, PhD - fongrong@ntu.edu.tw
- Department of Medical Genetics, National Taiwan University Hospital
