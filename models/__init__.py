"""
YOLO-Transformer Models Package

This package contains backbone implementations for hybrid YOLO-Transformer
models used in diabetic retinopathy detection.

Available models:
- CvTBackbone: Convolutional Vision Transformer (CvT-Tiny)
- ViTBackbone: Vision Transformer with lesion-aware enhancement (ViT-Base)
"""

from .cvt_backbone import CvTBackbone
from .vit_backbone import ViTBackbone, ViTConfig, LightweightUNet

__all__ = [
    'CvTBackbone',
    'ViTBackbone', 
    'ViTConfig',
    'LightweightUNet'
]

__version__ = '1.0.0'
