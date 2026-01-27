"""
CvT (Convolutional Vision Transformer) Backbone for YOLO Integration

This module implements a CvT-based backbone that can replace the default 
YOLO backbone for diabetic retinopathy detection tasks.

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import torch
import torch.nn as nn
import timm
import math


class CvTBackbone(nn.Module):
    """
    CvT Backbone: A Convolutional Vision Transformer backbone adapted for YOLO.
    
    This backbone uses ConViT-Tiny architecture with the following specifications:
    - Input size: 224x224 (consistent with pretrained weights)
    - Patch size: 16x16
    - Output channels: 320 (adapted for YOLO compatibility)
    
    The model uses Automatic Mixed Precision (AMP) for efficient inference
    instead of manual half-precision conversion.
    
    Args:
        pretrained (bool): Whether to load pretrained ImageNet weights. Default: True
    """
    
    def __init__(self, pretrained=True):
        super(CvTBackbone, self).__init__()

        # 1. Create ConViT-Tiny model
        self.convit = timm.create_model(
            'convit_tiny',
            pretrained=pretrained,
            img_size=224,
            patch_size=16,
            num_classes=0  # Remove classification head, keep feature extraction only
        )

        # 2. ConViT-Tiny output dimension (192 for tiny variant)
        out_channels = 192

        # 3. Adapter: 1x1 Conv to match YOLO's expected 320 channels
        self.adapter = nn.Conv2d(out_channels, 320, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the CvT backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, 224, 224]
        
        Returns:
            List[torch.Tensor]: List containing feature map of shape [B, 320, H, W]
        """
        # Use AMP for automatic mixed precision
        with torch.cuda.amp.autocast():
            # a. Extract ConViT features: [B, Tokens, C]
            features = self.convit.forward_features(x)
            B, T, C = features.shape

            # b. Remove CLS/dist tokens if present
            cls_dist_tokens = 0
            while int(math.sqrt(T - cls_dist_tokens))**2 != (T - cls_dist_tokens):
                cls_dist_tokens += 1
            if cls_dist_tokens > 0:
                features = features[:, cls_dist_tokens:, :]
                T -= cls_dist_tokens

            # c. Reshape to 2D feature map: [B, C, H, W] (14x14=196 tokens)
            grid_size = int(math.sqrt(T))
            if grid_size * grid_size != T:
                raise ValueError(f"Invalid token shape: expected square grid, got {T} tokens")
            features = features.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)

            # d. Adapt channel dimensions for YOLO compatibility
            adapted_features = self.adapter(features)

        return [adapted_features]


# -------------------- Test Script --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model (float32)
    model = CvTBackbone(pretrained=True).to(device)
    model.eval()

    # Prepare test input
    x = torch.randn(1, 3, 224, 224, device=device)

    # Run inference with AMP
    with torch.cuda.amp.autocast():
        features = model(x)

    for i, f in enumerate(features):
        print(f"Feature {i+1} shape: {f.shape}")
    # Expected output: [1, 320, 14, 14]
