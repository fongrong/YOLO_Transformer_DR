"""
ViT (Vision Transformer) Backbone for YOLO Integration

This module implements a ViT-based backbone with lesion-aware enhancement
for diabetic retinopathy detection and grading.

Features:
- ViT-Base architecture with 640x640 input support
- Lightweight U-Net for lesion mask prediction
- Focal loss with cost-sensitive penalty
- Lesion-guided feature enhancement

Author: Tsai I-Shiuan, Cheng Bor-Wen, Yang Feng-Jung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTConfig:
    """
    Configuration class for ViT backbone hyperparameters.
    
    Args:
        gamma (float): Focal loss gamma parameter. Default: 2.0
        pull_weight (float): Weight for pull loss component. Default: 1.0
        penalty_weight (float): Cost-sensitive penalty weight. Default: 2.0
        contrastive_temp (float): Temperature for contrastive learning. Default: 0.05
        alpha (float): Lesion enhancement strength. Default: 1.0
        std_thresh (float): Standard deviation threshold. Default: 0.01
        bce_weight (float): Binary cross-entropy weight for lesion loss. Default: 0.3
        loss_weights (dict): Dictionary of loss component weights.
    """
    
    def __init__(self,
                 gamma=2.0,
                 pull_weight=1.0,
                 penalty_weight=2.0,
                 contrastive_temp=0.05,
                 alpha=1.0,
                 std_thresh=0.01,
                 bce_weight=0.3,
                 loss_weights=None):
        self.gamma = gamma
        self.pull_weight = pull_weight
        self.penalty_weight = penalty_weight
        self.contrastive_temp = contrastive_temp
        self.alpha = alpha
        self.std_thresh = std_thresh
        self.bce_weight = bce_weight
        self.loss_weights = loss_weights or {
            "cls_loss": 1.0,
            "focal_loss": 1.0,
            "pull_loss": 0.8,
            "penalty_loss": 2.0,
            "lesion_loss": 0.3
        }


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for lesion mask prediction.
    
    A simplified encoder-decoder architecture for generating
    attention masks highlighting potential lesion regions.
    
    Input: [B, 3, H, W] RGB fundus image
    Output: [B, 1, H, W] Lesion probability map
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass: encode then decode to lesion mask."""
        return self.decoder(self.encoder(x))


class ViTBackbone(nn.Module):
    """
    Vision Transformer Backbone with Lesion-Aware Enhancement.
    
    This backbone combines:
    1. ViT-Base for global feature extraction
    2. Lightweight U-Net for lesion region attention
    3. Cost-sensitive loss for handling class imbalance
    
    The lesion mask is used to enhance input features before
    feeding to the ViT, improving sensitivity to subtle lesions.
    
    Args:
        cfg (ViTConfig): Configuration object with hyperparameters
    """
    
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        # ViT-Base with 2 output classes (binary: hemorrhage vs no hemorrhage)
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            img_size=640,
            patch_size=16,
            num_classes=2
        )
        
        # Lesion detection subnetwork
        self.lesion_unet = LightweightUNet()

        # Cost matrix for binary classification (2x2)
        # Higher penalty for misclassifying positive cases
        self.cost_matrix = torch.tensor([
            [1.0, cfg.penalty_weight],      # True negative, False positive
            [cfg.penalty_weight, 1.0]       # False negative, True positive
        ], dtype=torch.float32)

    def forward(self, x, pseudo_mask=None, cls_target=None):
        """
        Forward pass with optional loss computation.
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            pseudo_mask (torch.Tensor, optional): Ground truth lesion masks
            cls_target (torch.Tensor, optional): Classification labels
        
        Returns:
            tuple: (logits, lesion_loss, classification_loss)
        """
        # Generate lesion attention mask
        lesion_mask = self.lesion_unet(x)
        
        # Enhance input features using lesion mask
        # Scale factor is clamped to [0.8, 1.2] for stability
        enhancement = torch.clamp(1 + self.cfg.alpha * (lesion_mask - 0.5), 0.8, 1.2)
        x_enhanced = x * enhancement
        
        # ViT classification
        logits = self.vit(x_enhanced)
        preds = logits.argmax(dim=1)

        # Initialize losses
        cls_loss = penalty_loss = pull_loss = lesion_loss = None
        
        if cls_target is not None:
            # Focal loss for handling class imbalance
            ce = F.cross_entropy(logits, cls_target, reduction='none')
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.cfg.gamma * ce).mean()
            focal = focal * self.cfg.loss_weights['focal_loss']
            
            # Cost-sensitive penalty
            with torch.no_grad():
                weight = self.cost_matrix[cls_target, preds].to(logits.device)
            penalty_loss = (ce * weight).mean() * self.cfg.loss_weights['penalty_loss']
            
            # Lesion segmentation loss
            if pseudo_mask is not None:
                bce = F.binary_cross_entropy(lesion_mask, pseudo_mask)
                lesion_loss = self.cfg.bce_weight * bce
            
            # Total classification loss
            cls_loss = focal + penalty_loss
            
        return logits, lesion_loss, cls_loss


# -------------------- Test Script --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration and model
    cfg = ViTConfig()
    model = ViTBackbone(cfg).to(device)
    model.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 640, 640, device=device)
    
    with torch.no_grad():
        logits, lesion_loss, cls_loss = model(x)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {logits.argmax(dim=1)}")
