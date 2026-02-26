"""
Transfer Learning Utilities for HB Regression

This module provides utilities to initialize DSA-Mamba models with pretrained weights
from standard vision models (ResNet, DenseNet, etc.) to improve regression performance.

Key improvements:
1. Uses ImageNet pretrained backbone features
2. Initializes the regression head appropriately 
3. Handles architecture mismatch with layer projection
4. Supports fine-tuning strategies
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def load_pretrained_backbone_weights(model, pretrained_model_name='resnet50', device='cuda'):
    """
    Load pretrained ImageNet weights into the model's encoder layers.
    
    Args:
        model: DSA-Mamba VSSM model instance
        pretrained_model_name: Name of pretrained model ('resnet50', 'resnet101', 'densenet121', etc.)
        device: Device to load on
        
    Returns:
        model: Updated model with pretrained features
    """
    print(f"Loading pretrained {pretrained_model_name} weights...")
    
    # Load pretrained model
    if pretrained_model_name.startswith('resnet'):
        depth = int(pretrained_model_name.replace('resnet', ''))
        if depth == 50:
            pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif depth == 101:
            pretrained = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif pretrained_model_name.startswith('densenet'):
        pretrained = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif pretrained_model_name == 'vit_b':
        from torchvision.models import vit_b_16
        pretrained = vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        print(f"Pretrained model {pretrained_model_name} not supported, skipping pretrain loading")
        return model
    
    print(f"Pretrained model loaded. Now attempting to transfer suitable layers...")
    
    # Try to transfer relevant features
    try:
        # For models with patch_embed, try to adapt conv layers
        if hasattr(model, 'patch_embed'):
            patch_embed = model.patch_embed
            if hasattr(patch_embed, 'proj') and hasattr(pretrained, 'conv1'):
                # Match pretrained conv1 to our patch embed if possible
                try:
                    pretrained_conv = pretrained.conv1.weight  # [64, 3, 7, 7]
                    model_proj = patch_embed.proj.weight  # [embed_dim, 3, patch_size, patch_size]
                    
                    # Get device from model to ensure weights are on correct device
                    model_device = model_proj.device
                    pretrained_conv = pretrained_conv.to(model_device)
                    
                    # Average pool the pretrained weights to match our spatial size
                    if pretrained_conv.shape != model_proj.shape:
                        pretrained_conv = F.adaptive_avg_pool2d(
                            pretrained_conv, 
                            (model_proj.shape[-2], model_proj.shape[-1])
                        )
                        # Resize channels if needed
                        if pretrained_conv.shape[0] != model_proj.shape[0]:
                            # Pad or tile channels to match target dimension
                            target_channels = model_proj.shape[0]
                            current_channels = pretrained_conv.shape[0]
                            
                            if current_channels < target_channels:
                                # Pad channels by repeating/tiling
                                num_repeats = (target_channels + current_channels - 1) // current_channels
                                tiled = pretrained_conv.repeat(num_repeats, 1, 1, 1)
                                pretrained_conv = tiled[:target_channels]
                            else:
                                # Select first target_channels
                                pretrained_conv = pretrained_conv[:target_channels]
                    
                    patch_embed.proj.weight.data = pretrained_conv
                    print("✓ Transferred patch embedding weights from pretrained model")
                except Exception as e:
                    print(f"Could not transfer patch embedding: {e}")
    except Exception as e:
        print(f"Warning during feature transfer: {e}")
    
    return model


def initialize_regression_head(model, target_mean=None, target_std=None):
    """
    Initialize the regression head for better convergence.
    
    Args:
        model: Model with a regression head
        target_mean: Expected mean of target values (for clever initialization)
        target_std: Expected std of target values
    """
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        with torch.no_grad():
            # Initialize weights small for fine-tuning
            nn.init.normal_(model.head.weight, mean=0.0, std=0.01)
            if model.head.bias is not None:
                if target_mean is not None:
                    # Initialize bias to target mean for better starting point
                    nn.init.constant_(model.head.bias, target_mean)
                else:
                    nn.init.constant_(model.head.bias, 0.0)
        print("Regression head initialized")
        return model
    return model


def create_optimizer_with_lr_decay(model, base_lr=1e-4, weight_decay=1e-4, backbone_lr_factor=0.1):
    """
    Create optimizer with differentiated learning rates for backbone and head.
    
    Args:
        model: DSA-Mamba model
        base_lr: Learning rate for head
        weight_decay: Weight decay coefficient
        backbone_lr_factor: Factor to multiply for backbone learning rate (typically 0.1)
        
    Returns:
        Optimizer with layer-wise learning rates
    """
    param_groups = []
    
    # Separate parameters by whether they're in backbone or head
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    # Backbone gets lower learning rate for fine-tuning
    if backbone_params:
        # convert list to tuple to avoid foreach dispatch issues
        param_groups.append({
            'params': tuple(backbone_params),
            'lr': base_lr * backbone_lr_factor,
            'weight_decay': weight_decay
        })
    
    # Head gets full learning rate
    if head_params:
        param_groups.append({
            'params': tuple(head_params),
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    # Disable foreach to avoid _foreach_* API expecting tuples
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), foreach=False)
    print(f"Optimizer created with differentiated LR:")
    print(f"  Backbone LR: {base_lr * backbone_lr_factor:.2e}")
    print(f"  Head LR: {base_lr:.2e}")
    return optimizer


def create_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=50):
    """
    Create a scheduler with warmup followed by cosine annealing.
    
    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        
    Returns:
        Scheduler function to call each epoch
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class RegressionLossWithRangeAwareness(nn.Module):
    """
    Custom loss that prevents the network from collapsing to mean prediction.
    
    Combines MSE with variance and range constraints to encourage full output range usage.
    """
    def __init__(self, variance_weight=0.1, range_weight=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.variance_weight = variance_weight
        self.range_weight = range_weight
    
    def forward(self, pred, target):
        # Main MSE loss
        mse_loss = self.mse(pred, target)
        
        # Encourage variance in predictions
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        if pred_std > 1e-6 and target_std > 1e-6:
            variance_loss = self.variance_weight * ((pred_std - target_std) ** 2) / (target_std ** 2)
        else:
            variance_loss = 0.0
        
        # Encourage similar range
        pred_range = torch.max(pred) - torch.min(pred)
        target_range = torch.max(target) - torch.min(target)
        if pred_range > 1e-6 and target_range > 1e-6:
            range_loss = self.range_weight * ((pred_range - target_range) ** 2) / (target_range ** 2)
        else:
            range_loss = 0.0
        
        total_loss = mse_loss + variance_loss + range_loss
        return total_loss


def setup_transfer_learning(model, pretrained_backbone='resnet50', base_lr=1e-4, 
                           target_mean=None, target_std=None, device='cuda'):
    """
    Complete setup for transfer learning with all bells and whistles.
    
    Args:
        model: DSA-Mamba VSSM model
        pretrained_backbone: Name of pretrained model to use
        base_lr: Base learning rate
        target_mean: Mean of targets (for initialization)
        target_std: Std of targets
        device: Device to use
        
    Returns:
        tuple: (model, optimizer, criterion, scheduler)
    """
    print("\n" + "="*60)
    print("TRANSFER LEARNING SETUP")
    print("="*60)
    
    # Load pretrained weights
    model = load_pretrained_backbone_weights(model, pretrained_backbone, device)
    
    # Initialize regression head
    model = initialize_regression_head(model, target_mean, target_std)
    
    # Create optimizer with layer-wise LR
    optimizer = create_optimizer_with_lr_decay(model, base_lr=base_lr, backbone_lr_factor=0.1)
    
    # Create custom loss
    criterion = RegressionLossWithRangeAwareness(variance_weight=0.1, range_weight=0.05)
    
    # Create scheduler
    scheduler = create_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=50)
    
    print("="*60)
    print("\nTransfer learning setup complete!")
    print("Tips for training:")
    print("  1. Use --resume-from to continue training from checkpoint")
    print("  2. Monitor pred/true range ratio in validation output")
    print("  3. If predictions still near mean, try higher base_lr or longer training")
    print("="*60 + "\n")
    
    return model, optimizer, criterion, scheduler


if __name__ == "__main__":
    # Example usage
    from model.DSAmamba import VSSM as dsamamba
    
    model = dsamamba(in_chans=3, num_classes=1)
    model.to('cuda')
    
    # Pre-init lazy layers
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device='cuda')
        _ = model(dummy)
    
    # Setup transfer learning
    model, optimizer, criterion, scheduler = setup_transfer_learning(
        model, 
        pretrained_backbone='resnet50',
        base_lr=5e-4,
        target_mean=10.0,
        target_std=2.0,
        device='cuda'
    )
    
    print("Model ready for training with transfer learning!")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")
