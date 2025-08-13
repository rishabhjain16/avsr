#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class HF_ViT_Frontend(nn.Module):
    """
    Hugging Face pretrained Vision Transformer frontend for video processing.
    Processes video frames individually through a pretrained ViT and adds temporal modeling.
    """
    
    def __init__(
        self, 
        model_name="google/vit-base-patch16-224",
        freeze_backbone=False,
        output_dim=512,
        max_frames=None
    ):
        """
        Args:
            model_name: Hugging Face ViT model name
            freeze_backbone: Whether to freeze the pretrained ViT backbone
            output_dim: Output dimension (will be projected from ViT's hidden size)
            max_frames: Maximum number of frames to process (None for no limit)
        """
        super().__init__()
        
        self.max_frames = max_frames
        
        # Load pretrained ViT model
        try:
            self.vit = ViTModel.from_pretrained(model_name)
            hidden_size = self.vit.config.hidden_size
        except Exception as e:
            print(f"Failed to load {model_name}, falling back to base config: {e}")
            # Fallback to base ViT configuration
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
            self.vit = ViTModel(config)
            hidden_size = config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT backbone frozen")
        
        # Projection layer to desired output dimension
        self.projection = nn.Linear(hidden_size, output_dim)
        
        # Temporal modeling with a simple transformer
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=8,
                dim_feedforward=output_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
        print(f"Initialized HF_ViT_Frontend with:")
        print(f"  - Model: {model_name}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Frozen: {freeze_backbone}")
        print(f"  - Note: Input images will be resized to 224x224 for ViT")
    
    def forward(self, x):
        """
        Forward pass for video input.
        
        Args:
            x: Input tensor of shape (B, T, H, W) for grayscale or (B, T, C, H, W) for RGB
            
        Returns:
            Tensor of shape (B, T, output_dim)
        """
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # Handle different input formats
        if x.dim() == 4:  # (B, T, H, W) - grayscale
            x = x.unsqueeze(2)  # Add channel dimension: (B, T, 1, H, W)
        
        if x.shape[2] == 1:  # Grayscale - convert to RGB
            x = x.repeat(1, 1, 3, 1, 1)  # (B, T, 3, H, W)
        
        # Limit frames if specified
        if self.max_frames is not None and num_frames > self.max_frames:
            x = x[:, :self.max_frames]
            num_frames = self.max_frames
        
        # Process each frame through ViT
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t]  # (B, C, H, W)
            
            # Resize frame to ViT expected size (224x224)
            if frame.shape[-1] != 224 or frame.shape[-2] != 224:
                frame = torch.nn.functional.interpolate(
                    frame, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Pass through ViT
            outputs = self.vit(pixel_values=frame)
            
            # Use CLS token (first token) as frame representation
            frame_feat = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
            frame_features.append(frame_feat)
        
        # Stack temporal features (B, T, hidden_size)
        video_features = torch.stack(frame_features, dim=1)
        
        # Project to desired output dimension
        video_features = self.projection(video_features)  # (B, T, output_dim)
        
        # Apply temporal modeling
        video_features = self.temporal_encoder(video_features)  # (B, T, output_dim)
        
        # Final layer norm
        video_features = self.layer_norm(video_features)
        
        return video_features


def hf_vit():
    """Factory function for default pretrained ViT frontend."""
    return HF_ViT_Frontend()


def hf_vit_frozen():
    """Factory function for frozen pretrained ViT frontend."""
    return HF_ViT_Frontend(freeze_backbone=True)


def hf_vit_large():
    """Factory function for large pretrained ViT frontend."""
    return HF_ViT_Frontend(
        model_name="google/vit-large-patch16-224",
        output_dim=768
    )


def hf_vit_small():
    """Factory function for small pretrained ViT frontend.""" 
    return HF_ViT_Frontend(
        model_name="google/vit-small-patch16-224",
        output_dim=384
    )
