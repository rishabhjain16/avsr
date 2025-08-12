#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoViT(nn.Module):
    def __init__(self, img_size=88, patch_size=16, in_chans=1, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4., drop_rate=0.1, max_temporal_len=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Temporal modeling with transformer
        self.max_temporal_len = max_temporal_len
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_temporal_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim*2, 
            dropout=drop_rate, 
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(self, x):
        # Handle different input formats
        if x.dim() == 5:  
            # Check if this is (B, T, C, H, W) or (B, C, T, H, W)
            # If the third dimension is small (like 1 or 3), it's likely channels
            if x.shape[2] <= 3:  # (B, T, C, H, W) format from dataloader
                B, T, C, H, W = x.shape
                if C == 1:
                    x = x.squeeze(2)  # (B, T, H, W) for grayscale
                else:
                    # If RGB, convert to grayscale by taking mean across channels
                    x = x.mean(dim=2)  # (B, T, H, W)
            else:  # (B, C, T, H, W) - ResNet format (unlikely)
                B, C, T, H, W = x.shape
                x = x.transpose(1, 2)  # (B, T, C, H, W)
                if C == 1:
                    x = x.squeeze(2)  # (B, T, H, W) for grayscale
                else:
                    # If RGB, convert to grayscale by taking mean across channels
                    x = x.mean(dim=2)  # (B, T, H, W)
        elif x.dim() == 4:  # (B, T, H, W)
            B, T, H, W = x.shape
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 4D (B,T,H,W) or 5D (B,T,C,H,W) or (B,C,T,H,W)")
        
        # Process each frame
        frame_features = []
        for t in range(T):
            frame = x[:, t:t+1, :, :].squeeze(1).unsqueeze(1)  # (B, 1, H, W)
            
            # Patch embedding
            frame_embed = self.patch_embed(frame)  # (B, num_patches, embed_dim)
            B_frame, num_patches, embed_dim = frame_embed.shape
            
            # Add cls token
            cls_tokens = self.cls_token.expand(B_frame, -1, -1)
            frame_embed = torch.cat((cls_tokens, frame_embed), dim=1)  # (B, num_patches+1, embed_dim)
            
            # Handle positional embedding size mismatch
            if frame_embed.shape[1] != self.pos_embed.shape[1]:
                # Interpolate positional embeddings to match the actual number of patches
                pos_embed = self.interpolate_pos_embed(self.pos_embed, num_patches)
            else:
                pos_embed = self.pos_embed
            
            frame_embed = frame_embed + pos_embed
            frame_embed = self.pos_drop(frame_embed)
            
            # Apply transformer blocks
            for block in self.blocks:
                frame_embed = block(frame_embed)
            
            frame_embed = self.norm(frame_embed)
            frame_features.append(frame_embed[:, 0])  # Take cls token
        
        # Stack temporal features (B, T, embed_dim)
        video_features = torch.stack(frame_features, dim=1)
        
        # Add temporal positional encoding
        if T > self.max_temporal_len:
            raise ValueError(f"Sequence length {T} exceeds maximum temporal length {self.max_temporal_len}")
        video_features = video_features + self.temporal_pos_embed[:, :T, :]
        
        # Temporal modeling
        video_features = self.temporal_encoder(video_features)
        
        return video_features

    def interpolate_pos_embed(self, pos_embed, num_patches):
        """Interpolate positional embeddings to match the number of patches."""
        # pos_embed: (1, num_patches_original + 1, embed_dim)
        # We need: (1, num_patches + 1, embed_dim)
        
        cls_token_embed = pos_embed[:, 0:1, :]  # (1, 1, embed_dim)
        patch_embed = pos_embed[:, 1:, :]  # (1, num_patches_original, embed_dim)
        
        if patch_embed.shape[1] == num_patches:
            return pos_embed
        
        # Calculate grid sizes
        num_patches_original = patch_embed.shape[1]
        grid_size_original = int(num_patches_original ** 0.5)
        grid_size_new = int(num_patches ** 0.5)
        
        if grid_size_original ** 2 != num_patches_original or grid_size_new ** 2 != num_patches:
            # Fallback: just truncate or pad
            if num_patches < num_patches_original:
                patch_embed = patch_embed[:, :num_patches, :]
            else:
                # Repeat the last embedding
                pad_size = num_patches - num_patches_original
                pad_embed = patch_embed[:, -1:, :].repeat(1, pad_size, 1)
                patch_embed = torch.cat([patch_embed, pad_embed], dim=1)
        else:
            # Proper interpolation
            embed_dim = patch_embed.shape[2]
            patch_embed = patch_embed.reshape(1, grid_size_original, grid_size_original, embed_dim)
            patch_embed = patch_embed.permute(0, 3, 1, 2)  # (1, embed_dim, grid_size, grid_size)
            
            # Interpolate
            patch_embed = torch.nn.functional.interpolate(
                patch_embed, 
                size=(grid_size_new, grid_size_new), 
                mode='bicubic', 
                align_corners=False
            )
            
            patch_embed = patch_embed.permute(0, 2, 3, 1)  # (1, grid_size_new, grid_size_new, embed_dim)
            patch_embed = patch_embed.reshape(1, num_patches, embed_dim)
        
        # Concatenate cls token and patch embeddings
        new_pos_embed = torch.cat([cls_token_embed, patch_embed], dim=1)
        return new_pos_embed


def video_vit():
    return VideoViT(img_size=88, patch_size=16, max_temporal_len=1000)
