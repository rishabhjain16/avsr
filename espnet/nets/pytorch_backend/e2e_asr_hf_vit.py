#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from espnet.nets.pytorch_backend.frontend.hf_vit import hf_vit
from espnet.nets.pytorch_backend.frontend.resnet1d import audio_resnet
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E_HF_ViT(torch.nn.Module):
    """
    E2E ASR model using Hugging Face pretrained ViT frontend.
    Similar to the ResNet-based E2E but uses pretrained ViT for video processing.
    """
    
    def __init__(
        self, 
        odim, 
        modality="video",
        ctc_weight=0.1, 
        ignore_id=-1,
        vit_model_name="google/vit-base-patch16-224",
        freeze_vit=False,
        frontend_output_dim=512,
        encoder_dim=768
    ):
        super().__init__()

        self.modality = modality
        
        # Frontend selection
        if modality == "audio":
            self.frontend = audio_resnet()
            frontend_dim = 512  # ResNet output
        elif modality == "video":
            if freeze_vit:
                from espnet.nets.pytorch_backend.frontend.hf_vit import HF_ViT_Frontend
                self.frontend = HF_ViT_Frontend(
                    model_name=vit_model_name,
                    freeze_backbone=True,
                    output_dim=frontend_output_dim
                )
            else:
                from espnet.nets.pytorch_backend.frontend.hf_vit import HF_ViT_Frontend
                self.frontend = HF_ViT_Frontend(
                    model_name=vit_model_name,
                    freeze_backbone=False,
                    output_dim=frontend_output_dim
                )
            frontend_dim = frontend_output_dim
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Project from frontend to encoder dimension
        self.proj_encoder = torch.nn.Linear(frontend_dim, encoder_dim)

        # Conformer encoder (same as original)
        self.encoder = ConformerEncoder(
            attention_dim=encoder_dim,
            attention_heads=12,
            linear_units=encoder_dim * 4,  # 3072 for 768 dim
            num_blocks=12,
            cnn_module_kernel=31,
        )

        # Transformer decoder (same as original)
        self.decoder = TransformerDecoder(
            odim=odim,
            attention_dim=encoder_dim,
            attention_heads=12,
            linear_units=encoder_dim * 4,  # 3072 for 768 dim
            num_blocks=6,
        )

        # Token definitions
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # Loss functions
        self.ctc_weight = ctc_weight
        self.ctc = CTC(odim, encoder_dim, 0.1, reduce=True)
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)
        
        print(f"Initialized E2E_HF_ViT with:")
        print(f"  - Modality: {modality}")
        print(f"  - ViT model: {vit_model_name}")
        print(f"  - Freeze ViT: {freeze_vit}")
        print(f"  - Frontend dim: {frontend_dim}")
        print(f"  - Encoder dim: {encoder_dim}")

    def scorers(self):
        """Return scorer functions for beam search."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label):
        """
        Forward pass for training.
        
        Args:
            x: Input tensor 
            lengths: Sequence lengths
            label: Target labels
            
        Returns:
            loss, loss_ctc, loss_att, acc
        """
        # Handle length adjustment for audio
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        # Create padding mask
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        # Extract features using frontend
        x = self.frontend(x)  # (B, T, frontend_dim)
        
        # Project to encoder dimension
        x = self.proj_encoder(x)  # (B, T, encoder_dim)
        
        # Update lengths and padding mask for video (frontend might change sequence length)
        if self.modality == "video":
            actual_lengths = torch.full(
                (x.shape[0],), x.shape[1], 
                dtype=lengths.dtype, device=lengths.device
            )
            padding_mask = make_non_pad_mask(actual_lengths).to(x.device).unsqueeze(-2)
        else:
            actual_lengths = lengths
        
        # Conformer encoder
        x, _ = self.encoder(x, padding_mask)

        # CTC loss
        loss_ctc, ys_hat = self.ctc(x, actual_lengths, label)

        # Decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)

        # Combined loss
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Accuracy
        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
