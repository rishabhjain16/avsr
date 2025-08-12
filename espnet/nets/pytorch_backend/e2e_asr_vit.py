#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from espnet.nets.pytorch_backend.frontend.video_vit import video_vit
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E_ViT(torch.nn.Module):
    def __init__(self, odim, ctc_weight=0.1, ignore_id=-1):
        super().__init__()

        # VideoViT frontend (outputs 384-dim features)
        self.frontend = video_vit()
        
        # No projection - use 384 dims throughout (this was working)
        self.decoder = TransformerDecoder(
            odim=odim,
            attention_dim=384,  # Match ViT output
            attention_heads=6,  # Adjusted for 384 dims
            linear_units=1536,  # 4 * attention_dim
            num_blocks=6,
        )

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # Loss functions matching 384 dims
        self.ctc_weight = ctc_weight
        self.ctc = CTC(odim, 384, 0.1, reduce=True)  # Match ViT output
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)

    def scorers(self):
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label):
        # Create padding mask based on original lengths
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        # Extract features using VideoViT
        x = self.frontend(x)  # (B, T, 384)
        
        # No projection - use 384 dims directly
        
        # Get actual sequence lengths after ViT processing
        actual_lengths = torch.full((x.shape[0],), x.shape[1], dtype=lengths.dtype, device=lengths.device)
        
        # CTC loss - use actual output sequence lengths
        loss_ctc, ys_hat = self.ctc(x, actual_lengths, label)

        # Decoder loss - use original padding mask
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
