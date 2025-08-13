import torch
import torchaudio
import json
import os
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_hf_vit import E2E_HF_ViT
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


class ModelModule_HF_ViT(LightningModule):
    """
    Lightning module for Hugging Face pretrained ViT-based ASR model.
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        # Initialize HF ViT-based E2E model
        self.model = E2E_HF_ViT(
            odim=len(self.token_list),
            modality=getattr(args, "modality", "video"),
            ctc_weight=getattr(args, "ctc_weight", 0.1),
            vit_model_name=getattr(args, "vit_model_name", "google/vit-base-patch16-224"),
            freeze_vit=getattr(args, "freeze_vit", False),
            frontend_output_dim=getattr(args, "frontend_output_dim", 512),
            encoder_dim=getattr(args, "encoder_dim", 768)
        )

        # Load pretrained weights if provided
        if getattr(args, "pretrained_model_path", None):
            try:
                ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
                if "state_dict" in ckpt:
                    model_state = ckpt["state_dict"]
                    clean_state = {k.replace("model.", ""): v for k, v in model_state.items()}
                    self.model.load_state_dict(clean_state, strict=False)
                    print("Pretrained weights loaded successfully.")
                else:
                    self.model.load_state_dict(ckpt, strict=False)
                    print("Pretrained model loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

    def configure_optimizers(self):
        # Use different learning rates if ViT is frozen
        if getattr(self.args, "freeze_vit", False):
            # Only optimize non-frozen parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"Optimizing {len(params)} parameters (ViT frozen)")
        else:
            params = self.model.parameters()
            print("Optimizing all parameters (ViT unfrozen)")
        
        optimizer = torch.optim.AdamW(
            params, 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay, 
            betas=(0.9, 0.98)
        )
        
        scheduler = WarmupCosineScheduler(
            optimizer, 
            self.args.warmup_epochs, 
            self.args.max_epochs, 
            len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def forward(self, sample):
        """Forward pass for inference."""
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        
        # Extract features and decode
        x = self.model.frontend(sample.unsqueeze(0))
        x = self.model.proj_encoder(x)
        
        # Create padding mask
        padding_mask = torch.ones(1, 1, x.shape[1], device=x.device, dtype=torch.bool)
        enc_feat, _ = self.model.encoder(x, padding_mask)
        enc_feat = enc_feat.squeeze(0)
        
        # Beam search decoding
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        """Test step for evaluation."""
        # Extract features and decode
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        
        # Create padding mask
        padding_mask = torch.ones(1, 1, x.shape[1], device=x.device, dtype=torch.bool)
        enc_feat, _ = self.model.encoder(x, padding_mask)
        enc_feat = enc_feat.squeeze(0)
        
        # Beam search decoding
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        # Get actual target
        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        # Store results
        self.test_results.append({
            "sample_idx": sample_idx,
            "predicted": predicted,
            "actual": actual,
            "file_path": sample.get("file_path", f"sample_{sample_idx}")
        })

        # Compute metrics
        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch["inputs"].size(0)
        batch_sizes = self.all_gather(batch_size)
        loss *= batch_sizes.size(0) / batch_sizes.sum()

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        return loss

    def _step(self, batch, batch_idx, step_type):
        """Common step function for training and validation."""
        loss, loss_ctc, loss_att, acc = self.model(
            batch["inputs"], 
            batch["input_lengths"], 
            batch["targets"]
        )
        batch_size = len(batch["inputs"])

        # Log metrics
        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_test_epoch_start(self):
        """Initialize test metrics."""
        self.total_length = 0
        self.total_edit_distance = 0
        self.test_results = []
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        """Compute and save test results."""
        wer = self.total_edit_distance / self.total_length
        self.log("wer", wer)
        
        # Save results to file
        output_file = getattr(self.args, 'output_json', 'test_results_hf_vit.json')
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        results = {
            "wer": float(wer),
            "total_samples": len(self.test_results),
            "total_edit_distance": float(self.total_edit_distance),
            "total_length": float(self.total_length),
            "model_type": "HF_ViT",
            "model_config": {
                "vit_model_name": getattr(self.args, "vit_model_name", "google/vit-base-patch16-224"),
                "freeze_vit": getattr(self.args, "freeze_vit", False),
                "frontend_output_dim": getattr(self.args, "frontend_output_dim", 512),
                "encoder_dim": getattr(self.args, "encoder_dim", 768),
            },
            "predictions": self.test_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"HF_ViT Results saved to: {output_file}")
        print(f"HF_ViT WER: {wer:.4f} ({self.total_edit_distance}/{self.total_length})")


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    """Create beam search decoder."""
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
