# TTMEncoder: wraps IBM TTM r2/r3 backbone, projects patch embeddings to (B, D, num_patches, H).

import warnings
import torch
import torch.nn as nn
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from tsfm_public import TinyTimeMixerForPrediction


class TTMEncoder(nn.Module):

    def __init__(
        self,
        model_path: str,
        hidden_dim: int,
        freeze_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
        revision: str = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        load_kwargs = {"revision": revision} if revision else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = TinyTimeMixerForPrediction.from_pretrained(model_path, **load_kwargs)
        self.backbone = full_model.backbone
        # Must be read before del full_model — not present on backbone.config alone
        self._needs_freq_token = getattr(self.backbone.config, "frequency_token_vocab_size", 0) > 0
        del full_model

        self.patch_size = self.backbone.config.patch_length

        ckpt_path  = hf_hub_download(model_path, filename="model.safetensors", **load_kwargs)
        current_sd = self.backbone.state_dict()
        remapped   = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            # r2 checkpoint: "backbone.*"  |  r3 checkpoint: "residual_forecaster.backbone.*"
            prefix = "backbone." if any(k.startswith("backbone.") for k in all_keys) \
                     else "residual_forecaster.backbone."
            for k in all_keys:
                if not k.startswith(prefix):
                    continue
                new_key = k.removeprefix(prefix)
                if new_key in current_sd:
                    tensor = f.get_tensor(k)
                    if tensor.shape == current_sd[new_key].shape:
                        remapped[new_key] = tensor
        self.backbone.load_state_dict(remapped, strict=False)
        print(f"[TTMEncoder] loaded {len(remapped)}/{len(current_sd)} pretrained weights  "
              f"(patch_size={self.patch_size}  d_model={self.backbone.config.d_model}  "
              f"num_layers={self.backbone.config.num_layers})")

        self.proj = nn.Linear(self.backbone.config.d_model, hidden_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            if unfreeze_last_n_layers > 0:
                mixers = self.backbone.encoder.mlp_mixer_encoder.mixers
                n = len(mixers)
                for layer in mixers[n - unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)
                unfrozen = sum(p.numel() for layer in mixers[n - unfreeze_last_n_layers:]
                               for p in layer.parameters())
                print(f"[TTMEncoder] unfrozen last {unfreeze_last_n_layers}/{n} layers ({unfrozen:,} params)")

    def _backbone_out(self, x: torch.Tensor) -> torch.Tensor:
        if self._needs_freq_token:
            freq_token = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.backbone(x, freq_token=freq_token).last_hidden_state
        return self.backbone(x).last_hidden_state

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Raw backbone output before proj: (B, D, num_patches, d_model)."""
        return self._backbone_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._backbone_out(x))
