#!/usr/bin/env python3
"""ELF model wrapper for decoder-basin research.

Provides three clean interfaces:
  - denoise(z, t, sc_cfg, cfg, cond) -> x_hat
  - decode(x, sc_cfg, cond) -> logits   (nonlinear shared-weight readout, NOT bare unembedding)
  - encode(token_ids) -> x              (frozen T5-small + latent normalization)

All checkpoints are loaded with strict=True. Any key mismatch raises immediately.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

# Add the existing ELF src/ to path so we can import its modules
_ELF_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_ELF_SRC) not in sys.path:
    sys.path.insert(0, str(_ELF_SRC))

from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.sampling_utils import (
    get_sampling_steps, restore_cond, restore_vx,
    net_out_to_v_x, _forward_sample,
)
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch


# Checkpoint name -> HF repo id mapping
_CHECKPOINT_MAP = {
    "ELF-B-owt":   "embedded-language-flows/ELF-B-owt-torch",
    "ELF-B-de-en": "embedded-language-flows/ELF-B-de-en-torch",
    "ELF-B-xsum":  "embedded-language-flows/ELF-B-xsum-torch",
    "ELF-M-owt":   "embedded-language-flows/ELF-M-owt-torch",
    "ELF-L-owt":   "embedded-language-flows/ELF-L-owt-torch",
}

# Model name and config defaults per checkpoint
_CHECKPOINT_CONFIGS = {
    "ELF-B-owt": dict(
        model="ELF-B", max_length=1024, encoder_model_name="t5-small",
        latent_mean=0.0, latent_std=0.2, denoiser_noise_scale=2.0,
        denoiser_p_mean=-1.5, denoiser_p_std=0.8, t_eps=0.05,
        self_cond_prob=0.5, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        bottleneck_dim=128,
    ),
    "ELF-B-de-en": dict(
        model="ELF-B", max_length=128, max_input_length=64,
        encoder_model_name="t5-small",
        latent_mean=0.0, latent_std=0.2, denoiser_noise_scale=2.0,
        denoiser_p_mean=-1.5, denoiser_p_std=0.8, t_eps=0.05,
        self_cond_prob=0.5, label_drop_prob=0.1,
        num_time_tokens=4, num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4, bottleneck_dim=128,
        pad_token="eos",
    ),
    "ELF-B-xsum": dict(
        model="ELF-B", max_length=512, max_input_length=256,
        encoder_model_name="t5-small",
        latent_mean=0.0, latent_std=0.2, denoiser_noise_scale=2.0,
        denoiser_p_mean=-1.5, denoiser_p_std=0.8, t_eps=0.05,
        self_cond_prob=0.5, label_drop_prob=0.1,
        num_time_tokens=4, num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4, bottleneck_dim=128,
        pad_token="eos",
    ),
    "ELF-M-owt": dict(
        model="ELF-M", max_length=1024, encoder_model_name="t5-small",
        latent_mean=0.0, latent_std=0.2, denoiser_noise_scale=2.0,
        denoiser_p_mean=-1.5, denoiser_p_std=0.8, t_eps=0.05,
        self_cond_prob=0.5, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        bottleneck_dim=128,
    ),
    "ELF-L-owt": dict(
        model="ELF-L", max_length=1024, encoder_model_name="t5-small",
        latent_mean=0.0, latent_std=0.2, denoiser_noise_scale=2.0,
        denoiser_p_mean=-1.5, denoiser_p_std=0.8, t_eps=0.05,
        self_cond_prob=0.5, num_time_tokens=4,
        num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        bottleneck_dim=128,
    ),
}


class SimpleConfig:
    """Minimal config object matching the interface expected by sampling/generation utils."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"SimpleConfig({items})"

    def config_hash(self) -> str:
        """Deterministic hash of all config fields for reproducibility."""
        d = {k: v for k, v in sorted(vars(self).items())}
        return hashlib.sha256(json.dumps(d, default=str).encode()).hexdigest()[:12]


def _download_checkpoint(checkpoint_name: str) -> Path:
    """Download a checkpoint from HuggingFace and return the local directory."""
    if checkpoint_name not in _CHECKPOINT_MAP:
        # Treat as a local path
        p = Path(checkpoint_name)
        if p.exists():
            return p
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}. "
                         f"Known: {list(_CHECKPOINT_MAP.keys())}")

    repo_id = _CHECKPOINT_MAP[checkpoint_name]
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id, repo_type="model")
    return Path(local_dir)


def _find_checkpoint_file(ckpt_dir: Path) -> Path:
    """Find the actual checkpoint file inside a directory."""
    if ckpt_dir.is_file():
        return ckpt_dir
    # Look for checkpoint_* files, sorted by step descending
    candidates = sorted(ckpt_dir.glob("checkpoint_*"),
                        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
                        reverse=True)
    if candidates:
        return candidates[0]
    # Fallback: any .pt or .bin file
    for ext in ("*.pt", "*.bin", "*.pth"):
        files = list(ckpt_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}")


class ELFWrapper:
    """High-level wrapper for ELF model with clean denoise/decode/encode API.

    Loads checkpoint with strict key matching.
    All forward passes use bf16 autocast by default.
    """

    def __init__(self, checkpoint_name: str, device: str = "cuda",
                 use_bf16: bool = True):
        """Load an ELF checkpoint with strict key matching.

        Args:
            checkpoint_name: One of the known names (e.g. 'ELF-B-owt',
                'ELF-B-de-en') or a local path to a checkpoint directory/file.
            device: 'cuda' or 'cpu'
            use_bf16: Whether to use bf16 autocast for forward passes.
        """
        self.device = torch.device(device)
        self.use_bf16 = use_bf16 and self.device.type == "cuda"
        self.checkpoint_name = checkpoint_name

        # Resolve config
        base_name = checkpoint_name
        if base_name in _CHECKPOINT_CONFIGS:
            cfg_dict = _CHECKPOINT_CONFIGS[base_name]
        else:
            # Try to infer from path name
            for known in _CHECKPOINT_CONFIGS:
                if known.lower().replace("-", "") in str(checkpoint_name).lower().replace("-", ""):
                    cfg_dict = _CHECKPOINT_CONFIGS[known]
                    base_name = known
                    break
            else:
                raise ValueError(f"Cannot infer config for checkpoint: {checkpoint_name}")

        self.config = SimpleConfig(**cfg_dict)
        self.config.use_bf16 = use_bf16

        # Load encoder
        print(f"[ELFWrapper] Loading T5 encoder: {self.config.encoder_model_name}")
        enc_config, self.encoder = get_encoder(self.config.encoder_model_name, torch.float32)
        self.encoder = self.encoder.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder_dim = enc_config.d_model  # 512 for t5-small

        # Build model
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.encoder_model_name)
        self.vocab_size = self.tokenizer.vocab_size

        model_name = self.config.model
        print(f"[ELFWrapper] Creating {model_name} model (vocab={self.vocab_size})")
        self.model = ELF_models[model_name](
            text_encoder_dim=self.text_encoder_dim,
            max_length=self.config.max_length,
            attn_drop=0.0, proj_drop=0.0,
            num_time_tokens=self.config.num_time_tokens,
            num_self_cond_cfg_tokens=self.config.num_self_cond_cfg_tokens,
            num_model_mode_tokens=self.config.num_model_mode_tokens,
            bottleneck_dim=self.config.bottleneck_dim,
            vocab_size=self.vocab_size,
        )

        # Load checkpoint with STRICT key matching
        print(f"[ELFWrapper] Downloading/locating checkpoint: {checkpoint_name}")
        ckpt_dir = _download_checkpoint(checkpoint_name)
        ckpt_file = _find_checkpoint_file(ckpt_dir)
        print(f"[ELFWrapper] Loading checkpoint file: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        # Prefer EMA params (what released checkpoints use for inference)
        if "ema_params1" in ckpt and ckpt["ema_params1"]:
            state_dict = ckpt["ema_params1"]
            print(f"[ELFWrapper] Using EMA parameters ({len(state_dict)} keys)")
        elif "params" in ckpt:
            state_dict = ckpt["params"]
            print(f"[ELFWrapper] Using raw parameters ({len(state_dict)} keys)")
        else:
            # Might be a raw state_dict
            state_dict = ckpt
            print(f"[ELFWrapper] Using checkpoint as raw state_dict ({len(state_dict)} keys)")

        # STRICT loading — any mismatch is reported and raised
        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing or unexpected:
            msg = "[ELFWrapper] STRICT KEY MISMATCH — STOPPING.\n"
            if missing:
                msg += f"  Missing keys ({len(missing)}): {sorted(missing)[:10]}...\n"
            if unexpected:
                msg += f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}...\n"
            print(msg)
            raise RuntimeError(msg)

        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ELFWrapper] ✓ Loaded {checkpoint_name} — "
              f"{n_params/1e6:.1f}M params, strict=True, device={self.device}")

        # Store checkpoint step for logging
        self.checkpoint_step = ckpt.get("step", "unknown")

    # ─── Core API ───────────────────────────────────────────────────────

    @torch.no_grad()
    def denoise(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        sc_cfg: float = 1.0,
        x_pred_prev: Optional[torch.Tensor] = None,
        cond_seq: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the denoiser to predict clean state x_hat from noisy z at time t.

        Args:
            z: Noisy latent (B, L, D)
            t: Scalar time in [0, 1] (will be broadcast to batch)
            sc_cfg: Self-conditioning CFG scale
            x_pred_prev: Previous clean-state prediction for self-conditioning (B, L, D)
            cond_seq: Conditioning sequence (B, L, D), for conditional models
            cond_seq_mask: Binary mask (B, L), 1 = conditioning position

        Returns:
            x_hat: Predicted clean state (B, L, D)
        """
        B = z.shape[0]
        if cond_seq is None:
            cond_seq = torch.zeros_like(z)
            cond_seq_mask = torch.zeros((B, z.shape[1]), dtype=z.dtype, device=z.device)

        t_batch = torch.full((B,), float(t), dtype=z.dtype, device=z.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            # Build input with self-conditioning
            if self.config.self_cond_prob > 0 or self.config.num_self_cond_cfg_tokens > 0:
                if x_pred_prev is None:
                    x_pred_prev = restore_cond(
                        torch.zeros_like(z), cond_seq, cond_seq_mask)
                z_input = torch.cat([z, x_pred_prev], dim=-1)
                sc_batch = (
                    torch.full((B,), float(sc_cfg), dtype=z.dtype, device=z.device)
                    if self.config.num_self_cond_cfg_tokens > 0 else None
                )
                net_out = self.model(
                    z_input, t_batch, deterministic=True,
                    self_cond_cfg_scale=sc_batch,
                )
            else:
                net_out = self.model(z, t_batch, deterministic=True)

            _, x_hat = net_out_to_v_x(net_out, z, t_batch, self.config.t_eps)
            x_hat = restore_cond(x_hat, cond_seq, cond_seq_mask)

        return x_hat

    @torch.no_grad()
    def decode(
        self,
        x: torch.Tensor,
        sc_cfg: float = 1.0,
        cond_seq: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent states to token logits using the FULL nonlinear decoder.

        This uses mode='decode' (decoder_step_active=True), t=1, and the real
        shared-weight readout:
            hidden -> GELU(x @ proj_kernel + proj_bias) -> result @ unembed_kernel + unembed_bias

        NOT a bare unembedding matrix multiply. The distinction matters for every
        margin measurement in this project.

        Args:
            x: Clean or near-clean latent states (B, L, D)
            sc_cfg: Self-conditioning CFG scale
            cond_seq: Conditioning sequence for conditional models
            cond_seq_mask: Binary mask for conditioning positions

        Returns:
            logits: (B, L, V) token logits
        """
        B = x.shape[0]
        t_final = torch.ones((B,), dtype=x.dtype, device=x.device)
        sc_batch = (
            torch.full((B,), float(sc_cfg), dtype=x.dtype, device=x.device)
            if self.config.num_self_cond_cfg_tokens > 0 else None
        )

        # Build self-conditioning input: at decode time (t=1), there is no
        # prior prediction, so self-cond slot is zeros (matches _dlm_decode_batch)
        if self.config.self_cond_prob > 0 or self.config.num_self_cond_cfg_tokens > 0:
            x_input = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            x_input = x

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            _, decoder_logits = self.model(
                x_input, t_final, deterministic=True,
                self_cond_cfg_scale=sc_batch,
                decoder_step_active=True,
            )

        if decoder_logits is None:
            raise RuntimeError(
                "decoder_logits is None — model did not produce decoder output. "
                "Check that decoder_step_active=True is being honored and "
                "num_model_mode_tokens > 0."
            )

        return decoder_logits.float()  # Always return float32 logits for precision

    @torch.no_grad()
    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode token IDs to latent space via frozen T5 + latent normalization.

        Args:
            token_ids: (B, L) integer token IDs
            attention_mask: (B, L) binary mask, 1 = valid token

        Returns:
            x: (B, L, D) normalized latent embeddings
        """
        token_ids = token_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            raw_embeddings = self.encoder(
                input_ids=token_ids,
                attention_mask=attention_mask,
                deterministic=True,
            )

        # Apply ELF's latent normalization: (x - mean) / std
        x = (raw_embeddings.float() - self.config.latent_mean) / self.config.latent_std
        return x

    # ─── Convenience methods ────────────────────────────────────────────

    def decode_to_tokens(
        self,
        x: torch.Tensor,
        sc_cfg: float = 1.0,
        cond_seq: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent states to token IDs (argmax)."""
        logits = self.decode(x, sc_cfg=sc_cfg, cond_seq=cond_seq,
                             cond_seq_mask=cond_seq_mask)
        return logits.argmax(dim=-1)

    def decode_to_text(
        self,
        x: torch.Tensor,
        sc_cfg: float = 1.0,
        cond_seq: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
        skip_special_tokens: bool = True,
    ) -> list:
        """Decode latent states to text strings."""
        token_ids = self.decode_to_tokens(x, sc_cfg=sc_cfg, cond_seq=cond_seq,
                                          cond_seq_mask=cond_seq_mask)
        return self.tokenizer.batch_decode(
            token_ids.cpu(), skip_special_tokens=skip_special_tokens)

    def margin(self, x: torch.Tensor, sc_cfg: float = 1.0) -> torch.Tensor:
        """Compute the decoder margin m_i(x) = g_top1 - g_top2 at each position.

        This is the central measurement object for the project.

        Returns:
            margins: (B, L) margin values (always positive for correct decode)
        """
        logits = self.decode(x, sc_cfg=sc_cfg)  # (B, L, V)
        top2 = logits.topk(2, dim=-1).values     # (B, L, 2)
        return top2[:, :, 0] - top2[:, :, 1]      # (B, L)

    def margin_and_top_k(
        self, x: torch.Tensor, sc_cfg: float = 1.0, k: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute margin plus top-k logits and IDs.

        Returns:
            margins: (B, L)
            top_k_logits: (B, L, k)
            top_k_ids: (B, L, k)
        """
        logits = self.decode(x, sc_cfg=sc_cfg)
        top_k = logits.topk(k, dim=-1)
        margins = top_k.values[:, :, 0] - top_k.values[:, :, 1]
        return margins, top_k.values, top_k.indices

    @property
    def d_model(self) -> int:
        return self.text_encoder_dim

    @property
    def max_length(self) -> int:
        return self.config.max_length

    def decode_conditional_to_texts(
        self,
        x: torch.Tensor,
        sc_cfg: float = 1.0,
        cond_length: int = 64,
    ) -> list:
        """Decode latent states to text for CONDITIONAL models (De-En).

        Matches the Phase 0 pipeline exactly:
        1. Decode x -> logits -> argmax IDs
        2. shift_left to strip condition-side tokens
        3. mask_after_eos to zero out tokens after EOS
        4. tokenizer.decode with skip_special_tokens

        This is the CORRECT way to get hypothesis strings for BLEU.
        Do NOT use raw `ids[:, 64:128]` slicing — it omits EOS masking.
        """
        from utils.generation_utils import mask_after_eos, shift_left

        logits = self.decode(x, sc_cfg=sc_cfg)
        all_ids = logits.argmax(dim=-1)  # (B, L)
        B = all_ids.shape[0]
        gen_length = self.config.max_length - cond_length

        # shift_left expects per-sample condition lengths
        cond_len_per_sample = torch.full(
            (B,), cond_length, dtype=torch.int32, device=all_ids.device
        )
        predicted_ids = shift_left(all_ids, cond_len_per_sample, 0)[:, :gen_length]
        predicted_ids = mask_after_eos(
            predicted_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        )

        texts = []
        for i in range(B):
            text = self.tokenizer.decode(
                predicted_ids[i].cpu().numpy(), skip_special_tokens=True
            )
            texts.append(text)
        return texts

    def __repr__(self):
        return (f"ELFWrapper(checkpoint={self.checkpoint_name}, "
                f"model={self.config.model}, L={self.max_length}, "
                f"d={self.d_model}, device={self.device})")
