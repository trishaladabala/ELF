#!/usr/bin/env python3
"""Stage 0 — ELF-B Baseline Reproduction.

Loads the pretrained ELF-B-owt checkpoint, generates unconditional samples,
and computes Gen-PPL + entropy.  These numbers become the reference baseline
for all subsequent experiments.

Usage:
    python -m efm_pipeline.experiments.stage0_reproduce_elf [--num_samples 512] [--num_steps 32]

Reused from ELF:
    modules.model        — ELF_B factory
    utils.sampling_utils — ODE/SDE stepping, time schedule
    utils.encoder_utils  — T5 encoder, encode_text
    utils.metrics_utils  — Metrics (GPT-2 Gen-PPL, entropy)
    utils.generation_utils — _generate_samples_single_batch, _dlm_decode_batch
    utils.train_utils    — TrainState, unwrap_model
    utils.checkpoint_utils — HuggingFace download + restore
"""

import argparse
import os
import sys

import torch
import numpy as np

# Ensure ELF source is importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ELF_SRC = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)

from modules.model import ELF_models
from configs.config import Config, SamplingConfig
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import (
    _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos,
)
from utils.metrics_utils import Metrics as PPLMetrics
from utils.train_utils import TrainState

# Pipeline utilities.
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "..")))
from efm_pipeline.utils import set_seed, get_device, device_summary, ensure_dir, Timer
from efm_pipeline.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Checkpoint loading (simplified: we only need model weights, not optimizer)
# ---------------------------------------------------------------------------

def load_elf_b_weights(device: torch.device, checkpoint_path: str = "embedded-language-flows/ELF-B-owt-torch"):
    """Load ELF-B model weights from HuggingFace or local path.

    Returns:
        model:     ELF-B nn.Module in eval mode on *device*.
        config:    ELF Config with matching defaults.
        tokenizer: T5 tokenizer.
        encoder:   Frozen T5-Small encoder on *device*.
    """
    from transformers import T5EncoderModel, AutoTokenizer

    # --- Config (matches ELF-B-owt defaults) ---
    config = Config()
    config.model = "ELF-B"
    config.encoder_model_name = "t5-small"
    config.max_length = 1024
    config.bottleneck_dim = 128
    config.num_time_tokens = 4
    config.num_self_cond_cfg_tokens = 4
    config.num_model_mode_tokens = 4
    config.self_cond_prob = 0.5
    config.t_eps = 5e-2
    config.label_drop_prob = 0.1
    config.latent_mean = 0.0
    config.latent_std = 1.0

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)

    # --- T5 encoder (frozen) ---
    print(f"Loading T5 encoder: {config.encoder_model_name}")
    encoder = T5EncoderModel.from_pretrained(config.encoder_model_name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    # ELF wraps the encoder to return .last_hidden_state and accept `deterministic`.
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, input_ids, attention_mask=None, deterministic=True):
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state
    encoder = EncoderWrapper(encoder).to(device)

    # --- ELF-B model ---
    print(f"Building ELF-B model...")
    model_fn = ELF_models[config.model]
    text_encoder_dim = encoder.enc.config.d_model  # 512 for T5-Small
    model = model_fn(
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        bottleneck_dim=config.bottleneck_dim,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        vocab_size=tokenizer.vocab_size,
    )

    # --- Load weights ---
    print(f"Loading checkpoint: {checkpoint_path}")
    resolved_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    ckpt = None

    # Try local first.
    if os.path.exists(resolved_path):
        if os.path.isdir(resolved_path):
            from utils.checkpoint_utils import find_latest_checkpoint
            latest = find_latest_checkpoint(resolved_path)
            if latest:
                ckpt = torch.load(latest, map_location="cpu")
        elif os.path.isfile(resolved_path):
            ckpt = torch.load(resolved_path, map_location="cpu")

    # Try HuggingFace.
    if ckpt is None:
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading from HuggingFace: {checkpoint_path}")
            local_dir = snapshot_download(repo_id=checkpoint_path, repo_type="model")
            from utils.checkpoint_utils import find_latest_checkpoint
            latest = find_latest_checkpoint(local_dir)
            if latest:
                ckpt = torch.load(latest, map_location="cpu")
            elif os.path.isfile(os.path.join(local_dir, "checkpoint")):
                ckpt = torch.load(os.path.join(local_dir, "checkpoint"), map_location="cpu")
        except Exception as e:
            print(f"HuggingFace download failed: {e}")

    if ckpt is None:
        raise RuntimeError(
            f"Could not load checkpoint from '{checkpoint_path}'. "
            "Ensure the path exists locally or is a valid HuggingFace repo ID."
        )

    # Load state dict (prefer EMA params if available).
    params = ckpt.get("ema_params1", ckpt.get("params", ckpt))
    if isinstance(params, dict) and all(isinstance(v, torch.Tensor) for v in params.values()):
        model.load_state_dict(params, strict=False)
        loaded_keys = set(params.keys())
        model_keys = set(model.state_dict().keys())
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        if missing:
            print(f"  Warning: {len(missing)} missing keys (e.g. {list(missing)[:3]})")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys (e.g. {list(unexpected)[:3]})")
    else:
        raise ValueError("Checkpoint does not contain a valid state dict.")

    step = ckpt.get("step", 0)
    print(f"  Loaded checkpoint at step {step}")

    model = model.to(device).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    return model, config, tokenizer, encoder


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model, config, tokenizer, encoder, device,
    num_samples: int = 512,
    num_steps: int = 32,
    batch_size: int = 16,
    method: str = "ode",
    sde_gamma: float = 1.5,
    seed: int = 42,
    max_length: int = 1024,
):
    """Generate unconditional text samples using the ELF-B model.

    Returns:
        texts:  List[str] of generated samples.
    """
    model.eval()
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    sampling_config = SamplingConfig()
    sampling_config.sampling_method = method
    sampling_config.num_sampling_steps = [num_steps]
    sampling_config.sde_gamma = sde_gamma
    sampling_config.time_schedule = "logit_normal"

    # Get text_encoder_dim from model.
    text_encoder_dim = model.text_encoder_dim

    all_texts = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Generating {num_samples} samples ({num_batches} batches of {batch_size})...")
    for batch_idx in range(num_batches):
        curr_bs = min(batch_size, num_samples - len(all_texts))
        if curr_bs <= 0:
            break

        # Sample initial noise.
        z = torch.randn(
            curr_bs, max_length, text_encoder_dim,
            generator=generator, dtype=torch.float32, device=gen_device,
        ).to(device) * config.denoiser_noise_scale

        # Build time schedule.
        t_steps = get_sampling_steps(
            num_steps, time_schedule="logit_normal",
            device=device, dtype=torch.float32,
        )

        # Run ODE/SDE sampling loop.
        z_final = _generate_samples_single_batch(
            model=model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=1.0,
            self_cond_cfg_scale=1.0,
        )

        # Decode to token IDs using the model's decoder head.
        predicted_ids = _dlm_decode_batch(
            z=z_final, model=model, t_final_val=t_steps[-1],
            config=config, self_cond_cfg_scale=1.0,
        )

        # Mask after EOS.
        eos_id = tokenizer.eos_token_id or 1
        pad_id = tokenizer.pad_token_id or 0
        predicted_ids = mask_after_eos(predicted_ids, eos_id, pad_id)

        # Decode to text.
        texts = tokenizer.batch_decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
        all_texts.extend(texts)

        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} done ({len(all_texts)} samples)")

    return all_texts[:num_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 0: ELF-B Baseline Reproduction")
    parser.add_argument("--checkpoint", type=str, default="embedded-language-flows/ELF-B-owt-torch",
                        help="HuggingFace repo ID or local path to ELF-B checkpoint.")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--method", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--sde_gamma", type=float, default=1.5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gen_ppl_model", type=str, default="gpt2-large")
    args = parser.parse_args()

    # Setup.
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device_summary(device)}")

    logger = ExperimentLogger(args.output_dir, "stage0_elf_baseline")
    timer = Timer().start()

    # Load model.
    model, config, tokenizer, encoder = load_elf_b_weights(device, args.checkpoint)

    # Generate samples.
    texts = generate_samples(
        model=model, config=config, tokenizer=tokenizer, encoder=encoder,
        device=device, num_samples=args.num_samples, num_steps=args.num_steps,
        batch_size=args.batch_size, method=args.method, sde_gamma=args.sde_gamma,
        seed=args.seed, max_length=args.max_length,
    )

    # Print a few samples.
    print("\n--- Sample outputs ---")
    for i, text in enumerate(texts[:5]):
        print(f"  [{i}] {text[:200]}{'...' if len(text) > 200 else ''}")
    print()

    # Compute metrics using ELF's own Metrics class.
    print("Computing Gen-PPL and entropy...")
    metrics = PPLMetrics(
        gen_ppl_eval_model_name_or_path=args.gen_ppl_model,
        eval_ppl_batch_size=args.batch_size,
    )
    eval_result = metrics.record_generative_perplexity(
        text_samples=texts, max_length=args.max_length, retokenize=True,
    )

    elapsed = timer.stop()

    result = {
        "method": args.method,
        "num_steps": args.num_steps,
        "sde_gamma": args.sde_gamma if args.method == "sde" else None,
        "num_samples": len(texts),
        "max_length": args.max_length,
        "gen_ppl": eval_result["ppl"],
        "mean_entropy": eval_result["mean_entropy"],
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "gen_ppl_model": args.gen_ppl_model,
        "device": str(device),
    }

    print(f"\n{'='*60}")
    print(f"  ELF-B Baseline Results")
    print(f"  Method:       {args.method}" + (f" (γ={args.sde_gamma})" if args.method == "sde" else ""))
    print(f"  Steps:        {args.num_steps}")
    print(f"  Samples:      {len(texts)}")
    print(f"  Gen-PPL:      {eval_result['ppl']:.2f}")
    print(f"  Entropy:      {eval_result['mean_entropy']:.4f}")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  (Published ELF-B reference: Gen-PPL ~20-25)")
    print(f"{'='*60}\n")

    logger.save_result(result)
    print("Done.")


if __name__ == "__main__":
    main()
