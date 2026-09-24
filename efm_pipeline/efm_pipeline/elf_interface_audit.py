#!/usr/bin/env python3
"""ELF Interface Audit — Inspect real tensor shapes, forward pass, and decoder.

This script MUST be run before implementing efm_model.py.  It loads the
ELF-B model and traces the exact tensor shapes at every stage:

    1. T5 encoder output shape and normalisation
    2. Model forward pass (flow matching branch): input → bottleneck → blocks → output
    3. Model forward pass (decoder branch): input → proj → unembed → logits
    4. Sampling loop: ODE/SDE step function signatures
    5. Training objective shapes (flow matching L2 + CE decode)

The output is a structured JSON report that informs how the EFM wrapper must
adapt to the real ELF interfaces.

Usage:
    python -m efm_pipeline.elf_interface_audit [--checkpoint PATH] [--device cuda]
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

# ELF source.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ELF_SRC = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)

from modules.model import ELF_models, ELF
from modules.layers import TimestepEmbedder, BottleneckTextProj, FinalLayer
from configs.config import Config
from utils.sampling_utils import (
    add_noise, sample_timesteps, get_sampling_steps, net_out_to_v_x,
    _ode_step, _sde_step,
)

sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..")))
from efm_pipeline.utils import get_device, device_summary, ensure_dir


def audit_model_architecture(model: ELF):
    """Inspect model architecture and parameter shapes."""
    report = {"architecture": {}}
    arch = report["architecture"]

    arch["class"] = model.__class__.__name__
    arch["text_encoder_dim"] = model.text_encoder_dim
    arch["hidden_size"] = model.hidden_size
    arch["depth"] = model.depth
    arch["num_heads"] = model.num_heads
    arch["max_length"] = model.max_length
    arch["bottleneck_dim"] = model.bottleneck_dim
    arch["vocab_size"] = model.vocab_size
    arch["num_time_tokens"] = model.num_time_tokens
    arch["num_self_cond_cfg_tokens"] = model.num_self_cond_cfg_tokens
    arch["num_model_mode_tokens"] = model.num_model_mode_tokens

    # Parameter counts per component.
    components = {}
    for name, p in model.named_parameters():
        component = name.split(".")[0]
        if component not in components:
            components[component] = {"params": 0, "shapes": {}}
        components[component]["params"] += p.numel()
        components[component]["shapes"][name] = list(p.shape)
    arch["components"] = {k: {"num_params": v["params"]} for k, v in components.items()}
    arch["total_params"] = sum(p.numel() for p in model.parameters())

    # Key weight shapes.
    arch["key_shapes"] = {}
    for name in [
        "text_proj.proj1.weight", "text_proj.proj2.weight",   # Bottleneck
        "self_cond_proj.weight",                               # Self-conditioning
        "t_emb_tokens",                                        # Time prefix
        "final_layer.linear.weight",                           # Flow output
        "proj_kernel", "unembed_kernel",                       # Decoder
    ]:
        p = dict(model.named_parameters()).get(name)
        if p is not None:
            arch["key_shapes"][name] = list(p.shape)

    return report


def audit_forward_pass(model: ELF, device: torch.device):
    """Trace tensor shapes through the forward pass."""
    report = {"forward_pass": {}}
    fwd = report["forward_pass"]

    B, S, C = 2, 64, model.text_encoder_dim  # Batch, Seq, Encoder dim
    H = model.hidden_size

    # --- Input preparation ---
    x = torch.randn(B, S, C, device=device)  # Simulated T5 output
    t = torch.rand(B, device=device)          # Timesteps

    fwd["input"] = {
        "x_shape": [B, S, C],
        "t_shape": [B],
        "x_dtype": str(x.dtype),
        "note": "x is T5 encoder output (or noised latent), t is scalar per sample",
    }

    # --- Bottleneck projection ---
    with torch.no_grad():
        x_proj = model.text_proj(x.float())
    fwd["after_bottleneck"] = {
        "shape": list(x_proj.shape),
        "note": f"text_encoder_dim ({C}) → bottleneck_dim ({model.bottleneck_dim}) → hidden_size ({H})",
    }

    # --- Self-conditioning input ---
    x_selfcond = torch.randn(B, S, 2 * C, device=device)
    with torch.no_grad():
        x_sc_proj = model.self_cond_proj(x_selfcond.float())
    fwd["self_cond_projection"] = {
        "input_shape": list(x_selfcond.shape),
        "output_shape": list(x_sc_proj.shape),
        "note": f"2*text_encoder_dim ({2*C}) → text_encoder_dim ({C})",
    }

    # --- Time embedding + prefix tokens ---
    with torch.no_grad():
        time_emb = model.t_embedder(t)
    fwd["time_embedding"] = {
        "t_shape": [B],
        "emb_shape": list(time_emb.shape),
        "t_emb_tokens_shape": list(model.t_emb_tokens.shape),
        "note": f"Scalar t → sinusoidal ({model.t_embedder.frequency_embedding_size}d) → MLP → {H}d",
    }

    # --- Full forward (flow matching branch) ---
    with torch.no_grad():
        flow_out, decoder_logits = model(x, t, decoder_step_active=None,
                                         self_cond_cfg_scale=torch.zeros(B, device=device))
    fwd["flow_output"] = {
        "shape": list(flow_out.shape),
        "decoder_logits": "None (decoder_step_active=None)",
        "note": f"Output shape matches input: (B, S, text_encoder_dim={C})",
    }

    # --- Full forward (decoder branch) ---
    with torch.no_grad():
        flow_out2, dec_logits = model(x, t, decoder_step_active=True,
                                      self_cond_cfg_scale=torch.zeros(B, device=device))
    fwd["decoder_output"] = {
        "flow_shape": list(flow_out2.shape),
        "logits_shape": list(dec_logits.shape) if dec_logits is not None else None,
        "note": f"Logits: (B, S, vocab_size={model.vocab_size})",
    }

    # --- Decoder unembedding path ---
    fwd["decoder_path"] = {
        "proj_kernel_shape": list(model.proj_kernel.shape),
        "unembed_kernel_shape": list(model.unembed_kernel.shape),
        "note": f"hidden({H}) → GeLU(proj → {C}) → unembed → vocab({model.vocab_size})",
    }

    # --- With self-conditioning input ---
    x_2c = torch.randn(B, S, 2 * C, device=device)
    with torch.no_grad():
        flow_out_sc, _ = model(x_2c, t, decoder_step_active=None,
                               self_cond_cfg_scale=torch.zeros(B, device=device))
    fwd["with_self_cond"] = {
        "input_shape": list(x_2c.shape),
        "output_shape": list(flow_out_sc.shape),
        "note": f"When x is 2×C, self_cond_proj projects it to C first",
    }

    # --- With attention mask ---
    attn_mask = torch.ones(B, S, device=device)
    attn_mask[:, S//2:] = 0  # Mask second half.
    with torch.no_grad():
        flow_out_masked, _ = model(x, t, attention_mask=attn_mask,
                                   self_cond_cfg_scale=torch.zeros(B, device=device))
    fwd["with_attention_mask"] = {
        "mask_shape": list(attn_mask.shape),
        "output_shape": list(flow_out_masked.shape),
        "note": "attention_mask: 1=valid, 0=padded. Same shape as output.",
    }

    return report


def audit_sampling_interface(model: ELF, config: Config, device: torch.device):
    """Inspect the sampling loop interfaces."""
    report = {"sampling": {}}
    samp = report["sampling"]

    B, S, C = 2, 64, model.text_encoder_dim

    # --- Time schedule ---
    t_steps = get_sampling_steps(32, time_schedule="logit_normal", device=device)
    samp["time_schedule"] = {
        "num_steps": 32,
        "t_steps_shape": list(t_steps.shape),
        "t_range": [float(t_steps[0]), float(t_steps[-1])],
        "first_5": [round(float(t_steps[i]), 4) for i in range(min(5, len(t_steps)))],
    }

    # --- add_noise ---
    x0 = torch.randn(B, S, C, device=device)
    noise = torch.randn(B, S, C, device=device)
    t = torch.tensor([0.5, 0.8], device=device)
    z = add_noise(x0, noise, t, config)
    samp["add_noise"] = {
        "x0_shape": list(x0.shape),
        "noise_shape": list(noise.shape),
        "t_shape": list(t.shape),
        "z_shape": list(z.shape),
        "formula": "z = t*x0 + (1-t)*noise*noise_scale",
    }

    # --- net_out_to_v_x ---
    net_out = torch.randn(B, S, C, device=device)
    v, x_pred = net_out_to_v_x(net_out, z, t)
    samp["net_out_to_v_x"] = {
        "net_out_shape": list(net_out.shape),
        "v_shape": list(v.shape),
        "x_pred_shape": list(x_pred.shape),
        "note": "v = (x_pred - z) / (1 - t); x_pred = net_out",
    }

    # --- ODE step signature ---
    samp["ode_step_signature"] = {
        "args": "model, z, t, t_next, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask",
        "returns": "(z_next, x_pred)",
        "z_shape": f"(B, S, {C})",
        "t_type": "float scalar",
    }

    # --- SDE step signature ---
    samp["sde_step_signature"] = {
        "args": "model, z, t, t_next, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, gamma, generator",
        "returns": "(z_next, x_pred)",
        "extra": "gamma controls noise level; generator for reproducibility",
    }

    return report


def audit_training_objective(model: ELF, config: Config, device: torch.device):
    """Inspect the training loss computation shapes."""
    report = {"training": {}}
    tr = report["training"]

    B, S, C = 2, 64, model.text_encoder_dim

    # --- Flow matching loss ---
    x0 = torch.randn(B, S, C, device=device)
    noise = torch.randn(B, S, C, device=device)
    t = sample_timesteps(B, P_mean=0.8, P_std=0.8, time_schedule="logit_normal", device=device)
    z = add_noise(x0, noise, t, config)

    with torch.no_grad():
        flow_out, decoder_logits = model(z, t, decoder_step_active=True,
                                         self_cond_cfg_scale=torch.zeros(B, device=device))

    # Flow matching: L2 between predicted x0 and actual x0.
    flow_loss = F.mse_loss(flow_out, x0)

    tr["flow_matching_loss"] = {
        "prediction_shape": list(flow_out.shape),
        "target_shape": list(x0.shape),
        "loss_type": "MSE (L2) between predicted x0 and ground truth x0",
        "loss_value": float(flow_loss),
    }

    # Decoder CE loss.
    if decoder_logits is not None:
        fake_targets = torch.randint(0, model.vocab_size, (B, S), device=device)
        ce_loss = F.cross_entropy(
            decoder_logits.reshape(-1, model.vocab_size),
            fake_targets.reshape(-1),
        )
        tr["decode_loss"] = {
            "logits_shape": list(decoder_logits.shape),
            "targets_shape": list(fake_targets.shape),
            "loss_type": "Cross-entropy (flat vocab)",
            "loss_value": float(ce_loss),
        }

    tr["combined_objective"] = {
        "formula": "loss = flow_loss_weight * L2(flow_pred, x0) + decode_loss_weight * CE(logits, tokens)",
        "note": "ELF uses decoder_prob to stochastically select CE vs L2 branch per sample.",
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="ELF Interface Audit")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device_summary(device)}")

    # Build a minimal ELF-B model (no checkpoint needed — we just inspect shapes).
    config = Config()
    config.model = "ELF-B"
    config.max_length = 64
    config.bottleneck_dim = 128
    config.num_time_tokens = 4
    config.num_self_cond_cfg_tokens = 4
    config.num_model_mode_tokens = 4
    config.self_cond_prob = 0.5
    config.t_eps = 5e-2
    config.denoiser_noise_scale = 1.0

    model_fn = ELF_models["ELF-B"]
    model = model_fn(
        text_encoder_dim=512, max_length=64, bottleneck_dim=128,
        num_time_tokens=4, num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
        vocab_size=32128,
    ).to(device).eval()

    print(f"\n{'='*70}")
    print("  ELF-B INTERFACE AUDIT")
    print(f"{'='*70}\n")

    # Run all audits.
    full_report = {}
    full_report.update(audit_model_architecture(model))
    full_report.update(audit_forward_pass(model, device))
    full_report.update(audit_sampling_interface(model, config, device))
    full_report.update(audit_training_objective(model, config, device))

    # Print human-readable summary.
    print("=== Architecture ===")
    arch = full_report["architecture"]
    print(f"  Model: {arch['class']}")
    print(f"  Depth: {arch['depth']} layers, Hidden: {arch['hidden_size']}, Heads: {arch['num_heads']}")
    print(f"  Text encoder dim: {arch['text_encoder_dim']}, Bottleneck: {arch['bottleneck_dim']}")
    print(f"  Vocab: {arch['vocab_size']}, Total params: {arch['total_params']:,}")
    print(f"  Prefix tokens: time={arch['num_time_tokens']}, sc_cfg={arch['num_self_cond_cfg_tokens']}, mode={arch['num_model_mode_tokens']}")
    print()

    print("=== Forward Pass ===")
    fwd = full_report["forward_pass"]
    print(f"  Input:          {fwd['input']['x_shape']}  (B, S, text_encoder_dim)")
    print(f"  After bottleneck: {fwd['after_bottleneck']['shape']}  (B, S, hidden_size)")
    print(f"  Flow output:    {fwd['flow_output']['shape']}  (B, S, text_encoder_dim)")
    print(f"  Decoder logits: {fwd['decoder_output']['logits_shape']}  (B, S, vocab_size)")
    print(f"  Self-cond in:   {fwd['with_self_cond']['input_shape']}  (B, S, 2×text_encoder_dim)")
    print()

    print("=== Decoder Path ===")
    dp = fwd["decoder_path"]
    print(f"  proj_kernel:    {dp['proj_kernel_shape']}  (hidden → text_enc_dim)")
    print(f"  unembed_kernel: {dp['unembed_kernel_shape']}  (text_enc_dim → vocab)")
    print(f"  Path: {dp['note']}")
    print()

    print("=== Sampling ===")
    samp = full_report["sampling"]
    print(f"  t_steps shape:  {samp['time_schedule']['t_steps_shape']}  (num_steps+1 values in [0,1])")
    print(f"  add_noise:      z = t·x0 + (1-t)·noise·scale")
    print(f"  ODE step:       {samp['ode_step_signature']['returns']}")
    print(f"  SDE step:       adds gamma-controlled noise + corrective drift")
    print()

    print("=== Training Objective ===")
    tr = full_report["training"]
    print(f"  Flow loss:      MSE({tr['flow_matching_loss']['prediction_shape']}, {tr['flow_matching_loss']['target_shape']})")
    if "decode_loss" in tr:
        print(f"  Decode loss:    CE({tr['decode_loss']['logits_shape']} → {tr['decode_loss']['targets_shape']})")
    print(f"  Combined:       {tr['combined_objective']['formula']}")
    print()

    # Save JSON report.
    out_dir = ensure_dir(os.path.join(_THIS_DIR, "..", "results"))
    out_path = os.path.join(out_dir, "elf_interface_audit.json")
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"Full audit report saved → {out_path}")

    print(f"\n{'='*70}")
    print("  KEY FINDINGS FOR EFM WRAPPER DESIGN")
    print(f"{'='*70}")
    print(f"""
  1. INPUT:  (B, S, {arch['text_encoder_dim']}) — T5 encoder output, normalised.
             Self-cond input is (B, S, {2*arch['text_encoder_dim']}) = [z, x_pred].
  2. BOTTLENECK: {arch['text_encoder_dim']} → {arch['bottleneck_dim']} → {arch['hidden_size']} via BottleneckTextProj.
  3. TIME:   Global scalar t per sample → sinusoidal embed → MLP → {arch['hidden_size']}d → added to prefix tokens.
             EFM local time τ_i must produce *per-token* conditioning, injected AFTER bottleneck,
             added to the hidden states at each position (not prefix tokens).
  4. OUTPUT: Flow head returns (B, S, {arch['text_encoder_dim']}) — predicted clean x0.
             Decoder head returns (B, S, {arch['vocab_size']}) — logits.
  5. BLOCKS: {arch['depth']} ELFBlock layers. Each block: RMSNorm → Attention(RoPE) → RMSNorm → SwiGLU.
             For EFM, we reuse these blocks exactly.
  6. DECODER: hidden({arch['hidden_size']}) → GeLU(proj→{arch['text_encoder_dim']}) → unembed→vocab({arch['vocab_size']}).
             For EFM, reuse this decoder as-is.
  7. EXPAND: New tokens must be inserted as (S' > S) embeddings of dim {arch['text_encoder_dim']}.
             The expand operator operates in T5-embedding space, before the bottleneck.
""")


if __name__ == "__main__":
    main()
