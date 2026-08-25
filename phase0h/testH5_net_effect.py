#!/usr/bin/env python3
"""H5 — End-to-End Net Effect, Properly Powered.

Tests whether the full refinement pipeline improves naturally generated
toy-model output, measured with external-LM PPL and paired bootstrap CIs.
"""

import json
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF
from utils.sampling_utils import net_out_to_v_x, get_sampling_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
E_RESULTS = Path(__file__).resolve().parent.parent / "phase0e" / "results"
PHASE0C_RESULTS = Path(__file__).resolve().parent.parent / "phase0c" / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

T_EPS = 0.05
N_SAMPLES = 256
BATCH_SIZE = 64
N_BOOTSTRAP = 1000
ENTROPY_THRESHOLD_PERCENTILE = 80  # Only fix the worst 20% — conservative


def load_detokenizer():
    map_path = E_RESULTS / "detokenizer_map.json"
    with open(map_path, "r") as f:
        d_map = json.load(f)
    return {int(k): v for k, v in d_map.items()}


def decode_tokens_to_text(token_ids, detok_map):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid in detok_map:
            word = detok_map[tid]
            if word.startswith(' '):
                words.append(' ' + word[1:])
            else:
                words.append(word)
    return "".join(words).replace("  ", " ").strip()


def setup_external_lm():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"Loading distilgpt2 on {DEVICE}...")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE).eval()
    return model, tokenizer


def score_text(text, model, tokenizer, device=DEVICE):
    if not text.strip():
        return float('inf')
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float('inf')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


def load_real_checkpoint():
    ckpt_path = PHASE0C_RESULTS / "mini_elf_real_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ELF(
        text_encoder_dim=cfg["encoder_dim"], max_length=cfg["max_length"],
        hidden_size=cfg["hidden_size"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        bottleneck_dim=cfg["bottleneck_dim"], num_time_tokens=2,
        num_self_cond_cfg_tokens=0, num_model_mode_tokens=2,
        vocab_size=cfg["vocab_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, cfg


def ode_sample(model, n_samples, n_steps, encoder_dim, seq_len, device="cpu"):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z = torch.randn(n_samples, seq_len, encoder_dim, dtype=dtype, device=device)
        steps = get_sampling_steps(n_steps, time_schedule="uniform", device=device, dtype=dtype)
        for i in range(n_steps):
            t_curr, t_next = float(steps[i]), float(steps[i+1])
            h = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr, dtype=dtype, device=device)
            net_out, _ = model(z, t_batch, deterministic=True, decoder_step_active=None)
            v_pred, _ = net_out_to_v_x(net_out, z, t_batch, T_EPS)
            z = z + h * v_pred
    return z


def decode_batch(model, z, device):
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        t_final = torch.ones(z.shape[0], dtype=dtype, device=device)
        _, logits = model(z.to(device), t_final, deterministic=True,
                          decoder_step_active=torch.ones(z.shape[0], device=device))
        probs = F.softmax(logits.float(), dim=-1)
        tokens = logits.argmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return tokens.cpu().numpy(), entropy.cpu().numpy()


def apply_single_refinement_pass(model, z, noise_scale):
    """One pass: decode, flag high-entropy, re-corrupt flagged, re-decode."""
    tokens, entropy = decode_batch(model, z, DEVICE)
    thresh = np.percentile(entropy, ENTROPY_THRESHOLD_PERCENTILE)
    flagged = entropy > thresh

    z_new = z.clone()
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if flagged[i, j]:
                p = 0.3
                eps = torch.randn(z.shape[-1], device=z.device) * noise_scale
                z_new[i, j] = p * z[i, j] + (1 - p) * eps
    return z_new


def bootstrap_paired_ci(before, after, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI for paired difference (before - after). Positive = improvement."""
    rng = np.random.default_rng(42)
    min_len = min(len(before), len(after))
    diffs = np.array(before[:min_len]) - np.array(after[:min_len])  # positive = PPL decreased = improvement
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("H5 — End-to-End Net Effect (Properly Powered)")
    print(f"     N={N_SAMPLES} sequences, {N_BOOTSTRAP} bootstrap resamples")
    print("=" * 60)

    detok_map = load_detokenizer()
    lm_model, lm_tokenizer = setup_external_lm()
    elf_model, cfg = load_real_checkpoint()
    encoder_dim = cfg["encoder_dim"]
    seq_len = cfg["max_length"]

    # Generate natural outputs using ODE (the baseline sampler)
    print("\nGenerating natural model output (ODE, 16 steps)...")
    all_z = []
    for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
        bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
        torch.manual_seed(42 + batch_start)
        z = ode_sample(elf_model, bs, 16, encoder_dim, seq_len, device=DEVICE)
        all_z.append(z)
    z_original = torch.cat(all_z, dim=0)
    noise_scale = float(z_original.std())

    # Score BEFORE refinement
    print("Scoring before refinement...")
    before_tokens, before_entropy = decode_batch(elf_model, z_original, DEVICE)
    before_ppls = []
    for i in range(N_SAMPLES):
        text = decode_tokens_to_text(before_tokens[i], detok_map)
        ppl = score_text(text, lm_model, lm_tokenizer, DEVICE)
        if not math.isinf(ppl) and not math.isnan(ppl):
            before_ppls.append(ppl)
        else:
            before_ppls.append(50000.0)  # Cap for degenerate

    # Apply 1 pass of refinement (conservative — H3 showed divergence with more)
    print("Applying 1 refinement pass (conservative)...")
    z_refined = apply_single_refinement_pass(elf_model, z_original, noise_scale)

    # Score AFTER refinement
    print("Scoring after refinement...")
    after_tokens, after_entropy = decode_batch(elf_model, z_refined, DEVICE)
    after_ppls = []
    for i in range(N_SAMPLES):
        text = decode_tokens_to_text(after_tokens[i], detok_map)
        ppl = score_text(text, lm_model, lm_tokenizer, DEVICE)
        if not math.isinf(ppl) and not math.isnan(ppl):
            after_ppls.append(ppl)
        else:
            after_ppls.append(50000.0)

    # Paired bootstrap
    diff_mean, diff_lo, diff_hi = bootstrap_paired_ci(before_ppls, after_ppls)
    excludes_zero = (diff_lo > 0) or (diff_hi < 0)

    print(f"\n  Before PPL: {np.mean(before_ppls):.1f} ± {np.std(before_ppls):.1f}")
    print(f"  After PPL:  {np.mean(after_ppls):.1f} ± {np.std(after_ppls):.1f}")
    print(f"  Paired Δ:   {diff_mean:+.1f} [{diff_lo:+.1f}, {diff_hi:+.1f}]")
    print(f"  Excludes 0: {excludes_zero}")

    # How many sequences improved/worsened/unchanged?
    improved = sum(1 for b, a in zip(before_ppls, after_ppls) if a < b * 0.95)
    worsened = sum(1 for b, a in zip(before_ppls, after_ppls) if a > b * 1.05)
    unchanged = N_SAMPLES - improved - worsened
    print(f"\n  Improved:   {improved}/{N_SAMPLES} ({100*improved/N_SAMPLES:.1f}%)")
    print(f"  Worsened:   {worsened}/{N_SAMPLES} ({100*worsened/N_SAMPLES:.1f}%)")
    print(f"  Unchanged:  {unchanged}/{N_SAMPLES} ({100*unchanged/N_SAMPLES:.1f}%)")

    print(f"\n{'=' * 60}")
    if excludes_zero and diff_mean > 0:
        print("✅ H5: Refinement produces a statistically significant PPL improvement.")
        verdict = "SIGNIFICANT_IMPROVEMENT"
    elif excludes_zero and diff_mean < 0:
        print("❌ H5: Refinement significantly WORSENS PPL.")
        verdict = "SIGNIFICANT_WORSENING"
    else:
        print("🟡 H5: No statistically significant effect detected.")
        verdict = "NO_SIGNIFICANT_EFFECT"
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Paired scatter
    ax = axes[0]
    ax.scatter(before_ppls, after_ppls, alpha=0.3, s=15, color='#4C72B0')
    lims = [0, min(max(max(before_ppls), max(after_ppls)), 30000)]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
    ax.set_xlabel('Before PPL'); ax.set_ylabel('After PPL')
    ax.set_title('Paired Before/After PPL', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Bootstrap distribution
    ax = axes[1]
    rng = np.random.default_rng(42)
    min_len = min(len(before_ppls), len(after_ppls))
    diffs = np.array(before_ppls[:min_len]) - np.array(after_ppls[:min_len])
    boot_diffs = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    ax.hist(boot_diffs, bins=50, alpha=0.7, color='#55A868')
    ax.axvline(0, color='red', ls='--', lw=2, label='Zero (no effect)')
    ax.axvline(diff_lo, color='orange', ls=':', lw=1.5, label=f'95% CI [{diff_lo:+.0f}, {diff_hi:+.0f}]')
    ax.axvline(diff_hi, color='orange', ls=':', lw=1.5)
    ax.set_xlabel('PPL Improvement (Before - After)')
    ax.set_title('Bootstrap Distribution of Effect', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle("H5: End-to-End Refinement Net Effect", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "testH5_net_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {RESULTS_DIR / 'testH5_net_effect.png'}")

    with open(RESULTS_DIR / "testH5.json", "w") as f:
        json.dump({
            "before_ppl_mean": float(np.mean(before_ppls)),
            "after_ppl_mean": float(np.mean(after_ppls)),
            "diff_mean": diff_mean, "diff_ci_lo": diff_lo, "diff_ci_hi": diff_hi,
            "excludes_zero": excludes_zero,
            "improved": improved, "worsened": worsened, "unchanged": unchanged,
            "verdict": verdict,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
