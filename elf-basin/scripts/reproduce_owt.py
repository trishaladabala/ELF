#!/usr/bin/env python3
"""Phase 0 — Reproduce ELF-B-owt published numbers.

Target: 32-step SDE, γ=1.5, self-conditioning CFG=3, logit-normal time schedule
        (P_mean=−1.5, P_std=0.8), 1000 samples.
        Gen. PPL ≈ 24–26 under GPT-2-Large, entropy ≈ 5.15–5.20.

Reports: PPL (geometric), unigram entropy, distinct-1, distinct-2,
         repeated-4-gram fraction, wall-clock, batch size, peak VRAM.
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper, SimpleConfig
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch
from utils.metrics_utils import Metrics as PPLMetrics
from configs.config import SamplingConfig


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-owt"
NUM_SAMPLES = 1000
BATCH_SIZE = 8           # Per-device; fits in ~6 GB at L=1024
NUM_STEPS = 32
SDE_GAMMA = 1.5
SC_CFG = 3.0             # Self-conditioning CFG scale
CFG = 1.0                # Input-conditioning CFG scale (unconditional = 1)
TIME_SCHEDULE = "logit_normal"
SEED = 42
PPL_MODEL = "gpt2-large"
PPL_BATCH_SIZE = 16
RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "phase0_owt"


def compute_diversity_metrics(texts: list) -> dict:
    """Compute unigram entropy, distinct-1/2, repeated-4-gram fraction."""
    all_words = []
    all_bigrams = []
    total_4gram_count = 0
    repeated_4gram_count = 0

    for text in texts:
        words = text.split()
        all_words.extend(words)
        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i+1]))

        # Repeated 4-grams per sample
        fourgrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        counts = Counter(fourgrams)
        total_4gram_count += len(fourgrams)
        repeated_4gram_count += sum(c - 1 for c in counts.values() if c > 1)

    # Unigram entropy
    word_counts = Counter(all_words)
    total = sum(word_counts.values())
    probs = np.array([c / total for c in word_counts.values()])
    unigram_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # Distinct-1 and distinct-2
    distinct_1 = len(set(all_words)) / max(len(all_words), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    # Repeated 4-gram fraction
    rep_4gram_frac = repeated_4gram_count / max(total_4gram_count, 1)

    return {
        "unigram_entropy": unigram_entropy,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repeated_4gram_fraction": rep_4gram_frac,
        "total_words": total,
        "unique_words": len(set(all_words)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Load model ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  Phase 0: OWT Reproduction — Loading ELF-B-owt")
    print("="*70)

    wrapper = ELFWrapper(CHECKPOINT, device=device)
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    config = wrapper.config

    # Build a SamplingConfig for the existing generation pipeline
    sampling_config = SamplingConfig(
        sampling_method="sde",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        sde_gamma=SDE_GAMMA,
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    # ── Generate samples ───────────────────────────────────────────────
    print(f"\nGenerating {NUM_SAMPLES} samples...")
    print(f"  Steps: {NUM_STEPS}, SDE γ: {SDE_GAMMA}, SC-CFG: {SC_CFG}")
    print(f"  Batch size: {BATCH_SIZE}, Time schedule: {TIME_SCHEDULE}")

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
    else:
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    d_model = wrapper.d_model
    max_length = wrapper.max_length
    param_dtype = next(model.parameters()).dtype

    all_texts = []
    all_token_ids = []
    gen_start = time.time()
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for batch_idx in range(num_batches):
        current_batch = min(BATCH_SIZE, NUM_SAMPLES - len(all_texts))
        if current_batch <= 0:
            break

        # Sample time steps
        t_steps = get_sampling_steps(
            n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=param_dtype,
        )

        # Sample initial noise
        z = torch.randn(
            (current_batch, max_length, d_model),
            dtype=param_dtype, device=device,
        ) * config.denoiser_noise_scale

        # Generate
        latent = _generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=None, cond_seq_mask=None,
            config=config, sampling_config=sampling_config,
            cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
        )

        # Decode to tokens
        t_final = t_steps[-1].item()
        predicted_ids = _dlm_decode_batch(
            z=latent, model=model, t_final_val=t_final,
            config=config, self_cond_cfg_scale=SC_CFG,
        )

        # Mask after EOS
        eos_mask = (predicted_ids == eos_id)
        keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
        predicted_ids = torch.where(keep_mask, predicted_ids,
                                     torch.full_like(predicted_ids, pad_id))

        texts = tokenizer.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)
        all_texts.extend(texts)
        all_token_ids.extend(predicted_ids.cpu().tolist())

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            elapsed = time.time() - gen_start
            print(f"  Batch {batch_idx+1}/{num_batches} — "
                  f"{len(all_texts)}/{NUM_SAMPLES} samples — "
                  f"{elapsed:.1f}s elapsed")

    gen_time = time.time() - gen_start
    all_texts = all_texts[:NUM_SAMPLES]
    print(f"\nGeneration complete: {len(all_texts)} samples in {gen_time:.1f}s")

    # ── Score PPL ──────────────────────────────────────────────────────
    print(f"\nScoring with {PPL_MODEL}...")
    ppl_metrics = PPLMetrics(
        gen_ppl_eval_model_name_or_path=PPL_MODEL,
        eval_ppl_batch_size=PPL_BATCH_SIZE,
        eval_context_size=1024,
    )

    ppl_start = time.time()
    ppl_result = ppl_metrics.record_generative_perplexity(
        all_texts, max_length=1024, retokenize=True,
    )
    ppl_time = time.time() - ppl_start

    # ── Diversity metrics ──────────────────────────────────────────────
    div_metrics = compute_diversity_metrics(all_texts)

    # ── Peak VRAM ──────────────────────────────────────────────────────
    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # ── Config hash ────────────────────────────────────────────────────
    config_str = json.dumps({
        "checkpoint": CHECKPOINT, "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE, "num_steps": NUM_STEPS,
        "sde_gamma": SDE_GAMMA, "sc_cfg": SC_CFG, "cfg": CFG,
        "time_schedule": TIME_SCHEDULE, "seed": SEED,
        "ppl_model": PPL_MODEL,
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 0 — OWT REPRODUCTION RESULTS")
    print("="*70)
    print(f"  Checkpoint:       {CHECKPOINT} (step {wrapper.checkpoint_step})")
    print(f"  Config hash:      {config_hash}")
    print(f"  Seed:             {SEED}")
    print(f"  Samples:          {len(all_texts)}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Steps:            {NUM_STEPS}, SDE γ={SDE_GAMMA}, SC-CFG={SC_CFG}")
    print(f"  Gen. time:        {gen_time:.1f}s ({gen_time/60:.1f}m)")
    print(f"  PPL scoring time: {ppl_time:.1f}s ({ppl_time/60:.1f}m)")
    print(f"  Peak VRAM:        {peak_vram_mb:.0f} MB")
    print()
    print(f"  ┌─── Target ─────────────────────────────┐")
    print(f"  │ Gen. PPL:  24–26  (unofficial: 25.61)  │")
    print(f"  │ Entropy:   ~5.15–5.20                  │")
    print(f"  └────────────────────────────────────────┘")
    print()
    print(f"  ┌─── Measured ───────────────────────────┐")
    print(f"  │ Gen. PPL (geometric): {ppl_result['ppl']:.2f}")
    print(f"  │ Mean entropy:         {ppl_result['mean_entropy']:.3f}")
    print(f"  │ Unigram entropy:      {div_metrics['unigram_entropy']:.3f}")
    print(f"  │ Distinct-1:           {div_metrics['distinct_1']:.4f}")
    print(f"  │ Distinct-2:           {div_metrics['distinct_2']:.4f}")
    print(f"  │ Rep. 4-gram frac:     {div_metrics['repeated_4gram_fraction']:.4f}")
    print(f"  └────────────────────────────────────────┘")

    # Gate check
    target_ppl_low, target_ppl_high = 24 * 0.9, 26 * 1.1  # 10% tolerance
    ppl_ok = target_ppl_low <= ppl_result['ppl'] <= target_ppl_high
    print()
    if ppl_ok:
        print("  ✓ GATE 0 (OWT): PPL is within 10% of target range [24, 26]")
    else:
        print(f"  ✗ GATE 0 (OWT): PPL {ppl_result['ppl']:.2f} is OUTSIDE "
              f"10% tolerance of [24, 26]. STOP and diagnose.")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": wrapper.checkpoint_step,
        "config_hash": config_hash,
        "seed": SEED,
        "num_samples": len(all_texts),
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "sde_gamma": SDE_GAMMA,
        "sc_cfg": SC_CFG,
        "cfg": CFG,
        "time_schedule": TIME_SCHEDULE,
        "gen_ppl": ppl_result["ppl"],
        "mean_entropy": ppl_result["mean_entropy"],
        **div_metrics,
        "gen_time_s": gen_time,
        "ppl_time_s": ppl_time,
        "peak_vram_mb": peak_vram_mb,
        "gate_0_owt_pass": ppl_ok,
    }

    results_file = RESULTS_DIR / "owt_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Save a few sample texts
    samples_file = RESULTS_DIR / "owt_samples.txt"
    with open(samples_file, "w") as f:
        for i, text in enumerate(all_texts[:20]):
            f.write(f"--- Sample {i} ---\n{text}\n\n")
    print(f"  Sample texts saved to {samples_file}")

    return 0 if ppl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
