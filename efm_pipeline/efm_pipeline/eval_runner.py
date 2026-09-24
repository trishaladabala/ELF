#!/usr/bin/env python3
"""Standardised evaluation runner for all experiments.

Provides a single, consistent evaluation protocol:
    1. Load model checkpoint
    2. Generate N samples with fixed seed
    3. Decode to text using ELF's decoder (reused)
    4. Compute Gen-PPL via ELF's Metrics class (GPT-2)
    5. Compute unigram entropy
    6. Compute decoder confidence
    7. Save results to JSON + CSV

All experiments use this runner with identical settings so results
are directly comparable.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ELF imports.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ELF_SRC = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_sampling_steps, _ode_step
from utils.generation_utils import mask_after_eos

from efm_pipeline.utils import set_seed, get_device, ensure_dir, Timer
from efm_pipeline.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_unigram_entropy(texts: List[str]) -> float:
    """Compute unigram (word-level) entropy in bits."""
    all_words = " ".join(texts).split()
    if not all_words:
        return 0.0
    counts = Counter(all_words)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return entropy


def compute_decoder_confidence(
    model, embeddings: torch.Tensor, device: torch.device
) -> float:
    """Compute average decoder confidence (max softmax prob).

    Args:
        model: Model with decoder weights (proj_kernel, unembed_kernel).
        embeddings: (B, S, text_encoder_dim) — final embeddings.

    Returns:
        Average max probability across all tokens.
    """
    with torch.no_grad():
        # Run through bottleneck → blocks is already done; we need the
        # hidden states. For simplicity, use the model's forward pass.
        # Here we just use the decoder unembedding directly on the embeddings.
        # This assumes embeddings are in hidden_size space after the transformer.
        if hasattr(model, 'proj_kernel'):
            x_f32 = embeddings.float().to(device)
            hidden = F.gelu(
                x_f32 @ model.proj_kernel.to(device) + model.proj_bias.to(device),
                approximate="tanh",
            )
            logits = hidden @ model.unembed_kernel.to(device) + model.unembed_bias.to(device)
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()
            return confidence
    return 0.0


def compute_ngram_repetition(texts: List[str], n: int = 3) -> float:
    """Compute fraction of n-grams that are repeated within each text."""
    if not texts:
        return 0.0
    rep_rates = []
    for text in texts:
        words = text.split()
        if len(words) < n:
            rep_rates.append(0.0)
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique = len(set(ngrams))
        total = len(ngrams)
        rep_rate = 1.0 - (unique / total) if total > 0 else 0.0
        rep_rates.append(rep_rate)
    return float(np.mean(rep_rates))


# ---------------------------------------------------------------------------
# EFM Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_efm_samples(
    model,
    tokenizer,
    device: torch.device,
    num_samples: int = 512,
    num_steps: int = 32,
    batch_size: int = 8,
    max_length: int = 64,
    seed: int = 42,
    noise_scale: float = 1.0,
    local_time_mode: str = "expansion",
    expansion_fraction: float = 0.5,
) -> List[str]:
    """Generate unconditional samples from an EFM model.

    Returns:
        List of generated text strings.
    """
    model.eval()
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    text_encoder_dim = model.text_encoder_dim
    all_texts = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Number of original vs inserted tokens for expansion mode
    n_inserted = max(1, int(max_length * expansion_fraction))

    for batch_idx in range(num_batches):
        curr_bs = min(batch_size, num_samples - len(all_texts))
        if curr_bs <= 0:
            break

        # Initial noise.
        z = torch.randn(
            curr_bs, max_length, text_encoder_dim,
            generator=generator, dtype=torch.float32, device=gen_device,
        ).to(device) * noise_scale

        # For expansion mode, pick fixed insertion times and positions per sample
        if local_time_mode == "expansion":
            insertion_times = torch.zeros(curr_bs, max_length, device=device)
            # Original tokens (inserted at 0)
            # Inserted tokens get random insertion times ~ U(0, 1)
            for b in range(curr_bs):
                perm = torch.randperm(max_length, generator=generator, device=gen_device).to(device)
                inserted_mask = torch.zeros(max_length, dtype=torch.bool, device=device)
                inserted_mask[perm[:n_inserted]] = True
                # Sample insertion times for the inserted tokens
                t_ins = torch.rand(n_inserted, generator=generator, device=gen_device).to(device)
                insertion_times[b, inserted_mask] = t_ins

        # Time schedule.
        t_steps = get_sampling_steps(num_steps, time_schedule="uniform", device=device)

        # ODE loop.
        for i in range(len(t_steps) - 1):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            dt = t_next - t

            # Compute local times for current step
            if local_time_mode == "expansion":
                # τ_i = (t - t_ins_i) / (1 - t_ins_i)
                # But only if t > t_ins_i, else 0
                local_times = torch.zeros(curr_bs, max_length, device=device)
                active_mask = (t > insertion_times)
                denom = (1.0 - insertion_times).clamp(min=1e-6)
                local_times[active_mask] = (t - insertion_times[active_mask]) / denom[active_mask]
                local_times = local_times.clamp(0.0, 1.0)
            elif local_time_mode.startswith("constant_"):
                val = float(local_time_mode.split("_")[1])
                local_times = torch.full((curr_bs, max_length), val, device=device)
            elif local_time_mode == "none":
                local_times = None
            else:  # broadcast (default/fallback)
                local_times = torch.full((curr_bs, max_length), t, device=device)

            # Forward pass.
            t_batch = torch.full((curr_bs,), t, dtype=z.dtype, device=device)
            flow_out, _ = model(z, t_batch, local_times=local_times, decoder_step_active=False)

            # ODE step computation.
            t_clamped = max(t, 1e-3)

            # For per-token times, the velocity is per-token: v_i = (flow_out_i - z_i) / (1 - τ_i)
            # Actually, the standard flow-matching ODE step applies to the global t:
            # dz/dt = (x0 - z) / (1 - t).
            # Even if local times differ, the trajectory z(t) is defined w.r.t global t,
            # so the ODE integration step itself uses global t.
            v = (flow_out - z) / max(1.0 - t_clamped, 5e-2)
            z = z + dt * v

        # Final decode step.
        t_final = torch.ones(curr_bs, dtype=z.dtype, device=device)
        
        # Recompute local_times for t=1.0 exactly
        if local_time_mode == "expansion":
            final_local_times = torch.ones(curr_bs, max_length, device=device)
        elif local_time_mode.startswith("constant_"):
            val = float(local_time_mode.split("_")[1])
            final_local_times = torch.full((curr_bs, max_length), val, device=device)
        elif local_time_mode == "none":
            final_local_times = None
        else:
            final_local_times = torch.ones(curr_bs, max_length, device=device)

        _, decoder_logits = model(z, t_final, local_times=final_local_times, decoder_step_active=True)
        if decoder_logits is not None:
            predicted_ids = decoder_logits.argmax(dim=-1)
            eos_id = tokenizer.eos_token_id or 1
            pad_id = tokenizer.pad_token_id or 0
            predicted_ids = mask_after_eos(predicted_ids, eos_id, pad_id)
            texts = tokenizer.batch_decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
            all_texts.extend(texts)
        else:
            # Fallback: fill with empty strings.
            all_texts.extend([""] * curr_bs)

        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            print(f"  Generated {len(all_texts)}/{num_samples} samples")

    return all_texts[:num_samples]


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    texts: List[str],
    experiment_name: str,
    output_dir: str = "results",
    gen_ppl_model: str = "gpt2-large",
    eval_batch_size: int = 8,
    max_length: int = 1024,
) -> Dict:
    """Run the full evaluation pipeline on generated texts.

    Returns a dict of all metrics.
    """
    logger = ExperimentLogger(output_dir, experiment_name)
    timer = Timer().start()

    results = {
        "num_samples": len(texts),
        "avg_length_words": np.mean([len(t.split()) for t in texts]) if texts else 0,
    }

    # 1. Gen-PPL using ELF's Metrics class.
    print(f"[{experiment_name}] Computing Gen-PPL with {gen_ppl_model}...")
    metrics = PPLMetrics(
        gen_ppl_eval_model_name_or_path=gen_ppl_model,
        eval_ppl_batch_size=eval_batch_size,
    )
    ppl_result = metrics.record_generative_perplexity(
        text_samples=texts, max_length=max_length, retokenize=True,
    )
    results["gen_ppl"] = ppl_result["ppl"]
    results["mean_entropy_ppl"] = ppl_result["mean_entropy"]

    # 2. Unigram entropy.
    results["unigram_entropy_bits"] = compute_unigram_entropy(texts)

    # 3. N-gram repetition.
    results["trigram_repetition"] = compute_ngram_repetition(texts, n=3)

    # 4. Sample statistics.
    word_counts = [len(t.split()) for t in texts]
    results["avg_words"] = float(np.mean(word_counts)) if word_counts else 0
    results["std_words"] = float(np.std(word_counts)) if word_counts else 0
    empty_count = sum(1 for t in texts if len(t.strip()) == 0)
    results["empty_fraction"] = empty_count / max(len(texts), 1)

    elapsed = timer.stop()
    results["eval_time_seconds"] = round(elapsed, 1)

    # Save.
    logger.save_result(results)

    # Print summary.
    print(f"\n{'='*50}")
    print(f"  Evaluation: {experiment_name}")
    print(f"  Samples:         {results['num_samples']}")
    print(f"  Gen-PPL:         {results['gen_ppl']:.2f}")
    print(f"  Unigram Entropy: {results['unigram_entropy_bits']:.4f} bits")
    print(f"  3-gram Rep Rate: {results['trigram_repetition']:.4f}")
    print(f"  Avg Words:       {results['avg_words']:.1f} ± {results['std_words']:.1f}")
    print(f"  Empty Fraction:  {results['empty_fraction']:.4f}")
    print(f"  Time:            {elapsed:.0f}s")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated text samples")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to JSON file containing list of generated texts.")
    parser.add_argument("--experiment_name", type=str, default="eval")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--gen_ppl_model", type=str, default="gpt2-large")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    # Note: the generation script (like stage2) should handle generation logic
    # and save to JSON. This script only evaluates text.
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    texts = data if isinstance(data, list) else data.get("texts", [])

    evaluate(
        texts=texts,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        gen_ppl_model=args.gen_ppl_model,
        eval_batch_size=args.eval_batch_size,
    )
