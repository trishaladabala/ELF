#!/usr/bin/env python3
"""W=8 Learnability Gate: cheap 3-condition test before full sweep.

Tests whether W=8 is a viable difficulty level for EFM_Small at 50k steps
by running exactly three conditions:
  1. global_time_only  (no local time — lower bound)
  2. continuous         (full local time — upper capacity bound)
  3. lowrank_K2         (intermediate — the W=4 winner)

Uses identical training/evaluation infrastructure as the W=4/W=16 baselines
in experiments/baseline_matched_compute/run.py.

Before training, performs a dataset sanity check: verifies W=8 wave structure,
key-bank validity, and prints example samples.

After training, classifies results as:
  TRAINING_NOT_CONVERGED         — TF accuracy < 90%
  TRAINING_CONVERGED_GENERATION_FAILS — TF acc ≥ 90%, int dep acc ≤ random_threshold
  NONTRIVIAL_GENERATION          — TF acc ≥ 90%, int dep acc > random_threshold

Thresholds:
  - TF accuracy convergence threshold: 90% (the W=4 baseline reaches 99.6%)
  - Random baseline for internal dep acc: 1/vocab_size = 1/256 ≈ 0.39%
  - Nontrivial threshold: 5× random = 1.95% (any signal clearly above noise)
    Rationale: W=4 baseline global_time_only got 0.5% (≈random), while
    lowrank_K2 got 69.8%. Even the weakest non-random result should
    exceed 5× chance. W=16 baseline best was 2.2% (lowrank_K2), which
    would barely clear this threshold — borderline, as expected.

Usage (from efm_pipeline/):
    python -m experiments.run_w8_gate
"""
import os
import sys
import json
import argparse
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation


# ── Reference values from existing experiments ────────────────────────────

VOCAB_SIZE = 256
RANDOM_BASELINE = 1.0 / VOCAB_SIZE          # 0.00390625
NONTRIVIAL_THRESHOLD = 5.0 * RANDOM_BASELINE  # 0.01953125 (~1.95%)
TF_CONVERGENCE_THRESHOLD = 0.90

# W=4 baseline (50k steps) reference:
#   lowrank_K2 internal dep acc = 0.698
#   continuous internal dep acc = 0.511
#   global internal dep acc     = 0.005  (≈ random)

# W=16 baseline (50k steps) reference:
#   ALL conditions ≤ 0.022  (essentially random)


# ── Dataset sanity check ──────────────────────────────────────────────────

def validate_dataset(dataset, W, num_examples=3):
    """Verify W=8 dataset structure is correct. Print examples."""
    print(f"\n{'='*60}")
    print(f"  DATASET SANITY CHECK (W={W})")
    print(f"{'='*60}")

    seq_len = dataset.token_ids.shape[1]
    M = seq_len // W
    num_keys = (W - 1) * M

    print(f"  seq_len     = {seq_len}")
    print(f"  W           = {W}")
    print(f"  M (per wave)= {M}")
    print(f"  num_keys    = {num_keys}")
    print(f"  num_samples = {len(dataset)}")
    print(f"  vocab_size  = {dataset.vocab_size}")
    print(f"  wave_ids shape = {dataset.wave_ids.shape}")

    # Verify wave structure
    unique_waves = sorted(dataset.wave_ids.unique().tolist())
    sample0_waves = sorted(dataset.wave_ids[0].unique().tolist())
    print(f"  Unique waves in dataset: {unique_waves}")
    print(f"  Unique waves in sample 0: {sample0_waves}")
    assert len(unique_waves) == W, f"Expected {W} unique waves across dataset, got {len(unique_waves)}"

    # Count tokens per wave in sample 0
    wave_counts = {}
    for w in range(W):
        wave_counts[w] = (dataset.wave_ids[0] == w).sum().item()
    print(f"  Tokens per wave in sample 0: {wave_counts}")

    # Verify key-bank structure
    token_ids = dataset.token_ids[0]
    keys_section = token_ids[:num_keys]
    vals_section = token_ids[num_keys:]

    print(f"\n  Key section: positions 0..{num_keys-1} ({num_keys} tokens)")
    print(f"  Val section: positions {num_keys}..{seq_len-1} ({seq_len - num_keys} tokens)")

    # Check that values copy from correct keys in sample 0
    errors = 0
    val_waves = dataset.wave_ids[0, num_keys:]
    for i in range(M):
        w = val_waves[i].item()
        expected_token = token_ids[(w-1)*M + i].item()
        if vals_section[i].item() != expected_token:
            errors += 1
            print(f"  ✗ Mismatch at value index {i}: wave {w}, expected {expected_token}, got {vals_section[i].item()}")
            
    if errors > 0:
        print(f"\n  ✗ DATASET VALIDATION FAILED: {errors} total mismatches")
        sys.exit(1)
    else:
        print(f"  ✓ All {M} values in sample 0 correctly copy from their assigned key banks")

    # Print example samples
    print(f"\n  --- Example Samples ---")
    for i in range(min(num_examples, len(dataset))):
        tids = dataset.token_ids[i]
        wids = dataset.wave_ids[i]
        print(f"  Sample {i}:")
        print(f"    token_ids[:12] = {tids[:12].tolist()}")
        print(f"    wave_ids [:12] = {wids[:12].tolist()}")
        print(f"    token_ids[-8:] = {tids[-8:].tolist()}")
        print(f"    wave_ids [-8:] = {wids[-8:].tolist()}")

    print(f"\n  ✓ Dataset validation passed for W={W}")
    print(f"{'='*60}\n")


# ── Teacher-forced evaluation (identical to baseline_matched_compute) ─────

def evaluate_tf(model, dataset, device):
    """Teacher-forced evaluation at t=1."""
    model.eval()
    bs = min(1000, len(dataset.embeddings))
    x0 = dataset.embeddings[:bs].to(device)
    times = dataset.insertion_times[:bs].to(device)
    targets = dataset.token_ids[:bs].to(device)

    with torch.no_grad():
        t = torch.ones((bs,), device=device)
        _, logits = model(x0, t, local_times=times, decoder_step_active=True)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

    return loss.item(), acc


# ── Cheap output diversity diagnostic ─────────────────────────────────────

def output_diversity(generated_token_ids):
    """Minimal diversity diagnostic: unique tokens and per-sample entropy."""
    B, L = generated_token_ids.shape
    total_tokens = B * L

    # Global unique tokens
    flat = generated_token_ids.view(-1).tolist()
    unique_global = len(set(flat))

    # Per-sample unique tokens (mean)
    per_sample_unique = []
    for i in range(B):
        per_sample_unique.append(len(set(generated_token_ids[i].tolist())))
    mean_unique_per_sample = sum(per_sample_unique) / len(per_sample_unique)

    # Per-sample entropy
    import math
    entropies = []
    for i in range(B):
        counts = Counter(generated_token_ids[i].tolist())
        total = sum(counts.values())
        ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
        entropies.append(ent)
    mean_entropy = sum(entropies) / len(entropies)

    # Max possible entropy for seq_len tokens with vocab_size
    max_entropy = math.log2(min(L, VOCAB_SIZE))

    return {
        "unique_tokens_global": unique_global,
        "unique_tokens_per_sample_mean": round(mean_unique_per_sample, 2),
        "token_entropy_per_sample_mean": round(mean_entropy, 4),
        "max_possible_entropy": round(max_entropy, 4),
        "entropy_ratio": round(mean_entropy / max_entropy, 4) if max_entropy > 0 else 0.0,
    }


# ── Learnability classification ──────────────────────────────────────────

def classify_condition(tf_acc, internal_dep_acc):
    """Classify a single condition's learnability."""
    if tf_acc < TF_CONVERGENCE_THRESHOLD:
        return "TRAINING_NOT_CONVERGED"
    if internal_dep_acc <= NONTRIVIAL_THRESHOLD:
        return "TRAINING_CONVERGED_GENERATION_FAILS"
    return "NONTRIVIAL_GENERATION"


def classify_gate(results):
    """Overall gate classification from all conditions."""
    classifications = [r["classification"] for r in results.values()]
    if "NONTRIVIAL_GENERATION" in classifications:
        return "PASS"
    if "TRAINING_CONVERGED_GENERATION_FAILS" in classifications:
        return "FAIL_GENERATION"
    return "FAIL_TRAINING"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="W=8 learnability gate: 3-condition test before full sweep")
    parser.add_argument("--W", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gen-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="w8_gate_results")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_w8_gate")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)

    W = args.W

    # ── Gate conditions (exactly 3) ───────────────────────────────────────
    conditions = [
        ("global_time_only",  "none",      0, "none"),
        ("continuous",        "continuous", 0, "expansion"),
        ("lowrank_K2",        "lowrank",    2, "expansion"),
    ]

    # ── Dataset generation & validation ───────────────────────────────────
    dataset = OrderedAssemblyDataset(
        num_samples=5000, seq_len=64, vocab_size=VOCAB_SIZE,
        embed_dim=512, W=W, seed=args.seed,
    )

    validate_dataset(dataset, W)

    # ── Training & evaluation loop ────────────────────────────────────────
    all_results = {}

    for name, mode, K, lt_training in conditions:
        exp_name = f"w8_gate_{name}_s{args.seed}"
        metrics_file = os.path.join(args.output_dir, f"{exp_name}_metrics.json")
        ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

        if args.skip_completed and os.path.exists(metrics_file):
            print(f"\n  Skipping {name} — already completed")
            with open(metrics_file) as f:
                all_results[name] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"  Mode: {mode}, K: {K}, Steps: {args.steps}")
        print(f"{'='*60}")

        # Build model (identical to baseline)
        model = EFM_Small(
            vocab_size=VOCAB_SIZE,
            local_time_mode=mode,
            local_time_K=K,
            lowrank_basis="learned" if mode == "lowrank" else "fourier",
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Train (identical call to baseline_matched_compute/run.py)
        train(
            model=model, dataset=dataset, device=device,
            max_steps=args.steps, batch_size=args.batch_size,
            save_every=10000, log_every=500,
            checkpoint_dir=ckpt_dir, experiment_name=exp_name,
            output_dir=args.output_dir,
            local_time_training=lt_training,
        )

        # Load final checkpoint to extract loss breakdown
        final_ckpt_path = os.path.join(ckpt_dir, exp_name, f"checkpoint_{args.steps}.pt")
        final_loss_breakdown = {}
        if os.path.exists(final_ckpt_path):
            ckpt = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
            final_loss_breakdown["final_loss_from_ckpt"] = ckpt.get("loss", None)
            final_loss_breakdown["final_step"] = ckpt.get("step", None)

        # Read training result JSON for final_loss
        train_result_path = os.path.join(args.output_dir, f"{exp_name}_result.json")
        train_result = {}
        if os.path.exists(train_result_path):
            with open(train_result_path) as f:
                train_result = json.load(f)

        # Teacher-forced evaluation (identical to baseline)
        tf_nll, tf_acc = evaluate_tf(model, dataset, device)

        # Generation evaluation (identical to baseline)
        print(f"\n  Generating {args.num_gen_samples} samples...")
        generated_ids = generate_synthetic_samples(
            model=model, dataset=dataset, device=device,
            num_samples=args.num_gen_samples, num_steps=64,
            batch_size=64, seed=args.seed,
            local_time_mode=mode if mode != "none" else "none",
            W=W,
        )

        # Full evaluation
        metrics = full_evaluation(
            generated_token_ids=generated_ids[:args.num_gen_samples].cpu(),
            true_token_ids=dataset.token_ids[:args.num_gen_samples].cpu(),
            wave_ids=dataset.wave_ids[:args.num_gen_samples].cpu(),
            vocab_size=VOCAB_SIZE,
        )

        # Output diversity diagnostic (cheap)
        diversity = output_diversity(generated_ids[:args.num_gen_samples].cpu())

        # Classification
        classification = classify_condition(tf_acc, metrics["overall_internal_dep_acc"])

        # Assemble result
        result = {
            "experiment": exp_name,
            "W": W,
            "mode": mode,
            "K": K,
            "steps": args.steps,
            "seed": args.seed,
            "num_params": num_params,
            "timestamp": datetime.now().isoformat(),
            # Training metrics
            "final_loss": train_result.get("final_loss", final_loss_breakdown.get("final_loss_from_ckpt")),
            "elapsed_seconds": train_result.get("elapsed_seconds", None),
            # Teacher-forced metrics
            "tf_nll": tf_nll,
            "tf_acc": tf_acc,
            # Generation metrics
            **metrics,
            # Diversity
            "diversity": diversity,
            # Classification
            "classification": classification,
        }
        all_results[name] = result

        # Save per-condition result
        with open(metrics_file, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n  Results for {name}:")
        print(f"    TF NLL:           {tf_nll:.4f}")
        print(f"    TF Accuracy:      {tf_acc:.4f}")
        print(f"    Ext Dep Acc:      {metrics['overall_dep_acc']:.4f}")
        print(f"    Int Dep Acc:      {metrics['overall_internal_dep_acc']:.4f}")
        print(f"    Exact Match:      {metrics['overall_exact_match']:.4f}")
        print(f"    Unique tokens:    {diversity['unique_tokens_global']}/{VOCAB_SIZE}")
        print(f"    Entropy ratio:    {diversity['entropy_ratio']:.4f}")
        print(f"    Classification:   {classification}")

    # ── Summary ───────────────────────────────────────────────────────────
    gate_verdict = classify_gate(all_results)

    print(f"\n{'='*90}")
    print(f"  W=8 LEARNABILITY GATE — RESULTS")
    print(f"{'='*90}")
    print(f"{'Condition':<22s} {'TF Acc':>8s} {'Int Dep':>10s} {'Ext Dep':>10s} {'Exact':>8s} {'Entropy':>8s} {'Classification'}")
    print("-" * 90)
    for name, r in all_results.items():
        print(f"{name:<22s} {r['tf_acc']:>8.4f} "
              f"{r['overall_internal_dep_acc']:>10.4f} "
              f"{r['overall_dep_acc']:>10.4f} "
              f"{r['overall_exact_match']:>8.4f} "
              f"{r['diversity']['entropy_ratio']:>8.4f} "
              f"{r['classification']}")
    print("-" * 90)
    print(f"\n  Thresholds:")
    print(f"    TF convergence:        ≥ {TF_CONVERGENCE_THRESHOLD:.0%}")
    print(f"    Random baseline:       {RANDOM_BASELINE:.4f} (1/{VOCAB_SIZE})")
    print(f"    Nontrivial threshold:  {NONTRIVIAL_THRESHOLD:.4f} (5× random)")
    print(f"\n  Reference (W=4 baseline, 50k steps):")
    print(f"    global:     TF=99.6%, int_dep=0.5%  (≈ random)")
    print(f"    continuous: TF=99.6%, int_dep=51.1% (nontrivial)")
    print(f"    lowrank_K2: TF=99.6%, int_dep=69.8% (nontrivial)")
    print(f"\n  Reference (W=16 baseline, 50k steps):")
    print(f"    best:       TF=99.6%, int_dep=2.2%  (borderline)")
    print(f"\n  ╔{'═'*40}╗")
    print(f"  ║  GATE VERDICT: {gate_verdict:<22s} ║")
    print(f"  ╚{'═'*40}╝")

    if gate_verdict == "PASS":
        print(f"\n  W=8 shows nontrivial generation. Proceed to full sweep.")
        print(f"  Recommended command:")
        print(f"    python -m experiments.run_wk_sweep --W 8 --conditions global_time_only continuous lowrank_K1 lowrank_K2 lowrank_K4 lowrank_K8 lowrank_K16 --skip-completed")
    elif gate_verdict == "FAIL_GENERATION":
        print(f"\n  W=8 training converges but generation fails at 50k steps.")
        print(f"  Consider:")
        print(f"    A) Increasing training steps: python -m experiments.run_w8_gate --steps 100000")
        print(f"    B) Trying W=6 as intermediate difficulty")
    else:
        print(f"\n  W=8 training does not converge at 50k steps.")
        print(f"  This W value is too difficult for EFM_Small at this budget.")

    print(f"{'='*90}")

    # Save gate summary
    summary = {
        "gate_verdict": gate_verdict,
        "W": W,
        "steps": args.steps,
        "seed": args.seed,
        "thresholds": {
            "tf_convergence": TF_CONVERGENCE_THRESHOLD,
            "random_baseline": RANDOM_BASELINE,
            "nontrivial": NONTRIVIAL_THRESHOLD,
        },
        "reference": {
            "W4_baseline": {
                "global_int_dep_acc": 0.005,
                "continuous_int_dep_acc": 0.511,
                "lowrank_K2_int_dep_acc": 0.698,
            },
            "W16_baseline": {
                "best_int_dep_acc": 0.022,
                "best_condition": "lowrank_K2",
            },
        },
        "conditions": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(args.output_dir, "w8_gate_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()

