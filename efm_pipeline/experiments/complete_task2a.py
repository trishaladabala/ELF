#!/usr/bin/env python3
"""Complete the Task 2a sigma sweep that was interrupted.

Previous run completed:
  - W=4, sigma=0.0: all 3 conditions ✓
  - W=4, sigma=0.05: all 3 conditions ✓
  - W=4, sigma=0.1: global_time_only only ✓

This script picks up where it left off and completes:
  - W=4, sigma=0.1: continuous, lowrank_K4
  - W=4, sigma=0.2: all 3 conditions
  - W=16, sigma=0.0 through 0.2: all 3 conditions (12 runs)

Also re-evaluates with corrected generation (fixed synthetic_runner.py)
and reports internal dependency accuracy alongside TF metrics.

Usage (from efm_pipeline/):
    python -m experiments.complete_task2a
    python -m experiments.complete_task2a --skip-completed   # skip already-done runs
    python -m experiments.complete_task2a --reeval-only       # only re-evaluate existing checkpoints
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation


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
        loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

    return loss.item(), acc


def evaluate_generation(model, dataset, device, mode, W, num_samples=512):
    """Generate samples and evaluate with corrected internal consistency."""
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=num_samples, num_steps=64,
        batch_size=64, seed=42,
        local_time_mode=mode if mode != "none" else "none",
        W=W,
    )
    metrics = full_evaluation(
        generated_token_ids=generated_ids[:num_samples].cpu(),
        true_token_ids=dataset.token_ids[:num_samples].cpu(),
        wave_ids=dataset.wave_ids[:num_samples].cpu(),
        vocab_size=256,
    )
    return metrics


def is_completed(out_dir, exp_name):
    """Check if a run already has a completed checkpoint."""
    ckpt_dir = os.path.join(out_dir, "checkpoints", exp_name)
    if not os.path.isdir(ckpt_dir):
        return False
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt"):
                return True
    return False


def load_checkpoint(model, out_dir, exp_name, device):
    """Load the latest checkpoint for a given experiment."""
    ckpt_dir = os.path.join(out_dir, "checkpoints", exp_name)
    best_pt = None
    for root, dirs, files in os.walk(ckpt_dir):
        for f in sorted(files, reverse=True):
            if f.endswith(".pt"):
                best_pt = os.path.join(root, f)
                break
    if best_pt is None:
        return False
    ckpt = torch.load(best_pt, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return True


def main():
    parser = argparse.ArgumentParser(description="Complete Task 2a sigma sweep")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip runs that already have checkpoints")
    parser.add_argument("--reeval-only", action="store_true",
                        help="Only re-evaluate existing checkpoints (no training)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps per run (default: 5000)")
    args = parser.parse_args()

    device = get_device()
    set_seed(42)

    out_dir = "task2a_results"
    ensure_dir(out_dir)

    sigmas = [0.0, 0.05, 0.1, 0.2]
    Ws = [4, 16]
    conditions = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("lowrank_K4", "lowrank", 4),
    ]

    # Load existing results if available
    summary_file = os.path.join(out_dir, "summary_v2.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)
    else:
        results = {}

    print("=" * 70)
    print("  TASK 2a: SIGMA SWEEP (COMPLETION RUN)")
    print("=" * 70)

    total_runs = len(Ws) * len(sigmas) * len(conditions)
    completed = 0

    for W in Ws:
        w_key = str(W)
        if w_key not in results:
            results[w_key] = {}

        for sigma in sigmas:
            s_key = str(sigma)
            if s_key not in results[w_key]:
                results[w_key][s_key] = {}

            dataset = OrderedAssemblyDataset(
                num_samples=5000, seq_len=64, vocab_size=256,
                embed_dim=512, W=W, seed=42, sigma=sigma
            )

            for name, mode, K in conditions:
                completed += 1
                exp_name = f"W{W}_sig{sigma}_{name}"

                print(f"\n[{completed}/{total_runs}] --- {exp_name} ---")

                already_done = is_completed(out_dir, exp_name)

                if already_done and args.skip_completed and not args.reeval_only:
                    print(f"  ✓ Already completed, skipping (use --reeval-only to re-evaluate)")
                    continue

                model = EFM_Tiny(
                    vocab_size=256,
                    local_time_mode=mode,
                    local_time_K=K,
                    lowrank_basis="learned" if mode == "lowrank" else "fourier",
                ).to(device)

                if args.reeval_only:
                    if not already_done:
                        print(f"  ⚠ No checkpoint found, skipping")
                        continue
                    print(f"  Loading existing checkpoint for re-evaluation...")
                    if not load_checkpoint(model, out_dir, exp_name, device):
                        print(f"  ✗ Failed to load checkpoint")
                        continue
                elif already_done and args.skip_completed:
                    print(f"  ✓ Already completed, skipping")
                    continue
                else:
                    # Train
                    ckpt_dir = os.path.join(out_dir, "checkpoints", exp_name)
                    local_time_training = "expansion" if mode != "none" else "none"

                    print(f"  Training for {args.steps} steps...")
                    train(
                        model=model, dataset=dataset, device=device,
                        max_steps=args.steps, batch_size=32,
                        save_every=args.steps + 1, log_every=1000,
                        checkpoint_dir=ckpt_dir, experiment_name=exp_name,
                        output_dir=out_dir, local_time_training=local_time_training,
                        use_amp=True,
                    )

                # Evaluate: Teacher-Forced
                tf_nll, tf_acc = evaluate_tf(model, dataset, device)

                # Evaluate: Generation (with corrected code)
                gen_metrics = evaluate_generation(model, dataset, device, mode, W, num_samples=512)

                result = {
                    "tf_nll": tf_nll,
                    "tf_acc": tf_acc,
                    **gen_metrics,
                }
                results[w_key][s_key][name] = result

                print(f"  TF NLL:           {tf_nll:.4f}")
                print(f"  TF Accuracy:      {tf_acc:.4f}")
                print(f"  Ext Dep Acc:      {gen_metrics['overall_dep_acc']:.4f}")
                print(f"  Int Dep Acc:      {gen_metrics['overall_internal_dep_acc']:.4f}")

                # Save after each run (crash resilience)
                with open(summary_file, "w") as f:
                    json.dump(results, f, indent=4)

    # Final summary table
    print(f"\n{'='*80}")
    print(f"  TASK 2a COMPLETE RESULTS")
    print(f"{'='*80}")
    print(f"{'W':>3s}  {'Sigma':>6s}  {'Condition':<18s}  {'TF NLL':>8s}  {'TF Acc':>8s}  {'Int Dep':>8s}  {'Ext Dep':>8s}")
    print("-" * 80)

    for W in Ws:
        w_key = str(W)
        if w_key not in results:
            continue
        for sigma in sigmas:
            s_key = str(sigma)
            if s_key not in results[w_key]:
                continue
            for name, _, _ in conditions:
                if name not in results[w_key][s_key]:
                    continue
                m = results[w_key][s_key][name]
                print(f"{W:>3d}  {sigma:>6.2f}  {name:<18s}  "
                      f"{m['tf_nll']:>8.4f}  {m['tf_acc']:>8.4f}  "
                      f"{m.get('overall_internal_dep_acc', 0):>8.4f}  "
                      f"{m.get('overall_dep_acc', 0):>8.4f}")

    print("=" * 80)
    print(f"\nFull results saved → {summary_file}")


if __name__ == "__main__":
    main()
