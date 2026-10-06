#!/usr/bin/env python3
"""Diagnose the W=4 pilot vs baseline discrepancy.

Loads existing pilot (5k step) and baseline (50k step) checkpoints for
the three shared conditioning modes, then compares:
  - Training step count from checkpoint metadata
  - Parameter count
  - Teacher-forced NLL / accuracy
  - Generation-based internal & external dependency accuracy

Does NOT retrain anything — pure diagnostic.

Usage (from efm_pipeline/):
    python -m experiments.diagnose_w4_discrepancy
    python -m experiments.diagnose_w4_discrepancy --num-eval-samples 200
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation


def evaluate_tf(model, dataset, device, max_samples=1000):
    """Teacher-forced evaluation at t=1 (same approach as baseline_matched_compute/run.py)."""
    model.eval()
    bs = min(max_samples, len(dataset.embeddings))
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


def load_and_evaluate(ckpt_path, mode, K, dataset, device, W, num_eval_samples, label):
    """Load a checkpoint, evaluate TF and generation metrics."""
    if not os.path.exists(ckpt_path):
        print(f"  [{label}] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n--- {label} ---")
    print(f"  Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get('step', -1)
    final_loss = ckpt.get('loss', None)
    print(f"  Saved at step: {step}, final loss: {final_loss}")

    # Build model with matching architecture
    model = EFM_Small(
        vocab_size=256,
        local_time_mode=mode,
        local_time_K=K,
        lowrank_basis="learned" if mode == "lowrank" else "fourier",
    ).to(device)

    # Load weights
    try:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  ✓ Loaded state_dict successfully")
    except RuntimeError as e:
        print(f"  ⚠ State dict mismatch: {e}")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded with strict=False")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Teacher-forced evaluation
    tf_nll, tf_acc = evaluate_tf(model, dataset, device)
    print(f"  TF NLL:  {tf_nll:.4f}")
    print(f"  TF Acc:  {tf_acc:.4f}")

    # Generation evaluation
    print(f"  Generating {num_eval_samples} samples...")
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=num_eval_samples, num_steps=64,
        batch_size=64, seed=42,
        local_time_mode=mode if mode != "none" else "none",
        W=W,
    )

    metrics = full_evaluation(
        generated_token_ids=generated_ids[:num_eval_samples].cpu(),
        true_token_ids=dataset.token_ids[:num_eval_samples].cpu(),
        wave_ids=dataset.wave_ids[:num_eval_samples].cpu(),
        vocab_size=256,
    )

    print(f"  External dep acc:  {metrics['overall_dep_acc']:.4f}")
    print(f"  Internal dep acc:  {metrics['overall_internal_dep_acc']:.4f}")
    print(f"  Exact match:       {metrics['overall_exact_match']:.4f}")

    return {
        'step': step,
        'final_loss': final_loss,
        'params_total': num_params,
        'params_trainable': num_trainable,
        'tf_nll': tf_nll,
        'tf_acc': tf_acc,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose W=4 pilot vs baseline discrepancy")
    parser.add_argument('--num-eval-samples', type=int, default=500)
    parser.add_argument('--W', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='discrepancy_diagnosis')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.out_dir)

    W = args.W
    seed = args.seed

    # Build evaluation dataset (same as what both pilot and baseline used)
    dataset = OrderedAssemblyDataset(
        num_samples=5000, seq_len=64, vocab_size=256,
        embed_dim=512, W=W, seed=seed,
    )

    # Modes shared between pilot and baseline
    modes = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("lowrank_K4", "lowrank", 4),
    ]

    # Checkpoint path patterns
    pilot_pattern = f"synthetic_pilot_W{W}_results/checkpoints/W{W}_{{name}}_s{seed}/W{W}_{{name}}_s{seed}/checkpoint_5000.pt"
    baseline_pattern = f"baseline_results/checkpoints/baseline_W{W}_{{name}}_s{seed}/baseline_W{W}_{{name}}_s{seed}/checkpoint_50000.pt"

    report = {}

    print("=" * 70)
    print(f"  DIAGNOSING W={W} PILOT (5k) vs BASELINE (50k) DISCREPANCY")
    print("=" * 70)

    for name, mode, K in modes:
        report[name] = {}

        pilot_path = pilot_pattern.format(name=name)
        baseline_path = baseline_pattern.format(name=name)

        res_pilot = load_and_evaluate(
            pilot_path, mode, K, dataset, device, W,
            args.num_eval_samples, f"PILOT {name}"
        )
        if res_pilot:
            report[name]['pilot'] = res_pilot

        res_baseline = load_and_evaluate(
            baseline_path, mode, K, dataset, device, W,
            args.num_eval_samples, f"BASELINE {name}"
        )
        if res_baseline:
            report[name]['baseline'] = res_baseline

    # Print comparison table
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Mode':<20} {'Source':<10} {'Steps':<8} {'TF Acc':<8} {'Int Dep':<10} {'Ext Dep':<10}")
    print("-" * 90)
    for name, _, _ in modes:
        for source in ['pilot', 'baseline']:
            if source in report.get(name, {}):
                r = report[name][source]
                print(f"{name:<20} {source:<10} {r['step']:<8} {r['tf_acc']:<8.4f} "
                      f"{r['overall_internal_dep_acc']:<10.4f} {r['overall_dep_acc']:<10.4f}")
        print()

    # Save
    report_path = os.path.join(args.out_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved → {report_path}")


if __name__ == '__main__':
    main()
