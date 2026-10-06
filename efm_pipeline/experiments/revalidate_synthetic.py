#!/usr/bin/env python3
"""Re-validate synthetic pilots with corrected evaluation code.

Two bugs were fixed:
  1. synthetic_runner.py: continuous/lowrank modes now receive oracle timing
     during generation (previously defaulted to zeros).
  2. ground_truth_eval.py: now reports internal_dependency_accuracy (generated
     values vs model's own generated keys) in addition to external dep accuracy.

This script loads existing checkpoints from the W=4 and W=16 pilots,
re-generates samples with corrected local-time routing, and re-evaluates
with both internal and external consistency metrics.

Usage (from efm_pipeline/):
    python -m experiments.revalidate_synthetic
"""
import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation


def evaluate_tf(model, dataset, device):
    """Teacher-forced evaluation: feed ground-truth x0 at t=1, check predictions."""
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


def load_and_evaluate(ckpt_path, mode, K, W, device, dataset, num_gen_samples=1000):
    """Load a checkpoint and run both TF and generation evaluation."""
    model = EFM_Small(
        vocab_size=256,
        local_time_mode=mode,
        local_time_K=K,
        lowrank_basis="learned" if mode == "lowrank" else "fourier",
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle nested state_dict formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # Teacher-forced evaluation
    tf_nll, tf_acc = evaluate_tf(model, dataset, device)

    # Generation evaluation (with corrected local-time routing)
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=num_gen_samples, num_steps=64,
        batch_size=64, seed=42,
        local_time_mode=mode if mode != "none" else "none",
        W=W,
    )

    # Full evaluation (now includes both external and internal dep accuracy)
    metrics = full_evaluation(
        generated_token_ids=generated_ids[:num_gen_samples].cpu(),
        true_token_ids=dataset.token_ids[:num_gen_samples].cpu(),
        wave_ids=dataset.wave_ids[:num_gen_samples].cpu(),
        vocab_size=256,
    )

    metrics["tf_nll"] = tf_nll
    metrics["tf_acc"] = tf_acc

    return metrics


def find_checkpoint(base_dir, name):
    """Find the latest checkpoint for a given experiment name."""
    # Try common checkpoint locations
    candidates = [
        os.path.join(base_dir, "checkpoints", name, name, f"checkpoint_5000.pt"),
        os.path.join(base_dir, "checkpoints", name, f"checkpoint_5000.pt"),
        os.path.join(base_dir, "checkpoints", name, name, f"checkpoint_10000.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: glob for any checkpoint
    ckpt_dir = os.path.join(base_dir, "checkpoints", name)
    if os.path.isdir(ckpt_dir):
        for root, dirs, files in os.walk(ckpt_dir):
            for f in sorted(files, reverse=True):
                if f.endswith(".pt"):
                    return os.path.join(root, f)
    return None


def main():
    device = get_device()
    set_seed(42)

    out_dir = "revalidation_results"
    ensure_dir(out_dir)

    conditions = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("lowrank_K4", "lowrank", 4),
    ]

    all_results = {}

    for W in [4, 16]:
        pilot_dir = f"synthetic_pilot_W{W}_results"
        if not os.path.isdir(pilot_dir):
            print(f"⚠ Pilot directory {pilot_dir} not found, skipping W={W}")
            continue

        print(f"\n{'='*60}")
        print(f"  RE-VALIDATING W={W} (from {pilot_dir})")
        print(f"{'='*60}\n")

        dataset = OrderedAssemblyDataset(
            num_samples=5000, seq_len=64, vocab_size=256,
            embed_dim=512, W=W, seed=42
        )

        all_results[f"W{W}"] = {}

        for name, mode, K in conditions:
            exp_name = f"W{W}_{name}_s42"
            ckpt_path = find_checkpoint(pilot_dir, exp_name)

            if ckpt_path is None:
                print(f"  ⚠ No checkpoint found for {exp_name}, skipping")
                continue

            print(f"\n--- {exp_name} ---")
            print(f"  Checkpoint: {ckpt_path}")

            try:
                metrics = load_and_evaluate(
                    ckpt_path, mode, K, W, device, dataset,
                    num_gen_samples=1000,
                )
                all_results[f"W{W}"][name] = metrics

                print(f"  TF NLL:                {metrics['tf_nll']:.4f}")
                print(f"  TF Accuracy:           {metrics['tf_acc']:.4f}")
                print(f"  External Dep Acc:      {metrics['overall_dep_acc']:.4f}")
                print(f"  Internal Dep Acc:      {metrics['overall_internal_dep_acc']:.4f}")
                print(f"  Exact Match (vs GT):   {metrics['overall_exact_match']:.4f}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    out_file = os.path.join(out_dir, "revalidation_summary.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved → {out_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  REVALIDATION SUMMARY (with bug fixes applied)")
    print(f"{'='*70}")
    print(f"{'Condition':<30s} {'Ext Dep Acc':>12s} {'Int Dep Acc':>12s} {'TF Acc':>10s}")
    print("-" * 70)
    for wkey in sorted(all_results.keys()):
        for cond, m in all_results[wkey].items():
            label = f"{wkey}_{cond}"
            print(f"{label:<30s} {m['overall_dep_acc']:>12.4f} {m['overall_internal_dep_acc']:>12.4f} {m['tf_acc']:>10.4f}")
    print("=" * 70)
    print("\nNote: 'Ext Dep Acc' = generated values vs ground-truth keys")
    print("      'Int Dep Acc' = generated values vs model's own generated keys (PRIMARY)")


if __name__ == "__main__":
    main()
