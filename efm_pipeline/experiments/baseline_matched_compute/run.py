#!/usr/bin/env python3
"""Matched-compute baseline: sweep local-time modes on the synthetic task.

Trains EFM_Small (36M params, 50k steps) on the Ordered Assembly task
across multiple local-time conditioning modes to verify the U-shaped
compression curve on synthetic data.

This is the synthetic-task counterpart of the WikiText-2 validation phase.
It rules out the possibility that the U-shape is an artifact of model size
or general training dynamics.

Usage (from efm_pipeline/):
    python -m experiments.baseline_matched_compute.run
    python -m experiments.baseline_matched_compute.run --steps 50000
    python -m experiments.baseline_matched_compute.run --W 16
"""
import os
import sys
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="Matched-compute baseline sweep")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--W", type=int, default=4, help="Task complexity (number of waves)")
    parser.add_argument("--output_dir", type=str, default="baseline_results")
    parser.add_argument("--num_gen_samples", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    # Conditions to sweep — this is the U-shape test
    conditions = [
        ("global_time_only",  "none",      0, "none"),
        ("continuous",        "continuous", 0, "expansion"),
        ("quantized_K2",      "quantized", 2, "expansion"),
        ("quantized_K4",      "quantized", 4, "expansion"),
        ("lowrank_K2",        "lowrank",   2, "expansion"),
        ("lowrank_K4",        "lowrank",   4, "expansion"),
    ]

    dataset = OrderedAssemblyDataset(
        num_samples=5000, seq_len=64, vocab_size=256,
        embed_dim=512, W=args.W, seed=args.seed,
    )

    all_results = {}

    for name, mode, K, lt_training in conditions:
        exp_name = f"baseline_W{args.W}_{name}_s{args.seed}"
        ckpt_dir = os.path.join(args.output_dir, "checkpoints", exp_name)

        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"  Mode: {mode}, K: {K}, Steps: {args.steps}")
        print(f"{'='*60}")

        model = EFM_Small(
            vocab_size=256,
            local_time_mode=mode,
            local_time_K=K,
            lowrank_basis="learned" if mode == "lowrank" else "fourier",
        ).to(device)

        # Train
        train(
            model=model, dataset=dataset, device=device,
            max_steps=args.steps, batch_size=32,
            save_every=10000, log_every=500,
            checkpoint_dir=ckpt_dir, experiment_name=exp_name,
            output_dir=args.output_dir,
            local_time_training=lt_training,
        )

        # Teacher-forced evaluation
        tf_nll, tf_acc = evaluate_tf(model, dataset, device)

        # Generation evaluation (with corrected local-time routing)
        print(f"\n  Generating {args.num_gen_samples} samples...")
        generated_ids = generate_synthetic_samples(
            model=model, dataset=dataset, device=device,
            num_samples=args.num_gen_samples, num_steps=64,
            batch_size=64, seed=args.seed,
            local_time_mode=mode if mode != "none" else "none",
            W=args.W,
        )

        metrics = full_evaluation(
            generated_token_ids=generated_ids[:args.num_gen_samples].cpu(),
            true_token_ids=dataset.token_ids[:args.num_gen_samples].cpu(),
            wave_ids=dataset.wave_ids[:args.num_gen_samples].cpu(),
            vocab_size=256,
        )

        result = {
            "experiment": exp_name,
            "W": args.W,
            "mode": mode,
            "K": K,
            "steps": args.steps,
            "tf_nll": tf_nll,
            "tf_acc": tf_acc,
            **metrics,
        }
        all_results[name] = result

        # Save per-condition result
        with open(os.path.join(args.output_dir, f"{exp_name}_metrics.json"), "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n  Results for {name}:")
        print(f"    TF NLL:           {tf_nll:.4f}")
        print(f"    TF Accuracy:      {tf_acc:.4f}")
        print(f"    Ext Dep Acc:      {metrics['overall_dep_acc']:.4f}")
        print(f"    Int Dep Acc:      {metrics['overall_internal_dep_acc']:.4f}")
        print(f"    Exact Match:      {metrics['overall_exact_match']:.4f}")

    # Save combined summary
    summary_file = os.path.join(args.output_dir, f"baseline_summary_W{args.W}_s{args.seed}.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  BASELINE MATCHED-COMPUTE RESULTS (W={args.W}, {args.steps} steps)")
    print(f"{'='*80}")
    print(f"{'Condition':<22s} {'TF NLL':>8s} {'TF Acc':>8s} {'Int Dep':>8s} {'Ext Dep':>8s} {'Exact':>8s}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<22s} {r['tf_nll']:>8.4f} {r['tf_acc']:>8.4f} "
              f"{r['overall_internal_dep_acc']:>8.4f} {r['overall_dep_acc']:>8.4f} "
              f"{r['overall_exact_match']:>8.4f}")
    print("=" * 80)
    print(f"\nSummary saved → {summary_file}")


if __name__ == "__main__":
    main()
