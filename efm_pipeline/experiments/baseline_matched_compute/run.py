#!/usr/bin/env python3
"""Matched-compute baseline: global-time-only EFM on the synthetic task.

Reuses the existing training loop with local_time_mode="none" to show
that the U-shaped compression curve does NOT appear without per-token
local-time conditioning.

Usage:
    python experiments/baseline_matched_compute/run.py \
        --task <synthetic_task_name> \
        --steps 50000 \
        --seed 42
"""
import os
import sys
import argparse

# Setup path so we can import from efm_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train, EmbeddingDataset
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.utils import set_seed, get_device, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Matched-compute baseline (global time only)")
    parser.add_argument("--task", type=str, default="synthetic",
                        help="Synthetic task name (must match a dataset loader)")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="baseline_results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
    from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
    from efm_pipeline.synthetic.ground_truth_eval import full_evaluation
    import json

    # --- Load the synthetic dataset ---
    seq_len = 64
    vocab_size = 256
    W = 4
    dataset = OrderedAssemblyDataset(
        num_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size,
        embed_dim=512,  # matches EFM_Small
        W=W,
        seed=args.seed
    )

    # --- Build model: global-time-only (no local-time conditioner) ---
    model = EFM_Small(
        vocab_size=vocab_size,
        local_time_mode="none",
    ).to(device)

    # --- Train ---
    train(
        model=model,
        dataset=dataset,
        device=device,
        max_steps=args.steps,
        batch_size=32,
        save_every=10000,
        log_every=500,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        experiment_name=f"baseline_global_time_s{args.seed}",
        output_dir=args.output_dir,
        local_time_training="none",
    )

    # --- Evaluate ---
    print("\n--- Evaluating Baseline ---")
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=1000, local_time_mode="none", W=W
    )
    
    metrics = full_evaluation(
        generated_token_ids=generated_ids.cpu(),
        true_token_ids=dataset.token_ids[:1000].cpu(),
        wave_ids=dataset.wave_ids[:1000].cpu(),
        vocab_size=vocab_size,
    )
    
    out_file = os.path.join(args.output_dir, f"baseline_metrics_s{args.seed}.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Overall Dependency Accuracy: {metrics['overall_dep_acc']:.4f}")
    print(f"Overall Exact Match: {metrics['overall_exact_match']:.4f}")


if __name__ == "__main__":
    main()

