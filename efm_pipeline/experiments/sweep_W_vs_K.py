#!/usr/bin/env python3
"""Main synthetic task sweep: W vs K.

Trains EFM_Small (36M params) on the Ordered Assembly task for 50k steps.
Tests the U-shaped compression hypothesis across different local time modes:
- continuous
- quantized K=2
- quantized K=4
- lowrank K=2
- lowrank K=4
- none (global time only)
"""
import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir

from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation


def run_condition(
    name: str,
    dataset,
    device: torch.device,
    W: int,
    out_dir: str,
    max_steps: int,
    mode: str,
    K: int = 16,
    batch_size: int = 32,
):
    print(f"\n" + "="*50)
    print(f"Condition: {name}")
    print("="*50)
    
    ckpt_dir = os.path.join(out_dir, "checkpoints", name)
    metrics_file = os.path.join(out_dir, f"{name}_metrics.json")
    
    if os.path.exists(metrics_file):
        print(f"Already evaluated {name}, skipping.")
        with open(metrics_file) as f:
            return json.load(f)

    # 1. Build Model
    model = EFM_Small(
        vocab_size=dataset.vocab_size,
        local_time_mode=mode,
        local_time_K=K,
        lowrank_basis="learned" if mode == "lowrank" else "fourier",
    ).to(device)
    
    # 2. Train
    local_time_training = "expansion" if mode != "none" else "none"
    train(
        model=model,
        dataset=dataset,
        device=device,
        max_steps=max_steps,
        batch_size=batch_size,
        save_every=10000,
        log_every=500,
        checkpoint_dir=ckpt_dir,
        experiment_name=name,
        output_dir=out_dir,
        local_time_training=local_time_training,
        use_amp=True,
    )
    
    # 3. Evaluate
    print(f"\nEvaluating {name}...")
    # Load best/last checkpoint if needed, but train() leaves model in final state
    model.eval()
    
    num_eval_samples = 1000
    generated_ids = generate_synthetic_samples(
        model=model, dataset=dataset, device=device,
        num_samples=num_eval_samples, local_time_mode=mode, W=W
    )
    
    metrics = full_evaluation(
        generated_token_ids=generated_ids.cpu(),
        true_token_ids=dataset.token_ids[:num_eval_samples].cpu(),
        wave_ids=dataset.wave_ids[:num_eval_samples].cpu(),
        vocab_size=dataset.vocab_size,
    )
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"  Dependency Accuracy: {metrics['overall_dep_acc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", type=int, default=4, help="Number of waves (task complexity)")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps per condition")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    out_dir = f"synthetic_W{args.W}_results"
    ensure_dir(out_dir)
    
    print(f"Generating Ordered Assembly Dataset (W={args.W})...")
    dataset = OrderedAssemblyDataset(
        num_samples=5000,
        seq_len=64,
        vocab_size=256,
        embed_dim=512,  # matches EFM_Small
        W=args.W,
        seed=args.seed
    )
    
    conditions = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("quantized_K2", "quantized", 2),
        ("quantized_K4", "quantized", 4),
        ("lowrank_K2", "lowrank", 2),
        ("lowrank_K4", "lowrank", 4),
    ]
    
    all_metrics = {}
    for name, mode, K in conditions:
        exp_name = f"W{args.W}_{name}_s{args.seed}"
        metrics = run_condition(
            name=exp_name,
            dataset=dataset,
            device=device,
            W=args.W,
            out_dir=out_dir,
            max_steps=args.steps,
            mode=mode,
            K=K,
        )
        all_metrics[name] = metrics
        
    print("\n" + "="*60)
    print(f"SUMMARY FOR W={args.W}")
    print("="*60)
    print(f"{'Condition':<20} {'Dep Acc':>10}")
    print("-" * 32)
    for name, mode, K in conditions:
        acc = all_metrics[name].get('overall_dep_acc', 0)
        print(f"{name:<20} {acc:>10.4f}")


if __name__ == "__main__":
    main()

