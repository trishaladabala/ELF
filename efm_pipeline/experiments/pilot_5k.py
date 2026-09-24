#!/usr/bin/env python3
"""5k-step Pilot Sweep for Ordered Assembly.

Runs W=4 and W=16 tasks for 5,000 steps each.
Conditions tested:
1. global_time_only
2. continuous local time
3. lowrank K=4 (the best performer from anchor finding)

Usage:
    python experiments/pilot_5k.py
"""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from experiments.sweep_W_vs_K import run_condition
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.utils import set_seed, get_device, ensure_dir


def main():
    device = get_device()
    seed = 42
    set_seed(seed)
    
    steps = 5000
    batch_size = 32
    
    conditions = [
        ("global_time_only", "none", 0),
        ("continuous", "continuous", 0),
        ("lowrank_K4", "lowrank", 4),
    ]
    
    all_results = {}
    
    for W in [4, 16]:
        print(f"\n" + "#"*60)
        print(f"PILOT FOR W={W}")
        print("#"*60)
        
        out_dir = f"synthetic_pilot_W{W}_results"
        ensure_dir(out_dir)
        
        dataset = OrderedAssemblyDataset(
            num_samples=5000, seq_len=64, vocab_size=256,
            embed_dim=512, W=W, seed=seed
        )
        
        all_results[W] = {}
        for name, mode, K in conditions:
            exp_name = f"W{W}_{name}_s{seed}"
            metrics = run_condition(
                name=exp_name,
                dataset=dataset,
                device=device,
                W=W,
                out_dir=out_dir,
                max_steps=steps,
                mode=mode,
                K=K,
                batch_size=batch_size
            )
            all_results[W][name] = metrics

    print("\n" + "="*60)
    print("PILOT RESULTS SUMMARY (5k steps)")
    print("="*60)
    for W in [4, 16]:
        print(f"\nTask Complexity: W={W}")
        print(f"{'Condition':<20} {'Dep Acc':>10}")
        print("-" * 32)
        for name, _, _ in conditions:
            acc = all_results[W][name].get('overall_dep_acc', 0)
            print(f"{name:<20} {acc:>10.4f}")


if __name__ == "__main__":
    main()

