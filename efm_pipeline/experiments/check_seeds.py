#!/usr/bin/env python3
"""Run Continuous condition across 3 seeds for W=4 and W=16."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.experiments.sweep_W_vs_K import run_condition
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.utils import get_device, ensure_dir

def main():
    device = get_device()
    seeds = [42, 137, 2024]
    
    results = {4: [], 16: []}
    
    for W in [4, 16]:
        out_dir = f"synthetic_seeds_W{W}_results"
        ensure_dir(out_dir)
        
        for seed in seeds:
            dataset = OrderedAssemblyDataset(
                num_samples=5000, seq_len=64, vocab_size=256,
                embed_dim=512, W=W, seed=seed
            )
            
            name = f"W{W}_continuous_s{seed}"
            metrics = run_condition(
                name=name,
                dataset=dataset,
                device=device,
                W=W,
                out_dir=out_dir,
                max_steps=5000,
                mode="continuous",
                K=0,
                batch_size=32
            )
            results[W].append(metrics.get("overall_dep_acc", 0.0))
            
    print("\n" + "="*50)
    print("CONTINUOUS N=3 SUMMARY")
    print("="*50)
    for W in [4, 16]:
        accs = results[W]
        mean = np.mean(accs)
        std = np.std(accs)
        print(f"W={W}: {mean:.4f} ± {std:.4f}  (Runs: {[f'{x:.4f}' for x in accs]})")

if __name__ == "__main__":
    main()

