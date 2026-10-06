import os
import sys
import json
import argparse
import time
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation

ALL_CONDITIONS = [
    ('global_time_only', 'none', 0),
    ('continuous', 'continuous', 0),
    ('lowrank_K1', 'lowrank', 1),
    ('lowrank_K2', 'lowrank', 2),
    ('lowrank_K4', 'lowrank', 4),
    ('lowrank_K8', 'lowrank', 8),
    ('lowrank_K16', 'lowrank', 16),
    ('quantized_K1', 'quantized', 1),
    ('quantized_K2', 'quantized', 2),
    ('quantized_K4', 'quantized', 4),
    ('quantized_K8', 'quantized', 8),
    ('quantized_K16', 'quantized', 16),
]

def main():
    parser = argparse.ArgumentParser(description="Full WxK sweep for Ordered Assembly")
    parser.add_argument('--W', type=int, required=True, help="Width of the assembly tree")
    parser.add_argument('--steps', type=int, default=50000, help="Training steps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-eval-samples', type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory")
    parser.add_argument('--skip-completed', action='store_true', help="Skip if metrics file exists")
    parser.add_argument('--conditions', type=str, nargs='+', help="Specific conditions to run")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    output_dir = args.output_dir if args.output_dir else f"wk_sweep_W{args.W}_results"
    ensure_dir(output_dir)
    
    vocab_size = 256
    embed_dim = 512
    seq_len = 64
    
    dataset = OrderedAssemblyDataset(
        num_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        W=args.W,
        seed=args.seed
    )
    
    conditions_to_run = ALL_CONDITIONS
    if args.conditions:
        conditions_to_run = [c for c in ALL_CONDITIONS if c[0] in args.conditions]
    
    summary_results = []
    
    for name, mode, K in conditions_to_run:
        print(f"\n{'='*50}\nRunning condition: {name} (mode: {mode}, K: {K})\n{'='*50}")
        
        metrics_file = os.path.join(output_dir, f"{name}_metrics.json")
        if args.skip_completed and os.path.exists(metrics_file):
            print(f"Skipping {name}, metrics file already exists.")
            with open(metrics_file, 'r') as f:
                summary_results.append(json.load(f))
            continue
            
        lowrank_basis = 'learned' if mode == 'lowrank' else 'fourier'
        
        model = EFM_Small(
            vocab_size=vocab_size,
            local_time_mode=mode,
            local_time_K=K,
            lowrank_basis=lowrank_basis,
        ).to(device)
        
        local_time_training = 'none' if mode == 'none' else 'expansion'
        checkpoint_dir = os.path.join(output_dir, 'checkpoints', name)
        ensure_dir(checkpoint_dir)
        
        # Train
        train(
            model=model,
            dataset=dataset,
            device=device,
            max_steps=args.steps,
            batch_size=args.batch_size,
            save_every=10000,
            log_every=500,
            checkpoint_dir=checkpoint_dir,
            experiment_name=name,
            output_dir=output_dir,
            local_time_training=local_time_training,
        )
        
        # Evaluate
        print(f"Generating {args.num_eval_samples} evaluation samples for {name}...")
        generated_ids = generate_synthetic_samples(
            model=model,
            dataset=dataset,
            device=device,
            num_samples=args.num_eval_samples,
            local_time_mode=mode,
            W=args.W
        )
        
        print(f"Evaluating samples for {name}...")
        eval_metrics = full_evaluation(
            generated_ids.cpu(),
            dataset.token_ids[:args.num_eval_samples].cpu(),
            dataset.wave_ids[:args.num_eval_samples].cpu(),
            vocab_size=vocab_size
        )
        
        metrics = {
            'experiment_name': name,
            'W': args.W,
            'K': K,
            'mode': mode,
            'steps': args.steps,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'model_params': sum(p.numel() for p in model.parameters()),
            'metrics': eval_metrics
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        summary_results.append(metrics)
        
    summary_file = os.path.join(output_dir, f"sweep_summary_W{args.W}_s{args.seed}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
        
    print(f"\n{'='*50}\nSweep Summary W={args.W}\n{'='*50}")
    
    # Sort by overall_internal_dep_acc if available
    def get_sort_key(res):
        m = res.get('metrics', {})
        return m.get('overall_internal_dep_acc', m.get('internal_dep_acc', 0.0))
        
    summary_results.sort(key=get_sort_key, reverse=True)
    
    print(f"{'Experiment':<20} | {'Mode':<12} | {'K':<4} | {'Int. Dep Acc':<15} | {'Ext. Dep Acc':<15}")
    print("-" * 75)
    for res in summary_results:
        m = res.get('metrics', {})
        int_acc = m.get('overall_internal_dep_acc', 0.0)
        ext_acc = m.get('overall_dep_acc', 0.0)
        print(f"{res['experiment_name']:<20} | {res['mode']:<12} | {res['K']:<4} | {int_acc:<15.4f} | {ext_acc:<15.4f}")

if __name__ == '__main__':
    main()
