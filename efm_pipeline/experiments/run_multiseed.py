import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation

CONDITION_MAPPING = {
    'global_time_only': ('none', 0),
    'continuous': ('continuous', 0),
    'lowrank_K1': ('lowrank', 1),
    'lowrank_K2': ('lowrank', 2),
    'lowrank_K4': ('lowrank', 4),
    'lowrank_K8': ('lowrank', 8),
    'lowrank_K16': ('lowrank', 16),
    'quantized_K1': ('quantized', 1),
    'quantized_K2': ('quantized', 2),
    'quantized_K4': ('quantized', 4),
    'quantized_K8': ('quantized', 8),
    'quantized_K16': ('quantized', 16),
}

def main():
    parser = argparse.ArgumentParser(description="Multi-seed validation for Ordered Assembly task.")
    parser.add_argument('--W', type=int, required=True, help="Wave size W")
    parser.add_argument('--steps', type=int, default=50000, help="Training steps")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137, 2024], help="Seeds to test")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-eval-samples', type=int, default=1000, help="Number of eval samples")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory")
    parser.add_argument('--skip-completed', action='store_true', help="Skip completed experiments")
    parser.add_argument('--conditions', nargs='+', default=['global_time_only', 'continuous', 'lowrank_K2', 'lowrank_K4'], help="Conditions to test")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"multiseed_W{args.W}_results"
        
    ensure_dir(args.output_dir)
    device = get_device()
    
    results = {cond: {} for cond in args.conditions}
    
    for condition in args.conditions:
        if condition not in CONDITION_MAPPING:
            print(f"Warning: Unknown condition '{condition}', skipping.")
            continue
            
        mode, K = CONDITION_MAPPING[condition]
        lowrank_basis = 'learned' if mode == 'lowrank' else 'fourier'
        local_time_training = 'expansion' if mode != 'none' else 'none'
        
        for seed in args.seeds:
            exp_name = f"W{args.W}_{condition}_s{seed}"
            metrics_path = os.path.join(args.output_dir, f"{exp_name}_metrics.json")
            
            if args.skip_completed and os.path.exists(metrics_path):
                print(f"Skipping completed experiment: {exp_name}")
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                results[condition][seed] = metrics
                continue
                
            print(f"\n{'='*50}\nRunning {exp_name}\n{'='*50}")
            set_seed(seed)
            
            dataset = OrderedAssemblyDataset(
                num_samples=5000, 
                seq_len=64, 
                vocab_size=256, 
                embed_dim=512, 
                W=args.W, 
                seed=seed
            )
            
            model = EFM_Small(
                vocab_size=256,
                local_time_mode=mode,
                local_time_K=K,
                lowrank_basis=lowrank_basis
            )
            
            train(
                model=model,
                dataset=dataset,
                device=device,
                max_steps=args.steps,
                batch_size=args.batch_size,
                use_amp=True,
                save_every=10000,
                log_every=500,
                checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
                experiment_name=exp_name,
                output_dir=args.output_dir,
                local_time_training=local_time_training
            )
            
            # Evaluate
            generated_ids = generate_synthetic_samples(
                model=model,
                dataset=dataset,
                device=device,
                num_samples=args.num_eval_samples,
                local_time_mode=mode,
                W=args.W
            )
            
            eval_metrics = full_evaluation(
                generated_ids.cpu(),
                dataset.token_ids[:args.num_eval_samples].cpu(),
                dataset.wave_ids[:args.num_eval_samples].cpu(),
                vocab_size=256
            )
            
            metrics = {
                'W': args.W,
                'condition': condition,
                'seed': seed,
                'mode': mode,
                'K': K,
                'eval_metrics': eval_metrics
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            results[condition][seed] = metrics

    # Aggregate stats
    print("\n" + "="*60)
    print("MULTI-SEED SUMMARY")
    print(f"{'Condition':<20} | {'Internal Dep Acc':<20} | {'Exact Match':<15}")
    print("-" * 60)
    
    summary = {}
    for condition in args.conditions:
        if condition not in results:
            continue
            
        seed_results = results[condition]
        if not seed_results:
            continue
            
        internal_accs = [res['eval_metrics'].get('overall_internal_dep_acc', 0) for res in seed_results.values()]
        exact_matches = [res['eval_metrics'].get('overall_exact_match', 0) for res in seed_results.values()]
        
        mean_internal = np.mean(internal_accs)
        std_internal = np.std(internal_accs)
        mean_exact = np.mean(exact_matches)
        std_exact = np.std(exact_matches)
        
        print(f"{condition:<20} | {mean_internal:.4f} ± {std_internal:.4f}  | {mean_exact:.4f} ± {std_exact:.4f}")
        
        summary[condition] = {
            'internal_dep_acc_mean': float(mean_internal),
            'internal_dep_acc_std': float(std_internal),
            'exact_match_mean': float(mean_exact),
            'exact_match_std': float(std_exact),
            'seeds': {k: {'internal_dep_acc': v['eval_metrics'].get('overall_internal_dep_acc', 0), 
                          'exact_match': v['eval_metrics'].get('overall_exact_match', 0)} for k, v in seed_results.items()}
        }
        
    summary_path = os.path.join(args.output_dir, f"multiseed_summary_W{args.W}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved summary to {summary_path}")

if __name__ == '__main__':
    main()
