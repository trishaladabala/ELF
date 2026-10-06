import os
import sys
import argparse
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.synthetic.synthetic_runner import generate_synthetic_samples
from efm_pipeline.synthetic.ground_truth_eval import full_evaluation

def main():
    parser = argparse.ArgumentParser(description="Run W=16 Matched-Compute Baseline")
    parser.add_argument("--W", type=int, default=16, help="Window size for ordered assembly")
    parser.add_argument("--steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-eval-samples", type=int, default=1000, help="Number of samples to generate for evaluation")
    parser.add_argument("--output-dir", type=str, default="baseline_results", help="Directory to save results")
    parser.add_argument("--skip-completed", action="store_true", help="Skip already completed conditions")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    conditions = [
        ('global_time_only', 'none', 0),
        ('continuous', 'continuous', 0),
        ('quantized_K2', 'quantized', 2),
        ('quantized_K4', 'quantized', 4),
        ('lowrank_K2', 'lowrank', 2),
        ('lowrank_K4', 'lowrank', 4)
    ]

    vocab_size = 256
    seq_len = 64
    embed_dim = 512
    num_train_samples = 5000

    print(f"Generating OrderedAssemblyDataset with W={args.W}, seed={args.seed}")
    dataset = OrderedAssemblyDataset(
        num_samples=num_train_samples, 
        seq_len=seq_len, 
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        W=args.W, 
        seed=args.seed
    )

    summary = {}

    for name, mode, K in conditions:
        print(f"\n{'='*50}\nRunning condition: {name} (mode={mode}, K={K})\n{'='*50}")
        exp_name = f'baseline_W{args.W}_{name}_s{args.seed}'
        metrics_path = os.path.join(args.output_dir, f"{exp_name}_metrics.json")

        if args.skip_completed and os.path.exists(metrics_path):
            print(f"Skipping {name} - already completed.")
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                summary[name] = data['metrics']
            continue

        lowrank_basis = 'learned' if mode == 'lowrank' else 'fourier'
        
        model = EFM_Small(
            vocab_size=vocab_size,
            local_time_mode=mode,
            local_time_K=K,
            lowrank_basis=lowrank_basis,
        ).to(device)

        local_time_training = 'none' if mode == 'none' else 'expansion'
        checkpoint_dir = f"checkpoints/{exp_name}"
        ensure_dir(checkpoint_dir)

        print(f"Training...")
        train(
            model=model,
            dataset=dataset,
            device=device,
            max_steps=args.steps,
            batch_size=args.batch_size,
            save_every=10000,
            log_every=500,
            checkpoint_dir=checkpoint_dir,
            experiment_name=exp_name,
            output_dir=args.output_dir,
            local_time_training=local_time_training,
        )

        print(f"Generating and Evaluating...")
        generated_ids = generate_synthetic_samples(
            model=model,
            dataset=dataset,
            device=device,
            num_samples=args.num_eval_samples,
            local_time_mode=mode,
            W=args.W
        )

        metrics = full_evaluation(
            generated_token_ids=generated_ids.cpu(),
            true_token_ids=dataset.token_ids[:args.num_eval_samples].cpu(),
            wave_ids=dataset.wave_ids[:args.num_eval_samples].cpu(),
            vocab_size=vocab_size
        )

        print(f"Metrics for {name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        summary[name] = metrics

        results = {
            "experiment_name": exp_name,
            "W": args.W,
            "condition": name,
            "local_time_mode": mode,
            "K": K,
            "seed": args.seed,
            "steps": args.steps,
            "metrics": metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)

    summary_path = os.path.join(args.output_dir, f"baseline_summary_W{args.W}_s{args.seed}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary W={args.W}, seed={args.seed}:")
    print(f"{'Condition':<20} | {'Ext Dep Acc':<12} | {'Int Dep Acc':<12} | {'Exact Match':<12}")
    print("-" * 62)
    for name, m in summary.items():
        ext = m.get('overall_dep_acc', 0.0)
        int_dep = m.get('overall_internal_dep_acc', 0.0)
        exact = m.get('overall_exact_match', 0.0)
        print(f"{name:<20} | {ext:<12.4f} | {int_dep:<12.4f} | {exact:<12.4f}")

if __name__ == "__main__":
    main()
