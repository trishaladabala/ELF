import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.efm_model import EFM_Base
from efm_pipeline.utils import set_seed, get_device, ensure_dir

def get_conditions():
    return {
        'base_global_time_only': {'ckpt': 'base_global_time_only/base_global_time_only/checkpoint_200000.pt', 'mode': 'none', 'K': 0},
        'base_continuous_local_time': {'ckpt': 'base_continuous_local_time/base_continuous_local_time/checkpoint_200000.pt', 'mode': 'continuous', 'K': 0},
        'base_learned_lowrank_K2': {'ckpt': 'base_learned_lowrank_K2/base_learned_lowrank_K2/checkpoint_10000.pt', 'mode': 'lowrank', 'K': 2},
        'base_learned_lowrank_K4': {'ckpt': 'base_learned_lowrank_K4/base_learned_lowrank_K4/checkpoint_10000.pt', 'mode': 'lowrank', 'K': 4},
        'base_learned_lowrank_K8': {'ckpt': 'base_learned_lowrank_K8/base_learned_lowrank_K8/checkpoint_10000.pt', 'mode': 'lowrank', 'K': 8},
        'base_learned_lowrank_K16': {'ckpt': 'base_learned_lowrank_K16/base_learned_lowrank_K16/checkpoint_10000.pt', 'mode': 'lowrank', 'K': 16},
        'base_learned_quantized_K2': {'ckpt': 'base_learned_quantized_K2/base_learned_quantized_K2/checkpoint_10000.pt', 'mode': 'quantized', 'K': 2},
        'base_learned_quantized_K4': {'ckpt': 'base_learned_quantized_K4/base_learned_quantized_K4/checkpoint_10000.pt', 'mode': 'quantized', 'K': 4},
        'base_learned_quantized_K8': {'ckpt': 'base_learned_quantized_K8/base_learned_quantized_K8/checkpoint_10000.pt', 'mode': 'quantized', 'K': 8},
        'base_learned_quantized_K16': {'ckpt': 'base_learned_quantized_K16/base_learned_quantized_K16/checkpoint_10000.pt', 'mode': 'quantized', 'K': 16},
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate scaleup EFM Base models.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_scaleup", help="Directory containing the checkpoints")
    parser.add_argument("--output-dir", type=str, default="scaleup_eval_results", help="Directory to save evaluation results")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of text samples to generate")
    parser.add_argument("--num-steps", type=int, default=32, help="Number of ODE steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-completed", action="store_true", help="Skip already evaluated conditions")
    parser.add_argument("--conditions", type=str, nargs='+', help="Specific conditions to evaluate (names)")
    parser.add_argument("--gen-ppl-model", type=str, default="gpt2-large", help="Model to use for Gen-PPL evaluation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    all_conditions = get_conditions()
    if args.conditions:
        conditions_to_run = {k: v for k, v in all_conditions.items() if k in args.conditions}
    else:
        conditions_to_run = all_conditions

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    summary_file = os.path.join(args.output_dir, "scaleup_eval_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_results = json.load(f)
    else:
        summary_results = {}

    for condition_name, config in conditions_to_run.items():
        print(f"\\n--- Evaluating {condition_name} ---")
        
        out_json_path = os.path.join(args.output_dir, f"{condition_name}_eval.json")
        if args.skip_completed and os.path.exists(out_json_path):
            print(f"Result {out_json_path} already exists. Skipping.")
            continue

        ckpt_path = os.path.join(args.checkpoint_dir, config['ckpt'])
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        mode = config['mode']
        K = config['K']
        lowrank_basis = 'learned' if mode == 'lowrank' else 'fourier'

        model = EFM_Base(vocab_size=32128, local_time_mode=mode, local_time_K=K, lowrank_basis=lowrank_basis)
        
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

        gen_mode = 'none' if mode == 'none' else 'expansion'

        print(f"Generating {args.num_samples} samples with ODE steps {args.num_steps} (mode: {gen_mode})...")
        texts = generate_efm_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            max_length=64,
            seed=args.seed,
            local_time_mode=gen_mode
        )

        print(f"Evaluating Gen-PPL using {args.gen_ppl_model}...")
        results = evaluate(
            texts=texts,
            experiment_name=condition_name,
            output_dir=args.output_dir,
            gen_ppl_model=args.gen_ppl_model
        )

        summary_results[condition_name] = results
        
        with open(out_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=4)

        print(f"Finished {condition_name}. Results: {results}")

    print("\\n--- Summary Table ---")
    print(f"{'Condition':<35} | {'Gen-PPL':<15}")
    print("-" * 55)
    for cond, res in summary_results.items():
        gen_ppl_val = res.get('gen_ppl', 'N/A')
        print(f"{cond:<35} | {gen_ppl_val}")
        
if __name__ == "__main__":
    main()
