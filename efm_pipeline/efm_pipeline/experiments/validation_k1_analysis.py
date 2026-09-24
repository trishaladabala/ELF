import os
import sys
import json
import argparse
import torch

# Setup python path to include ELF root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.utils import set_seed, get_device, ensure_dir
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Analyze K=1 frozen result for EFM")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_val", help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="validation_phase", help="Directory for output results")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    ensure_dir(args.output_dir)

    print("Loading continuous_local_time checkpoint...")
    ckpt_path = os.path.join(args.checkpoint_dir, "continuous_local_time/continuous_local_time/checkpoint_50000.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found. Ensure continuous_local_time was trained.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # We assume EFM_Small based on the validation controls defaults.
    model = EFM_Small(vocab_size=32128).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Generating samples with local_time_mode='constant_0' (K=1 / frozen)...")
    with torch.no_grad():
        generated_texts = generate_efm_samples(
            model,
            num_samples=128,  # A reasonably small number of samples for validation eval
            batch_size=32,
            tokenizer=tokenizer,
            local_time_mode="constant_0",
            device=device
        )

    print("Evaluating K=1 frozen texts...")
    frozen_metrics = evaluate(generated_texts, experiment_name="frozen_k1", output_dir=args.output_dir)
    frozen_gen_ppl = frozen_metrics.get("gen_ppl", float('inf'))

    print(f"Frozen K=1 Gen-PPL: {frozen_gen_ppl:.4f}")
    
    # Save frozen result
    frozen_result_path = os.path.join(args.output_dir, "frozen_k1_result.json")
    with open(frozen_result_path, "w") as f:
        json.dump(frozen_metrics, f, indent=4)
    print(f"Saved results to {frozen_result_path}")

    # Load previous results for comparison
    def load_result(name):
        path = os.path.join(args.output_dir, f"{name}_result.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f).get("gen_ppl", float('inf'))
        return float('inf')

    global_time_only_ppl = load_result("global_time_only")
    continuous_local_time_ppl = load_result("continuous_local_time")

    print("\n### Validation Phase Comparison")
    print("| Model Configuration | Gen-PPL |")
    print("| :--- | :--- |")
    print(f"| Global Time Only (no local time) | {global_time_only_ppl:.4f} |")
    print(f"| Continuous Local Time (standard) | {continuous_local_time_ppl:.4f} |")
    print(f"| Frozen K=1 (eval at tau=0) | {frozen_gen_ppl:.4f} |")
    print("\nThis comparison shows whether setting tau=0 (K=1) just mimics the global-time-only model, or provides distinct benefits.")

if __name__ == "__main__":
    main()
