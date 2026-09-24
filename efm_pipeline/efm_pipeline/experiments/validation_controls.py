import os
import sys
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer

# Ensure ELF is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train, EmbeddingDataset
from efm_pipeline.eval_runner import generate_efm_samples, evaluate

def main():
    parser = argparse.ArgumentParser(description="EFM Validation Controls")
    parser.add_argument("--output_dir", type=str, default="validation_phase")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer t5-small...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    print("Loading dataset wikitext2...")
    dataset = EmbeddingDataset.from_text(
        dataset_name="wikitext2",
        encoder_name="t5-small",
        seq_length=64,
        max_samples=10000,
    )

    conditions = [
        {
            "name": "global_time_only",
            "time_mode": "none",
            "local_time_training": "none",
            "local_time_gen": "none"
        },
        {
            "name": "continuous_local_time",
            "time_mode": "continuous",
            "local_time_training": "expansion",
            "local_time_gen": "expansion"
        },
        {
            "name": "constant_0",
            "time_mode": "continuous",
            "local_time_training": "constant_0",
            "local_time_gen": "constant_0"
        },
        {
            "name": "shuffled_local_time",
            "time_mode": "continuous",
            "local_time_training": "shuffled",
            "local_time_gen": "shuffled"
        },
        {
            "name": "broadcast_baseline",
            "time_mode": "continuous",
            "local_time_training": "broadcast",
            "local_time_gen": "broadcast"
        }
    ]

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Starting condition: {condition['name']}")
        print(f"{'='*60}")

        model = EFM_Small(
            local_time_mode=condition["time_mode"],
            local_time_K=16,
            insertion_enabled=True,
            max_length=128,
        ).to(device)

        # Check if already trained
        expected_final_ckpt = os.path.join(args.checkpoint_dir, condition['name'], condition['name'], f"checkpoint_{args.steps}.pt")
        if os.path.exists(expected_final_ckpt):
            print(f"[{condition['name']}] Found existing checkpoint at {expected_final_ckpt}. Skipping training.")
            ckpt_dir_returned = os.path.dirname(expected_final_ckpt)
        else:
            print(f"[{condition['name']}] Training model...")
            ckpt_dir_returned = train(
                model=model,
                dataset=dataset,
                device=device,
                max_steps=args.steps,
                batch_size=8,
                checkpoint_dir=os.path.join(args.checkpoint_dir, condition['name']),
                experiment_name=condition['name'],
                output_dir=args.output_dir,
                local_time_training=condition["local_time_training"]
            )
        
        # Load best/last checkpoint to ensure we evaluate the right model state
        ckpt_dir = ckpt_dir_returned
        all_ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if all_ckpts:
            last_ckpt = os.path.join(ckpt_dir, all_ckpts[-1])
            print(f"[{condition['name']}] Loading weights from {last_ckpt}...")
            checkpoint = torch.load(last_ckpt, map_location=device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        print(f"[{condition['name']}] Generating samples...")
        samples = generate_efm_samples(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples=512,
            batch_size=8,
            max_length=64,
            local_time_mode=condition["local_time_gen"],
            seed=args.seed
        )

        print(f"[{condition['name']}] Evaluating...")
        evaluate(
            texts=samples,
            experiment_name=condition['name'],
            output_dir=args.output_dir
        )
        
        print(f"[{condition['name']}] Completed.")
        print("-" * 60)

if __name__ == "__main__":
    main()
