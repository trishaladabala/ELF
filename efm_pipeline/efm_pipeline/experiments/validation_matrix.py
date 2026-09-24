import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from efm_pipeline.efm_model import EFM_Small
from efm_pipeline.train_efm import train, EmbeddingDataset
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.utils import set_seed, get_device

def evaluate_and_save(model, tokenizer, device, mode, seed, name, out_dir):
    out_file = os.path.join(out_dir, f"{name}_seed{seed}_result.json")
    if os.path.exists(out_file):
        print(f"  [{name}] Already evaluated. Skipping.")
        return
    print(f"  [{name}] Generating samples...")
    samples = generate_efm_samples(
        model=model, tokenizer=tokenizer, device=device,
        num_samples=512, batch_size=8, max_length=64,
        local_time_mode=mode, seed=seed
    )
    print(f"  [{name}] Evaluating...")
    metrics = evaluate(samples, experiment_name=name if seed == 42 else f"{name}_s{seed}", output_dir=out_dir)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  [{name}] Gen-PPL: {metrics.get('gen_ppl', float('inf')):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("Please run without --dry-run. Matrix is run programmatically.")
        return

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    dataset = EmbeddingDataset(cache_path="data_cache/wikitext2_t5-small_len64.pt")
    
    seeds = [42, 137, 2024]
    base_dir = "/home/snu/trishal_sanjay_proj/ELF"
    out_dir = os.path.join(base_dir, "efm_pipeline/validation_phase")
    ckpt_base = os.path.join(base_dir, "efm_pipeline/checkpoints_val")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_base, exist_ok=True)

    for seed in seeds:
        print(f"\n=============================================")
        print(f"  SEED {seed}")
        print(f"=============================================")
        set_seed(seed)
        
        # 1. global_time_only
        name = "global_time_only"
        model_gt = EFM_Small(vocab_size=32128, local_time_mode="none").to(device)
        ckpt_path = os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}", name if seed == 42 else f"{name}_s{seed}", "checkpoint_50000.pt")
        if not os.path.exists(ckpt_path):
            print(f"Training {name} for 50k steps...")
            train(model_gt, dataset, device, max_steps=50000, batch_size=8, save_every=10000,
                  checkpoint_dir=os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}"), experiment_name=name if seed == 42 else f"{name}_s{seed}",
                  output_dir=out_dir, local_time_training="none")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_gt.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        evaluate_and_save(model_gt, tokenizer, device, "none", seed, name, out_dir)
        del model_gt

        # 2. continuous_local_time
        name = "continuous_local_time"
        model_cont = EFM_Small(vocab_size=32128, local_time_mode="continuous").to(device)
        ckpt_path = os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}", name if seed == 42 else f"{name}_s{seed}", "checkpoint_50000.pt")
        if not os.path.exists(ckpt_path):
            print(f"Training {name} for 50k steps...")
            train(model_cont, dataset, device, max_steps=50000, batch_size=8, save_every=10000,
                  checkpoint_dir=os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}"), experiment_name=name if seed == 42 else f"{name}_s{seed}",
                  output_dir=out_dir, local_time_training="expansion")
        cont_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_cont.load_state_dict(cont_checkpoint.get("model_state_dict", cont_checkpoint))
        evaluate_and_save(model_cont, tokenizer, device, "expansion", seed, name, out_dir)

        # 3. frozen K1, K2, K4 (evaluate continuous model with constant values or quantized logic)
        evaluate_and_save(model_cont, tokenizer, device, "constant_0", seed, "frozen_K1", out_dir)
        # Note: True frozen K>1 evaluating a continuous model requires 'quantized' logic without weights.
        # However, a continuous model hasn't learned the quantized embeddings, so evaluating it with 'quantized' local_time_mode will crash
        # because the continuous model uses _forward_continuous, not _forward_quantized!
        # The correct way to "freeze K>1" on a continuous model is to quantize the tau inputs themselves but still pass to _forward_continuous!
        # But our local_time.py doesn't have a mode for "continuous_but_quantized_tau". 
        # So we skip frozen K>1 and go straight to the learned fine-tunings, which is the actual hypothesis.
        del model_cont

        # 4. Fine-tunings (learned_quantized and learned_lowrank)
        for K in [2, 4]:
            for mode in ["quantized", "lowrank"]:
                name = f"learned_{mode}_K{K}"
                print(f"Fine-tuning {name}...")
                model = EFM_Small(vocab_size=32128, local_time_mode=mode, local_time_K=K, lowrank_basis="learned" if mode=="lowrank" else None).to(device)
                # Load continuous weights, strict=False because the local_time_conditioner parameters change
                model.load_state_dict(cont_checkpoint.get("model_state_dict", cont_checkpoint), strict=False)
                
                ckpt_path = os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}", name if seed == 42 else f"{name}_s{seed}", "checkpoint_5000.pt")
                if not os.path.exists(ckpt_path):
                    train(model, dataset, device, max_steps=5000, batch_size=8, save_every=5000,
                          checkpoint_dir=os.path.join(ckpt_base, name if seed == 42 else f"{name}_s{seed}"), experiment_name=name if seed == 42 else f"{name}_s{seed}",
                          output_dir=out_dir, local_time_training="expansion")
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                evaluate_and_save(model, tokenizer, device, mode, seed, name, out_dir)
                del model

if __name__ == "__main__":
    main()
