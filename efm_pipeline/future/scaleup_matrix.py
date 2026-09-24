#!/usr/bin/env python3
"""Scale-up experiment: ELF-B scale (105M params), 200k steps.

Tests whether the compression benefit holds at scale.
Sweeps K ∈ {2, 4, 8, 16} for both quantized and low-rank modes.

Usage:
    python -m efm_pipeline.experiments.scaleup_matrix
"""
import os
import sys
import json
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from efm_pipeline.efm_model import EFM_Base
from efm_pipeline.train_efm import train, EmbeddingDataset
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.utils import set_seed, get_device, ensure_dir


def evaluate_and_save(model, tokenizer, device, mode, name, out_dir):
    """Generate samples and evaluate, saving results to JSON."""
    out_file = os.path.join(out_dir, f"{name}_result.json")
    if os.path.exists(out_file):
        print(f"  [{name}] Already evaluated. Skipping.")
        data = json.load(open(out_file))
        print(f"  [{name}] Gen-PPL: {data.get('gen_ppl', 'N/A')}")
        return data
    print(f"  [{name}] Generating 512 samples...")
    samples = generate_efm_samples(
        model=model, tokenizer=tokenizer, device=device,
        num_samples=512, batch_size=16, max_length=128,
        local_time_mode=mode, seed=42
    )
    print(f"  [{name}] Evaluating...")
    metrics = evaluate(samples, experiment_name=name, output_dir=out_dir)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  [{name}] Gen-PPL: {metrics.get('gen_ppl', float('inf')):.2f}")
    return metrics


def find_checkpoint(ckpt_dir, exp_name, max_steps):
    """Find the final checkpoint for a given experiment."""
    # Try the nested directory structure from train()
    nested = os.path.join(ckpt_dir, exp_name, f"checkpoint_{max_steps}.pt")
    if os.path.exists(nested):
        return nested
    # Try flat structure
    flat = os.path.join(ckpt_dir, f"checkpoint_{max_steps}.pt")
    if os.path.exists(flat):
        return flat
    return None


def load_model_checkpoint(model, ckpt_path, device):
    """Load checkpoint, handling both dict and raw state_dict formats."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return checkpoint


def train_condition(name, model, dataset, device, ckpt_base, out_dir,
                    max_steps, local_time_training, batch_size=16):
    """Train a model if checkpoint doesn't exist, return checkpoint path."""
    ckpt_dir = os.path.join(ckpt_base, name)
    ckpt_path = find_checkpoint(ckpt_dir, name, max_steps)

    if ckpt_path is not None:
        print(f"  [{name}] Found checkpoint at {ckpt_path}. Skipping training.")
    else:
        print(f"  [{name}] Training for {max_steps} steps...")
        train(
            model=model,
            dataset=dataset,
            device=device,
            max_steps=max_steps,
            batch_size=batch_size,
            save_every=max(max_steps // 4, 5000),
            log_every=500,
            checkpoint_dir=ckpt_dir,
            experiment_name=name,
            output_dir=out_dir,
            local_time_training=local_time_training,
            use_amp=True,
        )
        ckpt_path = find_checkpoint(ckpt_dir, name, max_steps)

    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found for {name} after training")
    return ckpt_path


def main():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    dataset = EmbeddingDataset(cache_path="data_cache/wikitext2_t5-small_len64.pt")

    out_dir = "scaleup_results"
    ckpt_base = "checkpoints_scaleup"
    ensure_dir(out_dir)
    ensure_dir(ckpt_base)

    MAX_STEPS = 200_000
    FINETUNE_STEPS = 10_000
    BATCH_SIZE = 16  # Fits comfortably on A4000 with grad checkpointing
    K_VALUES = [2, 4, 8, 16]

    set_seed(42)

    print("=" * 60)
    print("  EFM SCALE-UP EXPERIMENT")
    print(f"  Model: EFM_Base (104M params)")
    print(f"  Training: {MAX_STEPS} steps, batch_size={BATCH_SIZE}")
    print(f"  Fine-tuning: {FINETUNE_STEPS} steps")
    print(f"  K values: {K_VALUES}")
    print("=" * 60)

    # =========================================================================
    # 1. BASELINE: Global Time Only (no local time conditioning)
    # =========================================================================
    print("\n\n### 1. Global Time Only Baseline ###")
    name = "base_global_time_only"
    model_gt = EFM_Base(vocab_size=32128, local_time_mode="none").to(device)
    ckpt_path = train_condition(
        name, model_gt, dataset, device, ckpt_base, out_dir,
        MAX_STEPS, "none", BATCH_SIZE
    )
    load_model_checkpoint(model_gt, ckpt_path, device)
    model_gt.eval()
    evaluate_and_save(model_gt, tokenizer, device, "none", name, out_dir)
    del model_gt
    torch.cuda.empty_cache()

    # =========================================================================
    # 2. CONTINUOUS: Full continuous local time (the oracle)
    # =========================================================================
    print("\n\n### 2. Continuous Local Time ###")
    name = "base_continuous_local_time"
    model_cont = EFM_Base(vocab_size=32128, local_time_mode="continuous").to(device)
    cont_ckpt_path = train_condition(
        name, model_cont, dataset, device, ckpt_base, out_dir,
        MAX_STEPS, "expansion", BATCH_SIZE
    )
    cont_checkpoint = load_model_checkpoint(model_cont, cont_ckpt_path, device)
    model_cont.eval()
    evaluate_and_save(model_cont, tokenizer, device, "expansion", name, out_dir)

    # =========================================================================
    # 3. FROZEN K=1 (constant tau=0 on continuous model)
    # =========================================================================
    print("\n\n### 3. Frozen K=1 ###")
    evaluate_and_save(model_cont, tokenizer, device, "constant_0",
                      "base_frozen_K1", out_dir)
    del model_cont
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. K SWEEP: Learned Quantized K={2,4,8,16}
    # =========================================================================
    print("\n\n### 4. Learned Quantized K Sweep ###")
    for K in K_VALUES:
        name = f"base_learned_quantized_K{K}"
        print(f"\n--- {name} ---")
        model = EFM_Base(
            vocab_size=32128, local_time_mode="quantized", local_time_K=K
        ).to(device)
        # Load continuous weights, strict=False for changed conditioner params
        sd = torch.load(cont_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            model.load_state_dict(sd["model_state_dict"], strict=False)
        else:
            model.load_state_dict(sd, strict=False)

        ckpt_path = train_condition(
            name, model, dataset, device, ckpt_base, out_dir,
            FINETUNE_STEPS, "expansion", BATCH_SIZE
        )
        load_model_checkpoint(model, ckpt_path, device)
        model.eval()
        evaluate_and_save(model, tokenizer, device, "quantized", name, out_dir)
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # 5. K SWEEP: Learned Low-Rank K={2,4,8,16}
    # =========================================================================
    print("\n\n### 5. Learned Low-Rank K Sweep ###")
    for K in K_VALUES:
        name = f"base_learned_lowrank_K{K}"
        print(f"\n--- {name} ---")
        model = EFM_Base(
            vocab_size=32128, local_time_mode="lowrank",
            local_time_K=K, lowrank_basis="learned"
        ).to(device)
        sd = torch.load(cont_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            model.load_state_dict(sd["model_state_dict"], strict=False)
        else:
            model.load_state_dict(sd, strict=False)

        ckpt_path = train_condition(
            name, model, dataset, device, ckpt_base, out_dir,
            FINETUNE_STEPS, "expansion", BATCH_SIZE
        )
        load_model_checkpoint(model, ckpt_path, device)
        model.eval()
        evaluate_and_save(model, tokenizer, device, "lowrank", name, out_dir)
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "=" * 60)
    print("  SCALE-UP RESULTS SUMMARY")
    print("=" * 60)

    results = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith("_result.json"):
            with open(os.path.join(out_dir, f)) as fh:
                data = json.load(fh)
                name = f.replace("_result.json", "")
                results[name] = data

    print(f"\n{'Condition':<40} {'Gen-PPL':>10} {'Avg Words':>10} {'Empty%':>8}")
    print("-" * 70)
    for name, data in sorted(results.items()):
        ppl = data.get("gen_ppl", float("nan"))
        words = data.get("avg_words", 0)
        empty = data.get("empty_fraction", 0) * 100
        print(f"{name:<40} {ppl:>10.2f} {words:>10.1f} {empty:>7.1f}%")


if __name__ == "__main__":
    main()

