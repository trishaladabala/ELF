#!/usr/bin/env python3
"""Stage 3b — Learned quantized conditioning.

Freezes the flow backbone, replaces continuous τ_i with a learned
nn.Embedding(K, hidden) table + adapter MLP. Fine-tunes only the
embedding and adapter while keeping the backbone frozen.

Usage:
    python -m efm_pipeline.experiments.stage3b_quantize_learned \
        --checkpoint checkpoints/stage1_continuous_baseline \
        --K_values 1,2,4,8,16 --finetune_steps 5000
"""

import argparse
import json
import os
import sys
from copy import deepcopy

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
_ELF_SRC = os.path.normpath(os.path.join(_PROJ_ROOT, "..", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.sampling_utils import add_noise, sample_timesteps

from efm_pipeline.efm_model import EFM_Tiny, EFM_Small
from efm_pipeline.local_time import LocalTimeConditioner
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.train_efm import EmbeddingDataset, EFMConfig
from efm_pipeline.utils import set_seed, get_device, device_summary, ensure_dir, amp_autocast
from efm_pipeline.logger import ExperimentLogger


def freeze_backbone(model):
    """Freeze all parameters except the local_time_conditioner."""
    for name, p in model.named_parameters():
        if "local_time_conditioner" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen backbone: {trainable:,} / {total:,} params trainable")


def replace_local_time_conditioner(model, K: int):
    """Replace the model's local-time conditioner with a quantized one."""
    model.local_time_conditioner = LocalTimeConditioner(
        hidden_size=model.hidden_size,
        mode="quantized",
        K=K,
    )
    return model


def finetune_quantized(
    model, dataset, device, K, steps, lr=5e-5, batch_size=4,
    experiment_name="stage3b", output_dir="results",
):
    """Fine-tune the quantized local-time conditioner with frozen backbone."""
    model.train()
    freeze_backbone(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    logger = ExperimentLogger(output_dir, f"{experiment_name}_K{K}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)
    efm_config = EFMConfig()

    for step in range(1, steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x0 = batch["embeddings"].to(device)
        target_ids = batch["token_ids"].to(device)
        B, S, D = x0.shape

        t = sample_timesteps(B, P_mean=0.8, P_std=0.8,
                             time_schedule="logit_normal", device=device)
        noise = torch.randn_like(x0)
        z = add_noise(x0, noise, t, efm_config)
        local_times = t.unsqueeze(1).expand(-1, S)

        optimizer.zero_grad()
        flow_out, decoder_logits = model(z, t, local_times=local_times, decoder_step_active=True)

        loss_flow = F.mse_loss(flow_out, x0)
        loss_decode = torch.tensor(0.0, device=device)
        if decoder_logits is not None:
            loss_decode = F.cross_entropy(
                decoder_logits.reshape(-1, model.vocab_size),
                target_ids.reshape(-1), ignore_index=0,
            )
        loss = loss_flow + 0.5 * loss_decode

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            logger.log_step({"step": step, "loss": loss.item(),
                             "flow": loss_flow.item(), "K": K})

        if not np.isfinite(loss.item()):
            print(f"  Non-finite loss at step {step}, stopping.")
            break

    logger.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 3b: Learned quantized conditioning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small"])
    parser.add_argument("--K_values", type=str, default="1,2,4,8,16")
    parser.add_argument("--finetune_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    K_values = [int(k) for k in args.K_values.split(",")]
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Load checkpoint and auto-detect architecture.
    ckpt_dir = args.checkpoint
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir
    else:
        ckpt_path = ckpt_dir
    base_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = base_ckpt["model_state_dict"]
    ckpt_hidden = state["blocks.0.norm1.weight"].shape[0]
    ckpt_depth = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
    has_insertion = any("insertion_head" in k for k in state)

    if ckpt_hidden == 512 and ckpt_depth == 6:
        args.model_size = "small"
    elif ckpt_hidden == 256 and ckpt_depth == 4:
        args.model_size = "tiny"
    print(f"  Detected model_size='{args.model_size}' (hidden={ckpt_hidden}, depth={ckpt_depth})")

    factory = EFM_Tiny if args.model_size == "tiny" else EFM_Small

    # Dataset for fine-tuning.
    dataset = EmbeddingDataset.from_text(
        dataset_name=args.dataset, seq_length=args.seq_length,
        max_samples=5000,
    )

    all_results = {}
    for K in K_values:
        print(f"\n{'='*50}")
        print(f"  K = {K} (learned quantized conditioning)")
        print(f"{'='*50}")

        set_seed(args.seed)

        # Build model, load backbone weights, replace local-time conditioner.
        model = factory(
            local_time_mode="continuous", max_length=args.seq_length * 2,
            insertion_enabled=has_insertion,
        )
        model.load_state_dict(state, strict=False)
        model = replace_local_time_conditioner(model, K)
        model = model.to(device)

        # Fine-tune.
        model = finetune_quantized(
            model, dataset, device, K,
            steps=args.finetune_steps, lr=args.learning_rate,
            batch_size=args.batch_size, output_dir=args.output_dir,
        )
        model.eval()

        # Generate and evaluate.
        texts = generate_efm_samples(
            model=model, tokenizer=tokenizer, device=device,
            num_samples=args.num_samples, num_steps=args.num_steps,
            batch_size=args.batch_size, max_length=args.seq_length,
            seed=args.seed,
        )
        result = evaluate(texts=texts, experiment_name=f"stage3b_quant_learned_K{K}",
                          output_dir=args.output_dir)
        result["K"] = K
        result["mode"] = "learned_quantized"
        result["finetune_steps"] = args.finetune_steps
        all_results[K] = result

    # Save summary.
    summary_path = os.path.join(ensure_dir(args.output_dir), "stage3b_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Learned Quantization Sweep Results")
    print(f"  {'K':>4} | {'Gen-PPL':>10} | {'Entropy':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}")
    for K in K_values:
        r = all_results.get(K, {})
        print(f"  {K:>4} | {r.get('gen_ppl', 0):>10.2f} | {r.get('unigram_entropy_bits', 0):>10.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
