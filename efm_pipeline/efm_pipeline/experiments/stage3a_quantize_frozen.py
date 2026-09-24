#!/usr/bin/env python3
"""Stage 3a — Frozen inference-time quantization sweep.

Takes the trained continuous-time EFM checkpoint (from Stage 1) and at
inference only, quantizes τ_i → floor(τ_i × K) for K ∈ {1,2,4,8,16,64}.
No retraining — tests how robust the model is to time compression.

Usage:
    python -m efm_pipeline.experiments.stage3a_quantize_frozen \
        --checkpoint checkpoints/stage1_continuous_baseline
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
_ELF_SRC = os.path.normpath(os.path.join(_PROJ_ROOT, "..", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)

import torch
from transformers import AutoTokenizer

from efm_pipeline.efm_model import EFM_Tiny, EFM_Small
from efm_pipeline.eval_runner import generate_efm_samples, evaluate
from efm_pipeline.utils import set_seed, get_device, device_summary, ensure_dir
from efm_pipeline.logger import ExperimentLogger


def quantize_local_times(tau: torch.Tensor, K: int) -> torch.Tensor:
    """Quantize continuous local times to K bins.

    τ_i → floor(τ_i × K) / K + 1/(2K)  (bin centre)
    """
    if K <= 0:
        return tau
    bin_ids = (tau * K).long().clamp(0, K - 1)
    return (bin_ids.float() + 0.5) / K


class QuantizedWrapper(torch.nn.Module):
    """Wraps an EFM model to quantize local times at inference.

    The backbone is frozen — only the quantization is applied.
    """

    def __init__(self, model, K: int):
        super().__init__()
        self.model = model
        self.K = K
        # Expose needed attributes.
        self.text_encoder_dim = model.text_encoder_dim
        self.vocab_size = model.vocab_size
        self.proj_kernel = model.proj_kernel
        self.proj_bias = model.proj_bias
        self.unembed_kernel = model.unembed_kernel
        self.unembed_bias = model.unembed_bias

    def forward(self, x, t, local_times=None, **kwargs):
        if local_times is not None:
            local_times = quantize_local_times(local_times, self.K)
        return self.model(x, t, local_times=local_times, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Stage 3a: Frozen quantization sweep")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small"])
    parser.add_argument("--K_values", type=str, default="1,2,4,8,16,64",
                        help="Comma-separated K values to test.")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device_summary(device)}")

    K_values = [int(k) for k in args.K_values.split(",")]
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Load checkpoint and auto-detect architecture.
    ckpt_dir = args.checkpoint
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1]) if ckpts else ckpt_dir
    else:
        ckpt_path = ckpt_dir

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    ckpt_hidden = state["blocks.0.norm1.weight"].shape[0]
    ckpt_depth = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
    has_insertion = any("insertion_head" in k for k in state)

    if ckpt_hidden == 512 and ckpt_depth == 6:
        args.model_size = "small"
    elif ckpt_hidden == 256 and ckpt_depth == 4:
        args.model_size = "tiny"
    print(f"  Detected model_size='{args.model_size}' (hidden={ckpt_hidden}, depth={ckpt_depth})")

    factory = EFM_Tiny if args.model_size == "tiny" else EFM_Small
    base_model = factory(
        local_time_mode="continuous", max_length=args.seq_length * 2,
        insertion_enabled=has_insertion,
    )

    base_model.load_state_dict(state)
    base_model = base_model.to(device).eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    # Sweep over K values.
    all_results = {}
    for K in K_values:
        print(f"\n{'='*50}")
        print(f"  K = {K} (frozen quantization)")
        print(f"{'='*50}")

        set_seed(args.seed)  # Same seed for each K.

        # Wrap model with quantization.
        quant_model = QuantizedWrapper(base_model, K)

        # Generate.
        texts = generate_efm_samples(
            model=quant_model, tokenizer=tokenizer, device=device,
            num_samples=args.num_samples, num_steps=args.num_steps,
            batch_size=args.batch_size, max_length=args.seq_length,
            seed=args.seed,
        )

        # Evaluate.
        result = evaluate(
            texts=texts,
            experiment_name=f"stage3a_quant_frozen_K{K}",
            output_dir=args.output_dir,
        )
        result["K"] = K
        result["mode"] = "frozen_quantized"
        all_results[K] = result

    # Save summary.
    summary_path = os.path.join(ensure_dir(args.output_dir), "stage3a_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved → {summary_path}")

    # Print comparison table.
    print(f"\n{'='*60}")
    print(f"  Frozen Quantization Sweep Results")
    print(f"  {'K':>4} | {'Gen-PPL':>10} | {'Entropy':>10} | {'3-gram Rep':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for K in K_values:
        r = all_results.get(K, {})
        print(f"  {K:>4} | {r.get('gen_ppl', 0):>10.2f} | {r.get('unigram_entropy_bits', 0):>10.4f} | {r.get('trigram_repetition', 0):>10.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
