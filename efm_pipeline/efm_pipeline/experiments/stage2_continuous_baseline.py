#!/usr/bin/env python3
"""Stage 2 — Evaluate the trained continuous-time EFM model.

Loads the Stage 1 checkpoint, generates samples, and computes metrics
using the standardised evaluation protocol (eval_runner.py).

Usage:
    python -m efm_pipeline.experiments.stage2_continuous_baseline \
        --checkpoint checkpoints/stage1_continuous_baseline
"""

import argparse
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
from efm_pipeline.utils import set_seed, get_device, device_summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Evaluate continuous EFM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small"])
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--insertion_enabled", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device_summary(device)}")

    # Find latest checkpoint.
    ckpt_dir = args.checkpoint
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        if ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    else:
        ckpt_path = ckpt_dir

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Auto-detect model size from checkpoint if user passed default.
    state = ckpt["model_state_dict"]
    ckpt_hidden = state["blocks.0.norm1.weight"].shape[0]
    ckpt_depth = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
    has_insertion = any("insertion_head" in k for k in state)
    if ckpt_hidden == 512 and ckpt_depth == 6:
        detected_size = "small"
    elif ckpt_hidden == 256 and ckpt_depth == 4:
        detected_size = "tiny"
    else:
        detected_size = args.model_size
    if detected_size != args.model_size:
        print(f"  Auto-detected model_size='{detected_size}' from checkpoint (overriding '{args.model_size}')")
        args.model_size = detected_size

    # Build model matching checkpoint architecture.
    factory = EFM_Tiny if args.model_size == "tiny" else EFM_Small
    model = factory(
        local_time_mode="continuous",
        max_length=args.seq_length * 2,
        insertion_enabled=has_insertion,
    )

    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"  Loaded from step {ckpt.get('step', '?')}")

    # Tokenizer for decoding.
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Generate samples.
    print(f"\nGenerating {args.num_samples} samples...")
    texts = generate_efm_samples(
        model=model, tokenizer=tokenizer, device=device,
        num_samples=args.num_samples, num_steps=args.num_steps,
        batch_size=args.batch_size, max_length=args.seq_length,
        seed=args.seed,
    )

    # Print samples.
    print("\n--- Samples ---")
    for i, text in enumerate(texts[:5]):
        print(f"  [{i}] {text[:200]}{'...' if len(text) > 200 else ''}")

    # Evaluate.
    evaluate(
        texts=texts,
        experiment_name="stage2_continuous_eval",
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
