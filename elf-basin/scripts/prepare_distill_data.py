#!/usr/bin/env python3
"""Phase 0: Prepare distillation training data.

Generates ODE trajectory pairs from the teacher ELF-B-de-en model.
Uses the EXACT same data pipeline as reproduce_deen.py (proper masks).

For each WMT14 De-En example and random seed, stores:
  - z_t values at 8 timesteps along the 64-step ODE trajectory
  - x_pred (predicted x_0) at each stored timestep
  - cond_seq, cond_seq_mask
  - The timestep values themselves
"""
import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from utils.sampling_utils import get_sampling_steps, restore_cond, _ode_step
from utils.encoder_utils import encode_text
from utils.data_utils import load_dataset_split, get_dataloader, get_pad_token_id
from configs.config import SamplingConfig
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-de-en"
NUM_SAMPLES = 3000
BATCH_SIZE = 16
NUM_STEPS = 64
CFG = 2.0
SC_CFG = 1.0
TIME_SCHEDULE = "logit_normal"
NUM_SEEDS = 3           # generate 3 trajectories per example (different noise)
BASE_SEED = 42

# Store z and x_pred at these ODE step indices (out of 64)
STORE_STEP_INDICES = [0, 8, 16, 24, 32, 40, 48, 56]

OUT_DIR = _SCRIPT_DIR.parent / "runs" / "distill_data"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Preparing Distillation Training Data")
    print("=" * 70)

    # ── Load model ─────────────────────────────────────────────────────
    wrapper = ELFWrapper(CHECKPOINT, device=device)
    model = wrapper.model
    config = wrapper.config
    d_model = wrapper.d_model
    tokenizer = wrapper.tokenizer
    encoder = wrapper.encoder

    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    param_dtype = next(model.parameters()).dtype

    # ── Load dataset (same pipeline as reproduce_deen.py) ──────────────
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    EVAL_DATA = "embedded-language-flows/wmt14_de-en_validation_t5"
    local_dir = snapshot_download(repo_id=EVAL_DATA, repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    actual_samples = min(NUM_SAMPLES, len(eval_dataset))
    print(f"Dataset: {len(eval_dataset)} examples, using {actual_samples}")

    dataloader = get_dataloader(
        eval_dataset[:actual_samples], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    # ── Pre-allocate memmaps ───────────────────────────────────────────
    total_pairs = actual_samples * NUM_SEEDS
    n_stored = len(STORE_STEP_INDICES)

    # For each stored step, save z_t and x_pred
    # Shape: (total_pairs, n_stored_steps, seq_len, d_model) — too big!
    # Instead store per-step files
    for si in STORE_STEP_INDICES:
        step_dir = OUT_DIR / f"step_{si}"
        step_dir.mkdir(exist_ok=True)

    # We'll accumulate in lists and save at the end
    # To avoid OOM, save incrementally per seed pass

    print(f"\nGenerating trajectories for {NUM_SEEDS} seeds × {actual_samples} examples")
    print(f"Storing z_t and x_pred at step indices: {STORE_STEP_INDICES}")

    # ── Estimate disk usage ────────────────────────────────────────────
    # Per step: total_pairs × 128 × 512 × 2 (fp16) ≈ total_pairs × 128KB
    bytes_per_step = total_pairs * config.max_length * d_model * 2  # fp16
    total_bytes = bytes_per_step * n_stored * 2  # z_t and x_pred
    total_gb = total_bytes / (1024 ** 3)
    print(f"Estimated disk usage: {total_gb:.2f} GB")
    if total_gb > 30:
        print("ERROR: Disk usage exceeds 30 GB limit. Reduce NUM_SEEDS or STORE_STEP_INDICES.")
        return 1

    # Pre-allocate all memmaps
    z_maps = {}
    x_maps = {}
    for si in STORE_STEP_INDICES:
        step_dir = OUT_DIR / f"step_{si}"
        z_maps[si] = np.lib.format.open_memmap(
            str(step_dir / "z_t.npy"), mode='w+',
            dtype=np.float16, shape=(total_pairs, config.max_length, d_model)
        )
        x_maps[si] = np.lib.format.open_memmap(
            str(step_dir / "x_pred.npy"), mode='w+',
            dtype=np.float16, shape=(total_pairs, config.max_length, d_model)
        )

    # Also store the cond_seq and cond_seq_mask (one per example, shared across seeds)
    cond_seq_map = np.lib.format.open_memmap(
        str(OUT_DIR / "cond_seq.npy"), mode='w+',
        dtype=np.float16, shape=(actual_samples, config.max_length, d_model)
    )
    cond_mask_map = np.lib.format.open_memmap(
        str(OUT_DIR / "cond_seq_mask.npy"), mode='w+',
        dtype=np.float16, shape=(actual_samples, config.max_length)
    )

    # ── Generate trajectories ──────────────────────────────────────────
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    step_kwargs_base = dict(
        model=model, config=config,
        cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
    )

    for seed_idx in range(NUM_SEEDS):
        seed = BASE_SEED + seed_idx
        print(f"\n--- Seed {seed} ({seed_idx+1}/{NUM_SEEDS}) ---")

        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        samples_done = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Seed {seed}")):
            if samples_done >= actual_samples:
                break

            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attention_mask = torch.from_numpy(
                np.array(batch["encoder_attention_mask"])).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(
                np.array(batch["cond_seq_mask"])).to(device).float()

            # Encode source text (same as reproduce_deen.py)
            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attention_mask,
                encoder=encoder, latent_mean=config.latent_mean,
                latent_std=config.latent_std,
            ).to(param_dtype)

            # Store cond_seq and mask (only on first seed pass)
            if seed_idx == 0:
                cond_seq_map[samples_done:samples_done+bsz] = cond_seq.cpu().numpy().astype(np.float16)
                cond_mask_map[samples_done:samples_done+bsz] = cond_seq_mask_arr.cpu().numpy().astype(np.float16)

            # Get timesteps for this batch
            t_steps = get_sampling_steps(
                n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )

            # Initial noise
            z = (torch.randn((bsz, config.max_length, d_model),
                             generator=generator, dtype=param_dtype, device=device)
                 * config.denoiser_noise_scale)

            z = restore_cond(z, cond_seq, cond_seq_mask_arr)
            x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask_arr)

            # Run 64-step ODE, storing z and x_pred at target steps
            n = t_steps.shape[0]
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                for i in range(n - 1):
                    t_val = t_steps[i].item()
                    t_next = t_steps[i + 1].item()

                    # Store BEFORE this step if it's a target step
                    if i in STORE_STEP_INDICES:
                        pair_start = seed_idx * actual_samples + samples_done
                        pair_end = pair_start + bsz
                        z_maps[i][pair_start:pair_end] = z.float().cpu().numpy().astype(np.float16)
                        x_maps[i][pair_start:pair_end] = x_pred.float().cpu().numpy().astype(np.float16)

                    # ODE step
                    if i < n - 2:
                        z, x_pred = _ode_step(
                            z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
                            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                            **step_kwargs_base
                        )
                    else:
                        # Last step always ODE
                        z, x_pred = _ode_step(
                            z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
                            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                            **step_kwargs_base
                        )

            samples_done += bsz

    # ── Store timestep schedule for reference ──────────────────────────
    # Generate one more set of timesteps for metadata
    t_steps_ref = get_sampling_steps(
        n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device="cpu", dtype=torch.float32,
    )

    metadata = {
        "num_samples": actual_samples,
        "num_seeds": NUM_SEEDS,
        "total_pairs": total_pairs,
        "num_steps": NUM_STEPS,
        "store_step_indices": STORE_STEP_INDICES,
        "cfg": CFG,
        "sc_cfg": SC_CFG,
        "time_schedule": TIME_SCHEDULE,
        "seq_len": config.max_length,
        "d_model": d_model,
        "base_seed": BASE_SEED,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Done. Saved {total_pairs} trajectory pairs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
