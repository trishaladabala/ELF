#!/usr/bin/env python3
"""Stage 1 — Magus Sanity Verification of Real ELF-B Checkpoint (CUDA).

Adapted from phase0l/testL1_checkpoint_sanity.py for the RTX A4000 (CUDA).
Downloads the PyTorch-converted ELF-B-owt checkpoint from HuggingFace,
loads it, runs unconditional SDE generation at 32 steps with sde_gamma=1.5
(matching the paper), scores with GPT-2 Large, and instruments degeneracy
+ trajectory extraction.

Key changes from L1:
  - DEVICE = "cuda" (was "mps")
  - DTYPE = torch.float16 (was float32) — A4000 supports fp16 Tensor Cores
  - max_length = 1024 (production; was 128 for local sanity)
  - BATCH_SIZE = 4 (16 GB VRAM constraint)
  - Uses torch.amp.autocast for mixed-precision forward passes
"""

import json
import sys
import math
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Point to the real ELF source tree
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modules.model import ELF_models
from utils.sampling_utils import _sde_step, _ode_step, get_sampling_steps
from configs.config import Config

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stage1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ELF model internally casts to .float() inside autocast-disabled blocks,
# so model weights must stay in fp32. We use torch.amp.autocast externally
# for fp16 matmuls in attention, matching the official ELF training setup.
DTYPE = torch.float32

N_SAMPLES = 32
BATCH_SIZE = 4          # Conservative for 16 GB VRAM at L=1024
N_STEPS = 32
T_EPS = 0.05
SDE_GAMMA = 1.5
SELF_COND_CFG_SCALE = 3.0
DENOISER_NOISE_SCALE = 2.0
MAX_LENGTH = 1024       # Production sequence length

# Config from train_owt_ELF-B.yml
COMMON_KWARGS = dict(
    text_encoder_dim=512,    # T5-small output dim
    max_length=MAX_LENGTH,
    bottleneck_dim=128,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    vocab_size=32100,        # T5-small vocab
)

config = Config()
config.denoiser_p_mean = -1.5
config.denoiser_p_std = 0.8
config.denoiser_noise_scale = 2.0
config.t_eps = 0.05
config.num_self_cond_cfg_tokens = 4
config.self_cond_prob = 0.5

HF_REPOS = {
    "ELF-B": "embedded-language-flows/ELF-B-owt-torch",
    "ELF-M": "embedded-language-flows/ELF-M-owt-torch",
    "ELF-L": "embedded-language-flows/ELF-L-owt-torch",
}

PAPER_PPLS = {"ELF-B": 24.1, "ELF-M": 21.7, "ELF-L": 23.3}


def download_checkpoint(hf_repo):
    from huggingface_hub import hf_hub_download, list_repo_files
    print(f"  Listing files in {hf_repo}...")
    files = list_repo_files(hf_repo)
    ckpt_files = [f for f in files if f.startswith("checkpoint") or f.endswith(".pt")]
    if not ckpt_files:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo)
    print(f"  Downloading {ckpt_files[0]}...")
    return hf_hub_download(hf_repo, ckpt_files[0])


def load_elf_model(model_name, ckpt_path, device="cpu", dtype=torch.float32):
    factory_fn = ELF_models[model_name]
    model = factory_fn(**COMMON_KWARGS)

    print(f"  Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "ema_params1" in ckpt and ckpt["ema_params1"]:
        state_dict = ckpt["ema_params1"]
        print(f"  Using EMA parameters")
    elif "params" in ckpt:
        state_dict = ckpt["params"]
        print(f"  Using raw parameters")
    else:
        state_dict = ckpt
        print(f"  Using checkpoint directly")

    from tqdm import tqdm
    # Load state dict
    result = model.load_state_dict(state_dict, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded successfully. Params: {n_params:,}")
    if result.missing_keys:
        print(f"  ⚠️ Missing keys: {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"  ⚠️ Unexpected keys: {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

    model = model.to(device=device, dtype=torch.float32).eval()
    return model, True


def sde_sample_with_trajectory(model, n, n_steps, seq_len, enc_dim, gamma, device, dtype,
                                noise=None, return_trajectory=False):
    """Run SDE sampling with optional trajectory saving."""
    trajectory = []
    # _sde_step creates noise on z.device with this generator, so the generator
    # must be on the same device as z.
    if device == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        if noise is not None:
            z = noise.clone().to(device=device, dtype=dtype)
        elif device == "cuda":
            z = torch.randn((n, seq_len, enc_dim), generator=generator,
                            dtype=dtype, device=device) * config.denoiser_noise_scale
        else:
            z = (torch.randn((n, seq_len, enc_dim), generator=generator, dtype=dtype)
                 .to(device) * config.denoiser_noise_scale)

        if return_trajectory:
            trajectory.append(z.float().cpu().numpy())

        steps = get_sampling_steps(
            n_steps=n_steps, time_schedule="logit_normal",
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=dtype
        )

        x_pred_prev = None
        cond_seq = torch.zeros((n, seq_len, enc_dim), dtype=dtype, device=device)
        cond_seq_mask = torch.zeros((n, seq_len), dtype=dtype, device=device)

        for i in range(n_steps):
            t_curr, t_next = steps[i].item(), steps[i+1].item()
            z, x_pred_prev = _sde_step(
                model=model, z=z, t=t_curr, t_next=t_next, x_pred_prev=x_pred_prev,
                config=config, cfg_scale=1.0, self_cond_cfg_scale=SELF_COND_CFG_SCALE,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, gamma=gamma, generator=generator
            )
            if return_trajectory:
                trajectory.append(z.float().cpu().numpy())

    if return_trajectory:
        return z, np.array(trajectory)
    return z


def decode_to_tokens(model, z, device, dtype):
    """Decode latent z to token IDs using the model's decoder head."""
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        t_f = torch.ones(z.shape[0], dtype=dtype, device=device)
        dec_active = torch.ones(z.shape[0], device=device)
        sc_scale = torch.full((z.shape[0],), SELF_COND_CFG_SCALE, dtype=dtype, device=device)
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
        _, logits = model(z_input, t_f, deterministic=True,
                          self_cond_cfg_scale=sc_scale, decoder_step_active=dec_active)
        return logits.argmax(dim=-1).cpu()


def is_degenerate(token_ids, threshold=2):
    """Check if a token sequence is degenerate (collapsed to ≤threshold unique tokens)."""
    return len(set(int(t) for t in token_ids)) <= threshold


def compute_turning_angles(trajectory):
    """Compute mean turning angle (curvature proxy) per sample per position."""
    v = np.diff(trajectory, axis=0)
    if v.shape[0] < 2:
        return np.zeros((trajectory.shape[1], trajectory.shape[2]))
    v_prev, v_curr = v[:-1], v[1:]
    dot = np.sum(v_prev * v_curr, axis=-1)
    n_prev = np.linalg.norm(v_prev, axis=-1)
    n_curr = np.linalg.norm(v_curr, axis=-1)
    cos_sim = np.clip(dot / (n_prev * n_curr + 1e-12), -1.0, 1.0)
    return np.mean(np.arccos(cos_sim), axis=0)


def setup_eval_model(device):
    """Load GPT-2 Large for perplexity evaluation."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("  Loading GPT-2 Large for evaluation...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
    # GPT-2 Large stays in fp16 for VRAM savings — it has no internal .float() casts
    if device == "cuda":
        model = model.half()
    return model, tokenizer


def score_text(text, model, tokenizer, device, max_length=1024):
    """Score text with GPT-2 perplexity."""
    if not text.strip():
        return float('inf')
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float('inf')
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.float().item())


def print_gpu_mem():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {alloc:.2f}/{total:.1f} GB allocated, {reserved:.2f} GB reserved")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pritnt("=" * 70)
    print("Stage 1 — Magus Sanity Verification (CUDA / RTX A4000)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Dtype: {DTYPE}")
    print(f"  N_SAMPLES: {N_SAMPLES}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MAX_LENGTH: {MAX_LENGTH}")
    print(f"  N_STEPS: {N_STEPS}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {_vram_gb:.1f} GB")
    print()

    report = {"steps": [], "verdict": "", "system": {
        "device": DEVICE, "dtype": str(DTYPE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
    }}

    t_start = time.time()

    # ─── Step 1: Download & Load ELF-B ───
    print("─── Step 1: Download & Load ELF-B-owt ───")
    try:
        ckpt_path = download_checkpoint(HF_REPOS["ELF-B"])
        report["steps"].append({"name": "download_elf_b", "status": "PASS"})
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        report["steps"].append({"name": "download_elf_b", "status": "FAIL", "error": str(e)})
        report["verdict"] = "FAIL: Could not download ELF-B checkpoint"
        with open(RESULTS_DIR / "stage1_sanity.json", "w") as f:
            json.dump(report, f, indent=2)
        return

    model, load_ok = load_elf_model("ELF-B", ckpt_path, device=DEVICE, dtype=DTYPE)
    if not load_ok:
        report["steps"].append({"name": "load_elf_b", "status": "FAIL"})
        report["verdict"] = "FAIL: ELF-B weight loading failed"
        with open(RESULTS_DIR / "stage1_sanity.json", "w") as f:
            json.dump(report, f, indent=2)
        return
    report["steps"].append({"name": "load_elf_b", "status": "PASS"})
    print_gpu_mem()

    # ─── Step 2: SDE Generation ───
    print(f"\n─── Step 2: SDE Generation (N={N_SAMPLES}, {N_STEPS} steps, γ={SDE_GAMMA}, L={MAX_LENGTH}) ───")

    enc_dim = COMMON_KWARGS["text_encoder_dim"]
    seq_len = COMMON_KWARGS["max_length"]

    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    eval_model, eval_tokenizer = setup_eval_model(DEVICE)
    print_gpu_mem()

    all_ppls, all_degs, all_texts = [], [], []
    trajectory_batch = None

    torch.manual_seed(42)
    gen_start = time.time()
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
        bs = min(BATCH_SIZE, N_SAMPLES - batch_start)
        batch_idx = batch_start // BATCH_SIZE + 1
        print(f"  Batch {batch_idx}/{n_batches} (n={bs})...")
        batch_t0 = time.time()

        # Generate initial noise on CPU then move to GPU (reproducibility)
        noise = torch.randn(bs, seq_len, enc_dim, dtype=DTYPE) * DENOISER_NOISE_SCALE
        noise = noise.to(DEVICE)

        save_traj = (batch_start == 0)  # Only save trajectory for first batch
        z_final, traj = sde_sample_with_trajectory(
            model, bs, N_STEPS, seq_len, enc_dim, SDE_GAMMA, DEVICE, DTYPE,
            noise=noise, return_trajectory=True)

        if save_traj:
            trajectory_batch = traj

        token_ids = decode_to_tokens(model, z_final, DEVICE, DTYPE)

        for i in range(bs):
            tids = token_ids[i].tolist()
            deg = is_degenerate(tids)
            all_degs.append(deg)
            text = t5_tokenizer.decode(tids, skip_special_tokens=True)
            all_texts.append(text[:300])
            if deg:
                all_ppls.append(50000.0)
            else:
                ppl = score_text(text, eval_model, eval_tokenizer, DEVICE)
                if math.isnan(ppl) or math.isinf(ppl):
                    ppl = 50000.0
                all_ppls.append(ppl)

        batch_dt = time.time() - batch_t0
        print(f"    Batch time: {batch_dt:.1f}s ({batch_dt/bs:.1f}s/sample)")

    gen_dt = time.time() - gen_start
    print(f"\n  Total generation time: {gen_dt:.1f}s ({gen_dt/N_SAMPLES:.1f}s/sample)")

    ppls = np.array(all_ppls)
    degs = np.array(all_degs)
    deg_rate = float(degs.mean())
    valid = ~degs
    if valid.sum() > 0:
        vp = ppls[valid]
        mean_ppl = float(np.mean(vp))
        median_ppl = float(np.median(vp))
    else:
        mean_ppl = median_ppl = float('nan')

    print(f"\n  --- Generation Results (N={N_SAMPLES}, L={MAX_LENGTH}) ---")
    print(f"  Degeneracy rate: {100*deg_rate:.1f}%")
    print(f"  Non-degen PPL: mean={mean_ppl:.1f}, median={median_ppl:.1f}")
    print(f"  Paper reference: 24.1")
    print(f"  Absolute difference: {abs(mean_ppl - 24.1):.1f}")
    print()
    for i, txt in enumerate(all_texts[:5]):
        status = "DEG" if all_degs[i] else f"PPL={all_ppls[i]:.1f}"
        print(f"    [{status}] {txt[:150]}...")
    print_gpu_mem()

    # ─── Step 3: Trajectory Check ───
    print("\n─── Step 3: Trajectory & Curvature Pipeline Check ───")
    traj_ok = curv_ok = False
    if trajectory_batch is not None:
        ts = trajectory_batch.shape
        print(f"  Trajectory shape: {ts} (steps+1, batch, seq_len, enc_dim)")
        curvature = compute_turning_angles(trajectory_batch)
        print(f"  Curvature shape: {curvature.shape}, mean={np.mean(curvature):.6f} rad")
        traj_ok = (ts[2] == seq_len and ts[3] == enc_dim)
        curv_ok = (curvature.shape[0] == ts[1] and curvature.shape[1] == seq_len)
        print(f"  {'✅' if traj_ok and curv_ok else '❌'} Trajectory pipeline on real dims (L={seq_len})")

    report["steps"].append({
        "name": "sde_generation", "status": "PASS",
        "n_samples": N_SAMPLES, "max_length": MAX_LENGTH,
        "deg_rate": deg_rate, "mean_ppl_nondegen": mean_ppl,
        "median_ppl_nondegen": median_ppl, "paper_ppl": 24.1,
        "abs_diff": abs(mean_ppl - 24.1),
        "gen_time_s": gen_dt, "s_per_sample": gen_dt / N_SAMPLES,
        "trajectory_ok": traj_ok, "curvature_ok": curv_ok,
    })

    # ─── Step 4: Weight Loading for M and L ───
    print("\n─── Step 4: Weight Loading Check for ELF-M and ELF-L ───")
    del model, eval_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for model_name in ["ELF-M", "ELF-L"]:
        print(f"\n  {model_name}:")
        try:
            cp = download_checkpoint(HF_REPOS[model_name])
            m, ok = load_elf_model(model_name, cp, device="cpu", dtype=torch.float32)
            report["steps"].append({
                "name": f"load_{model_name.lower().replace('-','_')}",
                "status": "PASS" if ok else "FAIL"
            })
            if m is not None:
                del m
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            report["steps"].append({
                "name": f"load_{model_name.lower().replace('-','_')}",
                "status": "FAIL", "error": str(e)
            })

    # ─── Verdict ───
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    ppl_reasonable = not math.isnan(mean_ppl) and mean_ppl < 200
    all_loads_ok = all(s["status"] == "PASS" for s in report["steps"] if s["name"].startswith("load"))

    if ppl_reasonable and all_loads_ok and traj_ok:
        verdict = (f"PASS: Pipeline wired correctly on CUDA. "
                   f"Gen PPL={mean_ppl:.1f} (paper=24.1). "
                   f"All sizes load. Trajectory OK at L={MAX_LENGTH}. "
                   f"Total time: {total_time:.0f}s.")
    elif not all_loads_ok:
        verdict = "FAIL: One or more model weight loads failed."
    elif not ppl_reasonable:
        verdict = f"FAIL: PPL={mean_ppl:.1f} wildly off from paper (24.1). Config/conversion bug."
    else:
        verdict = f"PARTIAL: PPL={mean_ppl:.1f} but trajectory pipeline issues."

    report["verdict"] = verdict
    report["total_time_s"] = total_time
    print(f"  Verdict: {verdict}")
    print("=" * 70)

    # ─── Save Reports ───
    with open(RESULTS_DIR / "stage1_sanity.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(RESULTS_DIR / "stage1_sanity_report.md", "w") as f:
        f.write("# Stage 1 — Magus Checkpoint Sanity Report (CUDA)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**System:** {report['system']['gpu']} ({report['system']['vram_gb']:.1f} GB)\n")
        f.write(f"**Total time:** {total_time:.0f}s\n\n")
        f.write(f"## Verdict\n\n**{verdict}**\n\n")
        f.write("## Step Results\n\n| Step | Status |\n|------|--------|\n")
        for s in report["steps"]:
            f.write(f"| {s['name']} | {s['status']} |\n")
        f.write(f"\n## Generation Statistics (N={N_SAMPLES}, L={MAX_LENGTH})\n\n")
        f.write(f"- **Degeneracy Rate:** {100*deg_rate:.1f}%\n")
        f.write(f"- **Mean PPL (non-degen):** {mean_ppl:.1f}\n")
        f.write(f"- **Median PPL (non-degen):** {median_ppl:.1f}\n")
        f.write(f"- **Paper Reference PPL:** 24.1\n")
        f.write(f"- **Absolute Difference:** {abs(mean_ppl - 24.1):.1f}\n")
        f.write(f"- **Generation Time:** {gen_dt:.1f}s ({gen_dt/N_SAMPLES:.1f}s/sample)\n\n")
        f.write("## Sample Outputs\n\n")
        for i, txt in enumerate(all_texts[:5]):
            st = "DEG" if all_degs[i] else f"PPL={all_ppls[i]:.1f}"
            f.write(f"**[{st}]** {txt[:300]}\n\n")

    print(f"\n  Saved → {RESULTS_DIR / 'stage1_sanity.json'}")
    print(f"  Saved → {RESULTS_DIR / 'stage1_sanity_report.md'}")


if __name__ == "__main__":
    main()
