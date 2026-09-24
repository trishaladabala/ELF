#!/usr/bin/env python3
"""Phase 0 — Reproduce ELF-B-de-en published numbers.

Target: 64-step ODE, self-conditioning CFG=1, input-condition CFG=2,
        WMT14 De-En validation set.
        BLEU ≈ 26.4

Reports: BLEU (sacreBLEU), ROUGE-1/2/L, wall-clock, batch size, peak VRAM.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper, SimpleConfig
from modules.t5_encoder import get_encoder
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import (
    _generate_samples_single_batch, _dlm_decode_batch,
    mask_after_eos, shift_left,
)
from utils.encoder_utils import encode_text
from utils.data_utils import load_dataset_split, get_dataloader, get_pad_token_id
from utils.metrics_utils import compute_bleu, compute_rouge
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-de-en"
NUM_SAMPLES = 3000          # WMT14 De-En validation is ~3000 examples
BATCH_SIZE = 32             # L=128 is small; batch 32 fits easily
NUM_STEPS = 64
CFG = 2.0                   # Input-condition CFG scale
SC_CFG = 1.0                # Self-conditioning CFG scale
TIME_SCHEDULE = "logit_normal"
SEED = 42

# WMT14 De-En data
EVAL_DATA = "embedded-language-flows/wmt14_de-en_validation_t5"

RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "phase0_deen"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Load model ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  Phase 0: De-En Reproduction — Loading ELF-B-de-en")
    print("="*70)

    wrapper = ELFWrapper(CHECKPOINT, device=device)
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    config = wrapper.config

    # Add missing config fields for data utils compatibility
    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    # Use the frozen encoder from the wrapper
    encoder = wrapper.encoder

    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    print(f"\nLoading eval dataset: {EVAL_DATA}")
    local_dir = snapshot_download(repo_id=EVAL_DATA, repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    actual_samples = min(NUM_SAMPLES, len(eval_dataset))
    print(f"Dataset size: {len(eval_dataset)}, using {actual_samples} samples")

    dataloader = get_dataloader(
        eval_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    # ── Generate ───────────────────────────────────────────────────────
    print(f"\nGenerating with {NUM_STEPS}-step ODE, CFG={CFG}, SC-CFG={SC_CFG}")

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
    else:
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    d_model = wrapper.d_model
    param_dtype = next(model.parameters()).dtype

    all_generated = []  # (index, reference, hypothesis, source)
    gen_time_total = 0.0
    decode_time_total = 0.0
    samples_processed = 0

    from configs.config import SamplingConfig
    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    gen_start = time.time()
    pbar = tqdm(total=(actual_samples + BATCH_SIZE - 1) // BATCH_SIZE,
                desc="Generating (De-En)")

    for batch_idx, batch in enumerate(dataloader):
        if samples_processed >= actual_samples:
            break

        bsz = batch["input_ids"].shape[0]
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        encoder_attention_mask = torch.from_numpy(
            np.array(batch["encoder_attention_mask"])).to(device).float()
        cond_seq_mask_arr = torch.from_numpy(
            np.array(batch["cond_seq_mask"])).to(device).float()

        t_steps = get_sampling_steps(
            n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=param_dtype,
        )

        cond_seq = encode_text(
            input_ids=input_ids, attention_mask=encoder_attention_mask,
            encoder=encoder, latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        ).to(param_dtype)

        z = (torch.randn((bsz, config.max_length, d_model),
                          generator=generator, dtype=param_dtype, device=device)
             * config.denoiser_noise_scale)

        gen_batch_start = time.time()
        latent = _generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
            config=config, sampling_config=sampling_config,
            cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
        )
        gen_time_total += time.time() - gen_batch_start

        gen_length = config.max_length - config.max_input_length
        cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)

        dec_start = time.time()
        t_final_val = t_steps[-1].item()
        predicted_ids = _dlm_decode_batch(
            z=latent, model=model, t_final_val=t_final_val,
            config=config, self_cond_cfg_scale=SC_CFG,
        )
        predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :gen_length]
        predicted_ids = mask_after_eos(
            predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        decode_time_total += time.time() - dec_start

        for i in range(bsz):
            if samples_processed >= actual_samples:
                break
            text = tokenizer.decode(
                predicted_ids[i].detach().cpu().numpy(),
                skip_special_tokens=True)
            ref = batch["target"][i] if "target" in batch else ""
            src = batch["input"][i] if "input" in batch else ""
            all_generated.append((samples_processed, ref, text, src))
            samples_processed += 1

        pbar.update(1)
    pbar.close()

    total_gen_time = time.time() - gen_start
    print(f"\nGeneration complete: {len(all_generated)} samples in {total_gen_time:.1f}s")
    print(f"  Forward pass time: {gen_time_total:.1f}s, Decode time: {decode_time_total:.1f}s")

    # ── Score BLEU / ROUGE ─────────────────────────────────────────────
    hypotheses = [gen for _, _, gen, _ in all_generated]
    references = [ref for _, ref, _, _ in all_generated]

    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge(hypotheses, references)
    print(f"\n  BLEU:    {bleu:.2f}")
    print(f"  ROUGE-1: {rouge['rouge1']:.2f}")
    print(f"  ROUGE-2: {rouge['rouge2']:.2f}")
    print(f"  ROUGE-L: {rouge['rougeL']:.2f}")

    # ── Peak VRAM ──────────────────────────────────────────────────────
    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # ── Config hash ────────────────────────────────────────────────────
    config_str = json.dumps({
        "checkpoint": CHECKPOINT, "num_samples": actual_samples,
        "batch_size": BATCH_SIZE, "num_steps": NUM_STEPS,
        "cfg": CFG, "sc_cfg": SC_CFG,
        "time_schedule": TIME_SCHEDULE, "seed": SEED,
        "eval_data": EVAL_DATA,
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 0 — DE-EN REPRODUCTION RESULTS")
    print("="*70)
    print(f"  Checkpoint:       {CHECKPOINT} (step {wrapper.checkpoint_step})")
    print(f"  Config hash:      {config_hash}")
    print(f"  Seed:             {SEED}")
    print(f"  Samples:          {len(all_generated)}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Steps:            {NUM_STEPS}, ODE, CFG={CFG}, SC-CFG={SC_CFG}")
    print(f"  Gen. time:        {total_gen_time:.1f}s ({total_gen_time/60:.1f}m)")
    print(f"  Peak VRAM:        {peak_vram_mb:.0f} MB")
    print()
    print(f"  ┌─── Target ─────────────────────────────┐")
    print(f"  │ BLEU:    ~26.4                         │")
    print(f"  └────────────────────────────────────────┘")
    print()
    print(f"  ┌─── Measured ───────────────────────────┐")
    print(f"  │ BLEU:    {bleu:.2f}")
    print(f"  │ ROUGE-1: {rouge['rouge1']:.2f}")
    print(f"  │ ROUGE-2: {rouge['rouge2']:.2f}")
    print(f"  │ ROUGE-L: {rouge['rougeL']:.2f}")
    print(f"  └────────────────────────────────────────┘")

    # Gate check (10% tolerance)
    target_bleu = 26.4
    bleu_ok = abs(bleu - target_bleu) / target_bleu <= 0.10
    print()
    if bleu_ok:
        print(f"  ✓ GATE 0 (De-En): BLEU {bleu:.2f} is within 10% of target {target_bleu}")
    else:
        print(f"  ✗ GATE 0 (De-En): BLEU {bleu:.2f} is OUTSIDE 10% tolerance of "
              f"target {target_bleu}. STOP and diagnose.")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": wrapper.checkpoint_step,
        "config_hash": config_hash,
        "seed": SEED,
        "num_samples": len(all_generated),
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "cfg": CFG,
        "sc_cfg": SC_CFG,
        "time_schedule": TIME_SCHEDULE,
        "eval_data": EVAL_DATA,
        "bleu": bleu,
        **rouge,
        "gen_time_s": total_gen_time,
        "peak_vram_mb": peak_vram_mb,
        "gate_0_deen_pass": bleu_ok,
    }

    results_file = RESULTS_DIR / "deen_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Save sample translations
    samples_file = RESULTS_DIR / "deen_samples.jsonl"
    with open(samples_file, "w") as f:
        for idx, ref, hyp, src in all_generated[:50]:
            f.write(json.dumps({
                "id": idx, "source": src, "reference": ref, "hypothesis": hyp,
            }, ensure_ascii=False) + "\n")
    print(f"  Sample translations saved to {samples_file}")

    return 0 if bleu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
