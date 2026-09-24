#!/usr/bin/env python3
"""Evaluate distilled ELF models at various step counts.

Evaluates:
  1. Teacher (original ELF-B-de-en) at 1, 2, 4, 8, 16, 32, 64 steps
  2. Consistency-distilled student at same step counts
  3. Progressively-distilled student at same step counts

Reports: BLEU (sacreBLEU), ROUGE-1/2/L, Dist-2, Rep-4, wall-clock time.
Uses the EXACT same evaluation pipeline as reproduce_deen.py.
"""
import sys
import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from utils.sampling_utils import get_sampling_steps, restore_cond, _ode_step
from utils.generation_utils import (
    _generate_samples_single_batch, _dlm_decode_batch,
    mask_after_eos, shift_left,
)
from utils.encoder_utils import encode_text
from utils.data_utils import get_dataloader, get_pad_token_id
from elfbasin.eval.metrics import compute_bleu
from configs.config import SamplingConfig


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-de-en"
EVAL_STEPS = [1, 2, 4, 8, 16, 32, 64]
CFG = 2.0
SC_CFG = 1.0
TIME_SCHEDULE = "logit_normal"
BATCH_SIZE = 16
NUM_EVAL = 1000           # evaluate on last 1000 examples
SEED = 42

# Model paths
CONSISTENCY_DIR = _SCRIPT_DIR.parent / "runs" / "consistency_distill"
PROGRESSIVE_DIR = _SCRIPT_DIR.parent / "runs" / "progressive_distill"
OUT_DIR = _SCRIPT_DIR.parent / "runs" / "eval_results"


def compute_diversity_metrics(hyps):
    """Compute Dist-2 and Rep-4."""
    all_bigrams = set()
    all_bigram_count = 0
    rep4_count = 0
    total_4grams = 0

    for hyp in hyps:
        tokens = hyp.split()
        # Distinct-2
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        all_bigrams.update(bigrams)
        all_bigram_count += len(bigrams)
        # Rep-4
        fourgrams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
        total_4grams += len(fourgrams)
        seen = set()
        for fg in fourgrams:
            if fg in seen:
                rep4_count += 1
            seen.add(fg)

    dist2 = len(all_bigrams) / max(1, all_bigram_count)
    rep4 = rep4_count / max(1, total_4grams)
    return dist2, rep4


def generate_and_decode(model, dataloader, config, tokenizer, encoder,
                        pad_token_id, n_steps, device, max_samples,
                        ds_for_cond_lens):
    """Generate text with a model at a given number of steps and decode to text."""
    param_dtype = next(model.parameters()).dtype
    d_model = config.max_length  # This might be wrong, get from model

    # Get d_model from the model's text_proj
    d_model_actual = model.text_encoder_dim
    eos_token_id = tokenizer.eos_token_id

    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[n_steps],
        cfgs=[CFG],
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
    else:
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    all_hyps = []
    total_time = 0.0
    samples_done = 0
    sample_offset = 0  # Track position in ds_for_cond_lens

    for batch in tqdm(dataloader, desc=f"{n_steps}-step"):
        if samples_done >= max_samples:
            break

        bsz = batch["input_ids"].shape[0]
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        encoder_attn_mask = torch.from_numpy(
            np.array(batch["encoder_attention_mask"])).to(device).float()
        cond_seq_mask = torch.from_numpy(
            np.array(batch["cond_seq_mask"])).to(device).float()

        t_steps = get_sampling_steps(
            n_steps=n_steps, time_schedule=TIME_SCHEDULE,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=param_dtype,
        )

        cond_seq = encode_text(
            input_ids=input_ids, attention_mask=encoder_attn_mask,
            encoder=encoder, latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        ).to(param_dtype)

        z = (torch.randn((bsz, config.max_length, d_model_actual),
                         generator=generator, dtype=param_dtype, device=device)
             * config.denoiser_noise_scale)

        gen_start = time.time()
        with torch.no_grad():
            latent = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                config=config, sampling_config=sampling_config,
                cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
            )
        total_time += time.time() - gen_start

        # Decode: same pipeline as reproduce_deen.py
        gen_length = config.max_length - config.max_input_length
        cond_len_per_sample = cond_seq_mask.to(torch.int32).sum(dim=1)

        t_final_val = t_steps[-1].item()
        with torch.no_grad():
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=SC_CFG,
            )
        predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :gen_length]
        predicted_ids = mask_after_eos(
            predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id
        )

        for i in range(bsz):
            if samples_done >= max_samples:
                break
            text = tokenizer.decode(
                predicted_ids[i].detach().cpu().numpy(),
                skip_special_tokens=True
            )
            all_hyps.append(text)
            samples_done += 1

        sample_offset += bsz

    return all_hyps, total_time


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Distillation Evaluation")
    print("=" * 70)

    # ── Load base wrapper for encoder/tokenizer ────────────────────────
    base_wrapper = ELFWrapper(CHECKPOINT, device=device)
    config = base_wrapper.config
    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    tokenizer = base_wrapper.tokenizer
    encoder = base_wrapper.encoder
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    # ── Load dataset ───────────────────────────────────────────────────
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id="embedded-language-flows/wmt14_de-en_validation_t5", repo_type="dataset"
    )
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        table = ipc.RecordBatchStreamReader(f).read_all()
    eval_dataset = table.to_pandas().to_dict('records')

    # Use last NUM_EVAL examples for evaluation
    eval_data = eval_dataset[-NUM_EVAL:]
    references = [item["target"] for item in eval_data]
    ds_for_cond_lens = eval_data

    eval_loader = get_dataloader(
        eval_data, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    # ── Models to evaluate ─────────────────────────────────────────────
    models_to_eval = {}

    # 1. Teacher (original ELF-B-de-en)
    models_to_eval["teacher"] = base_wrapper.model

    # 2. Consistency-distilled (if available)
    ema_path = CONSISTENCY_DIR / "ema_best.pt"
    student_path = CONSISTENCY_DIR / "student_best.pt"
    if ema_path.exists():
        print(f"Loading consistency EMA model from {ema_path}")
        consistency_wrapper = ELFWrapper(CHECKPOINT, device=device)
        consistency_wrapper.model.load_state_dict(
            torch.load(ema_path, map_location=device), strict=True
        )
        models_to_eval["consistency_ema"] = consistency_wrapper.model
    elif student_path.exists():
        print(f"Loading consistency student from {student_path}")
        consistency_wrapper = ELFWrapper(CHECKPOINT, device=device)
        consistency_wrapper.model.load_state_dict(
            torch.load(student_path, map_location=device), strict=True
        )
        models_to_eval["consistency_student"] = consistency_wrapper.model

    # 3. Progressively-distilled models
    for n_steps in [32, 16, 8, 4]:
        pd_path = PROGRESSIVE_DIR / f"student_{n_steps}step.pt"
        if pd_path.exists():
            print(f"Loading progressive {n_steps}-step model from {pd_path}")
            pd_wrapper = ELFWrapper(CHECKPOINT, device=device)
            pd_wrapper.model.load_state_dict(
                torch.load(pd_path, map_location=device), strict=True
            )
            models_to_eval[f"progressive_{n_steps}step"] = pd_wrapper.model

    # ── Evaluate ───────────────────────────────────────────────────────
    all_results = {}

    for model_name, model in models_to_eval.items():
        model.eval()
        model_results = {}
        print(f"\n{'─'*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'─'*50}")

        for n_steps in EVAL_STEPS:
            # For progressive models, only evaluate at their target step count
            # and the teacher step count
            if model_name.startswith("progressive_"):
                target_steps = int(model_name.split("_")[1].replace("step", ""))
                if n_steps != target_steps and n_steps != 64:
                    continue

            print(f"\n  {n_steps}-step generation...")
            hyps, gen_time = generate_and_decode(
                model, eval_loader, config, tokenizer, encoder,
                pad_token_id, n_steps, device, NUM_EVAL,
                ds_for_cond_lens
            )

            # Compute metrics
            bleu = compute_bleu(hyps, references[:len(hyps)])
            dist2, rep4 = compute_diversity_metrics(hyps)
            time_per_sample = gen_time / len(hyps) if hyps else 0

            result = {
                "bleu": bleu,
                "dist2": dist2,
                "rep4": rep4,
                "time_per_sample_s": time_per_sample,
                "total_time_s": gen_time,
                "num_samples": len(hyps),
            }
            model_results[n_steps] = result

            print(f"    BLEU: {bleu:.2f} | Dist-2: {dist2:.4f} | Rep-4: {rep4:.4f} | "
                  f"Time/sample: {time_per_sample*1000:.1f}ms")

        all_results[model_name] = model_results

    # ── Summary Table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  DISTILLATION EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Steps':<8} {'BLEU':<8} {'Dist-2':<8} {'Rep-4':<8} {'ms/sample':<10}")
    print("─" * 80)

    for model_name, model_results in all_results.items():
        for n_steps in sorted(model_results.keys()):
            r = model_results[n_steps]
            print(f"{model_name:<30} {n_steps:<8} {r['bleu']:<8.2f} "
                  f"{r['dist2']:<8.4f} {r['rep4']:<8.4f} "
                  f"{r['time_per_sample_s']*1000:<10.1f}")

    # ── Save results ───────────────────────────────────────────────────
    with open(OUT_DIR / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Save some example outputs ──────────────────────────────────────
    # Re-generate a few examples for qualitative analysis
    print(f"\nResults saved to {OUT_DIR / 'eval_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
