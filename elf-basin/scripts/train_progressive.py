#!/usr/bin/env python3
"""Progressive Distillation baseline for ELF-B-de-en.

Implements 4-stage progressive distillation (64→32→16→8→4 steps).
Each stage: the teacher takes 2 steps, the student is trained to match
with 1 step (in latent space, MSE loss on generation positions).

This is our BASELINE — ELF's authors already did this (Appendix B).
We must beat it with consistency distillation to have a paper.
"""
import sys
import os
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from utils.sampling_utils import (
    get_sampling_steps, restore_cond, _ode_step, _forward_sample,
    net_out_to_v_x, restore_vx, add_noise,
)
from utils.encoder_utils import encode_text
from utils.data_utils import get_dataloader, get_pad_token_id
from configs.config import SamplingConfig


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-de-en"
CFG = 2.0
SC_CFG = 1.0
TIME_SCHEDULE = "logit_normal"

# Progressive distillation stages
STAGES = [
    {"teacher_steps": 64, "student_steps": 32},
    {"teacher_steps": 32, "student_steps": 16},
    {"teacher_steps": 16, "student_steps": 8},
    {"teacher_steps": 8,  "student_steps": 4},
]

# Training per stage
BATCH_SIZE = 8
LR = 5e-5
NUM_EPOCHS_PER_STAGE = 15
GRAD_CLIP = 1.0
SEED = 42

NUM_TRAIN = 2000
NUM_VAL = 1000
OUT_DIR = _SCRIPT_DIR.parent / "runs" / "progressive_distill"


def conditional_mse_loss(pred, target, cond_seq_mask):
    """MSE loss only on generation positions."""
    gen_mask = (1.0 - cond_seq_mask).unsqueeze(-1)
    num_gen = gen_mask.sum().clamp(min=1.0)
    diff = (pred - target) ** 2
    return (diff * gen_mask).sum() / num_gen


def generate_with_n_steps(model, z, n_steps, cond_seq, cond_seq_mask, config,
                          cfg_scale, sc_cfg_scale, time_schedule):
    """Generate using a model with exactly n_steps ODE steps."""
    param_dtype = next(model.parameters()).dtype
    device = z.device

    t_steps = get_sampling_steps(
        n_steps=n_steps, time_schedule=time_schedule,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=param_dtype,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=sc_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    for i in range(len(t_steps) - 1):
        t_val = t_steps[i].item()
        t_next = t_steps[i + 1].item()
        z, x_pred = _ode_step(z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)

    return z, x_pred


def train_one_stage(stage_idx, stage_config, teacher_model, student_model,
                    train_loader, val_loader, config, device, out_dir):
    """Train one progressive distillation stage."""
    teacher_steps = stage_config["teacher_steps"]
    student_steps = stage_config["student_steps"]

    print(f"\n{'='*70}")
    print(f"  Stage {stage_idx+1}: {teacher_steps} → {student_steps} steps")
    print(f"{'='*70}")

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    student_model.train()
    param_dtype = next(teacher_model.parameters()).dtype
    encoder = None
    # We need the encoder from the wrapper — reconstruct it
    # Actually, we pass it from main

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=LR, weight_decay=0.01
    )
    total_steps = NUM_EPOCHS_PER_STAGE * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    generator = torch.Generator(device=device).manual_seed(SEED + stage_idx)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 4

    for epoch in range(NUM_EPOCHS_PER_STAGE):
        student_model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch in tqdm(train_loader, desc=f"Stage {stage_idx+1} Ep {epoch+1}"):
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attn_mask = torch.from_numpy(
                np.array(batch["encoder_attention_mask"])).to(device).float()
            cond_seq_mask = torch.from_numpy(
                np.array(batch["cond_seq_mask"])).to(device).float()

            with torch.no_grad():
                cond_seq = encode_text(
                    input_ids=input_ids, attention_mask=encoder_attn_mask,
                    encoder=train_one_stage.encoder,
                    latent_mean=config.latent_mean, latent_std=config.latent_std,
                ).to(param_dtype)

            # Sample noise
            d_model = cond_seq.shape[-1]
            z_noise = (torch.randn((bsz, config.max_length, d_model),
                                   generator=generator, dtype=param_dtype, device=device)
                       * config.denoiser_noise_scale)

            # Teacher: run student_steps * 2 = teacher_steps of the ODE
            # then take the result of every 2 teacher steps as the target
            # for the student's 1 step

            # Get teacher timestep schedule
            t_teacher = get_sampling_steps(
                n_steps=teacher_steps, time_schedule=TIME_SCHEDULE,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )

            # Pick a random student step to train on
            student_step_idx = torch.randint(0, student_steps, (1,)).item()

            # The student's timestep schedule has student_steps+1 entries
            # For the student, step i goes from t_student[i] to t_student[i+1]
            # The equivalent teacher range is [t_teacher[2*i], t_teacher[2*i+2]]
            # (teacher takes 2 steps for student's 1 step)
            teacher_start_idx = student_step_idx * 2
            teacher_mid_idx = teacher_start_idx + 1
            teacher_end_idx = teacher_start_idx + 2

            t_start = t_teacher[teacher_start_idx].item()
            t_mid = t_teacher[teacher_mid_idx].item()
            t_end = t_teacher[teacher_end_idx].item()

            # Run teacher to get to the starting point
            z = restore_cond(z_noise.clone(), cond_seq, cond_seq_mask)
            x_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

            with torch.no_grad():
                step_kwargs = dict(
                    model=teacher_model, config=config,
                    cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                )
                for i in range(teacher_start_idx):
                    z, x_prev = _ode_step(
                        z=z, t=t_teacher[i].item(), t_next=t_teacher[i+1].item(),
                        x_pred_prev=x_prev, **step_kwargs
                    )

            z_start = z.detach().clone()
            x_prev_start = x_prev.detach().clone()

            # Teacher: 2 steps from z_start
            with torch.no_grad():
                z_mid, x_mid = _ode_step(
                    z=z_start, t=t_start, t_next=t_mid, x_pred_prev=x_prev_start,
                    **step_kwargs
                )
                z_target, x_target = _ode_step(
                    z=z_mid, t=t_mid, t_next=t_end, x_pred_prev=x_mid,
                    **step_kwargs
                )

            # Student: 1 step from z_start to match z_target
            # The student uses its own velocity prediction
            t_start_batch = torch.full((bsz,), t_start, dtype=param_dtype, device=device)
            v_student, x_student = _forward_sample(
                model=student_model, z=z_start, t_batch=t_start_batch,
                x_pred_prev=x_prev_start, config=config,
                cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )

            # Student's z after 1 step (from t_start to t_end)
            z_student = z_start + (t_end - t_start) * v_student

            # Loss: match the teacher's 2-step output (in latent space)
            loss = conditional_mse_loss(
                z_student.float(), z_target.float(), cond_seq_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} — Train loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s")

        # Simple validation
        student_model.eval()
        val_loss_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                bsz = batch["input_ids"].shape[0]
                input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
                ea = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
                cm = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()

                cs = encode_text(
                    input_ids=input_ids, attention_mask=ea,
                    encoder=train_one_stage.encoder,
                    latent_mean=config.latent_mean, latent_std=config.latent_std,
                ).to(param_dtype)

                d = cs.shape[-1]
                zn = (torch.randn((bsz, config.max_length, d), dtype=param_dtype, device=device)
                      * config.denoiser_noise_scale)
                zn = restore_cond(zn, cs, cm)
                xp = restore_cond(torch.zeros_like(zn), cs, cm)

                t_t = get_sampling_steps(
                    n_steps=teacher_steps, time_schedule=TIME_SCHEDULE,
                    P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                    device=device, dtype=param_dtype,
                )

                mid_s = student_steps // 2
                ti = mid_s * 2
                for i in range(ti):
                    zn, xp = _ode_step(
                        model=teacher_model, z=zn, t=t_t[i].item(), t_next=t_t[i+1].item(),
                        x_pred_prev=xp, config=config, cfg_scale=CFG,
                        self_cond_cfg_scale=SC_CFG, cond_seq=cs, cond_seq_mask=cm,
                    )

                ts = t_t[ti].item()
                tm = t_t[ti+1].item()
                te = t_t[ti+2].item()

                zm, xm = _ode_step(model=teacher_model, z=zn, t=ts, t_next=tm,
                                   x_pred_prev=xp, config=config, cfg_scale=CFG,
                                   self_cond_cfg_scale=SC_CFG, cond_seq=cs, cond_seq_mask=cm)
                zt, xt = _ode_step(model=teacher_model, z=zm, t=tm, t_next=te,
                                   x_pred_prev=xm, config=config, cfg_scale=CFG,
                                   self_cond_cfg_scale=SC_CFG, cond_seq=cs, cond_seq_mask=cm)

                tb = torch.full((bsz,), ts, dtype=param_dtype, device=device)
                vs, xs = _forward_sample(model=student_model, z=zn, t_batch=tb,
                                         x_pred_prev=xp, config=config, cfg_scale=CFG,
                                         self_cond_cfg_scale=SC_CFG, cond_seq=cs, cond_seq_mask=cm)
                zs = zn + (te - ts) * vs

                vl = conditional_mse_loss(zs.float(), zt.float(), cm)
                val_loss_total += vl.item()
                val_n += 1

        avg_val = val_loss_total / max(1, val_n)
        print(f"  Val loss: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(student_model.state_dict(),
                       out_dir / f"stage{stage_idx+1}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint for this stage
    best_path = out_dir / f"stage{stage_idx+1}_best.pt"
    if best_path.exists():
        student_model.load_state_dict(torch.load(best_path, map_location=device))
    return student_model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Progressive Distillation (Baseline)")
    print("=" * 70)

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # Load initial teacher
    teacher_wrapper = ELFWrapper(CHECKPOINT, device=device)
    config = teacher_wrapper.config
    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    tokenizer = teacher_wrapper.tokenizer
    encoder = teacher_wrapper.encoder
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    # Store encoder for use in training function
    train_one_stage.encoder = encoder

    # Load dataset
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id="embedded-language-flows/wmt14_de-en_validation_t5", repo_type="dataset"
    )
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        table = ipc.RecordBatchStreamReader(f).read_all()
    eval_dataset = table.to_pandas().to_dict('records')

    train_loader = get_dataloader(
        eval_dataset[:NUM_TRAIN], batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    val_loader = get_dataloader(
        eval_dataset[NUM_TRAIN:NUM_TRAIN+NUM_VAL], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    # Progressive distillation loop
    teacher_model = teacher_wrapper.model
    history = []

    for stage_idx, stage_config in enumerate(STAGES):
        # Create fresh student from current teacher
        student_wrapper = ELFWrapper(CHECKPOINT, device=device)
        student_model = student_wrapper.model
        # Initialize student from teacher's weights
        student_model.load_state_dict(teacher_model.state_dict())
        for p in student_model.parameters():
            p.requires_grad_(True)

        student_model = train_one_stage(
            stage_idx, stage_config, teacher_model, student_model,
            train_loader, val_loader, config, device, OUT_DIR
        )

        # Save this stage's model
        torch.save(student_model.state_dict(),
                   OUT_DIR / f"student_{stage_config['student_steps']}step.pt")

        # Student becomes the next teacher
        teacher_model = student_model
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        history.append({
            "stage": stage_idx + 1,
            "teacher_steps": stage_config["teacher_steps"],
            "student_steps": stage_config["student_steps"],
        })

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Progressive distillation complete. Models saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
