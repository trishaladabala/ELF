#!/usr/bin/env python3
"""Consistency Flow Distillation for ELF-B-de-en.

Trains a student model (same architecture, initialized from teacher) to satisfy
the self-consistency property: for any two adjacent points on a teacher ODE
trajectory, the student's consistency function maps them to the same x_0.

Key innovation: Conditional Consistency (CCFD) — the consistency loss is applied
only to the generation portion of the sequence, respecting cond_seq_mask.

The consistency function:
    f_θ(z_t, t) = x_pred = model(z_t, t)  (the model's predicted clean output)

Loss:
    L = || f_θ(z_{t_n}, t_n) - sg(f_{θ⁻}(z_{t_{n+1}}, t_{n+1})) ||²
    where z_{t_{n+1}} is obtained by one teacher ODE step from z_{t_n},
    θ⁻ is the EMA of θ, and sg() is stop-gradient.
"""
import sys
import os
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from utils.sampling_utils import (
    get_sampling_steps, restore_cond, _ode_step,
    _forward_sample, net_out_to_v_x, restore_vx,
)
from utils.encoder_utils import encode_text
from utils.data_utils import get_dataloader, get_pad_token_id
from configs.config import SamplingConfig


# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT = "ELF-B-de-en"
NUM_STEPS_TEACHER = 64
CFG = 2.0
SC_CFG = 1.0
TIME_SCHEDULE = "logit_normal"

# Training
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 30
WARMUP_STEPS = 500
EMA_DECAY_START = 0.999
EMA_DECAY_END = 0.9999
GRAD_CLIP = 1.0
SAVE_EVERY = 5  # save checkpoint every N epochs
SEED = 42

# Dataset
NUM_TRAIN = 2000       # first 2000 examples for training
NUM_VAL = 1000         # last 1000 for validation
DATA_DIR = _SCRIPT_DIR.parent / "runs" / "distill_data"
OUT_DIR = _SCRIPT_DIR.parent / "runs" / "consistency_distill"


class OnTheFlyTrajectoryDataset(Dataset):
    """Generates (z_t, t, cond_seq, cond_mask) on the fly.

    For each example, samples a random timestep t from the teacher's schedule,
    creates z_t = t*cond_seq + (1-t)*noise*scale (with proper conditioning),
    and returns the pair needed for consistency loss.
    """
    def __init__(self, eval_dataset, wrapper, num_examples, start_idx=0):
        self.examples = eval_dataset[start_idx:start_idx + num_examples]
        self.num_examples = len(self.examples)
        self.wrapper = wrapper

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return idx  # We batch-process in the training loop


def get_consistency_function(model, z, t_batch, x_pred_prev, config,
                              cfg_scale, sc_cfg_scale, cond_seq, cond_seq_mask):
    """Compute the consistency function f_θ(z_t, t) = predicted x_0.

    This is just the model's prediction at (z_t, t), which is x_pred.
    """
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=sc_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return x_pred


def teacher_ode_step(teacher_model, z, t_val, t_next, x_pred_prev,
                     config, cfg_scale, sc_cfg_scale, cond_seq, cond_seq_mask):
    """One teacher ODE step from t to t_next."""
    with torch.no_grad():
        z_next, x_pred_next = _ode_step(
            model=teacher_model, z=z, t=t_val, t_next=t_next,
            x_pred_prev=x_pred_prev,
            config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=sc_cfg_scale,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        )
    return z_next, x_pred_next


@torch.no_grad()
def update_ema(student_model, ema_model, decay):
    """Update EMA model parameters."""
    for p_student, p_ema in zip(student_model.parameters(), ema_model.parameters()):
        p_ema.data.mul_(decay).add_(p_student.data, alpha=1.0 - decay)


def conditional_mse_loss(pred, target, cond_seq_mask):
    """MSE loss only on generation (non-condition) positions.

    cond_seq_mask: (B, L), 1 = condition position, 0 = generation position.
    We want loss only where cond_seq_mask == 0.
    """
    gen_mask = (1.0 - cond_seq_mask).unsqueeze(-1)  # (B, L, 1)
    num_gen = gen_mask.sum().clamp(min=1.0)
    diff = (pred - target) ** 2
    loss = (diff * gen_mask).sum() / num_gen
    return loss


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Consistency Flow Distillation Training")
    print("=" * 70)

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # ── Load teacher ───────────────────────────────────────────────────
    print("\nLoading teacher model...")
    teacher_wrapper = ELFWrapper(CHECKPOINT, device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    config = teacher_wrapper.config
    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    d_model = teacher_wrapper.d_model
    tokenizer = teacher_wrapper.tokenizer
    encoder = teacher_wrapper.encoder
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    param_dtype = next(teacher_model.parameters()).dtype

    # ── Create student (copy of teacher) ───────────────────────────────
    print("Creating student model (from teacher weights)...")
    student_wrapper = ELFWrapper(CHECKPOINT, device=device)
    student_model = student_wrapper.model
    student_model.train()
    for p in student_model.parameters():
        p.requires_grad_(True)

    # ── Create EMA model ───────────────────────────────────────────────
    print("Creating EMA model...")
    ema_wrapper = ELFWrapper(CHECKPOINT, device=device)
    ema_model = ema_wrapper.model
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ── Load dataset ───────────────────────────────────────────────────
    print("\nLoading dataset...")
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    EVAL_DATA = "embedded-language-flows/wmt14_de-en_validation_t5"
    local_dir = snapshot_download(repo_id=EVAL_DATA, repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')

    train_data = eval_dataset[:NUM_TRAIN]
    val_data = eval_dataset[NUM_TRAIN:NUM_TRAIN + NUM_VAL]

    train_loader = get_dataloader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    val_loader = get_dataloader(
        val_data, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = NUM_EPOCHS * len(train_loader)

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training loop ──────────────────────────────────────────────────
    print(f"\nTraining for {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"Total steps: {total_steps}")
    print(f"Student params: {sum(p.numel() for p in student_model.parameters()):,}")

    generator = torch.Generator(device=device).manual_seed(SEED)
    global_step = 0
    best_val_loss = float('inf')
    history = []

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attn_mask = torch.from_numpy(
                np.array(batch["encoder_attention_mask"])).to(device).float()
            cond_seq_mask = torch.from_numpy(
                np.array(batch["cond_seq_mask"])).to(device).float()

            # Encode source text
            with torch.no_grad():
                cond_seq = encode_text(
                    input_ids=input_ids, attention_mask=encoder_attn_mask,
                    encoder=encoder, latent_mean=config.latent_mean,
                    latent_std=config.latent_std,
                ).to(param_dtype)

            # Sample teacher timestep schedule
            t_steps = get_sampling_steps(
                n_steps=NUM_STEPS_TEACHER, time_schedule=TIME_SCHEDULE,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )

            # Sample random starting noise
            z_init = (torch.randn((bsz, config.max_length, d_model),
                                  generator=generator, dtype=param_dtype, device=device)
                      * config.denoiser_noise_scale)
            z_init = restore_cond(z_init, cond_seq, cond_seq_mask)
            x_pred_init = restore_cond(torch.zeros_like(z_init), cond_seq, cond_seq_mask)

            # Pick a random step index for this batch (which adjacent pair to use)
            # Uniform over all steps — this samples the full trajectory
            step_idx = torch.randint(0, NUM_STEPS_TEACHER - 1, (1,)).item()

            # Run teacher ODE from step 0 to step_idx to get z_{t_n}
            z = z_init
            x_pred_prev = x_pred_init
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                for i in range(step_idx + 1):
                    if i < step_idx:
                        t_val = t_steps[i].item()
                        t_next = t_steps[i + 1].item()
                        z, x_pred_prev = _ode_step(
                            model=teacher_model, z=z, t=t_val, t_next=t_next,
                            x_pred_prev=x_pred_prev,
                            config=config, cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                        )

            # Now z is at step step_idx, x_pred_prev is the teacher's prediction there
            z_t_n = z.detach().clone()
            t_n = t_steps[step_idx].item()
            t_n_next = t_steps[step_idx + 1].item()

            # Teacher ODE step: z_{t_n} → z_{t_{n+1}}
            with torch.no_grad():
                z_t_n1, teacher_x_pred_n1 = _ode_step(
                    model=teacher_model, z=z_t_n, t=t_n, t_next=t_n_next,
                    x_pred_prev=x_pred_prev,
                    config=config, cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                )

            # ── Consistency Loss ───────────────────────────────────────
            # Student predicts x_0 from z_{t_n}
            t_n_batch = torch.full((bsz,), t_n, dtype=param_dtype, device=device)
            student_x_pred_n = get_consistency_function(
                student_model, z_t_n, t_n_batch, x_pred_prev,
                config, CFG, SC_CFG, cond_seq, cond_seq_mask,
            )

            # EMA model predicts x_0 from z_{t_{n+1}} (target, no gradient)
            t_n1_batch = torch.full((bsz,), t_n_next, dtype=param_dtype, device=device)
            with torch.no_grad():
                ema_x_pred_n1 = get_consistency_function(
                    ema_model, z_t_n1.detach(), t_n1_batch, teacher_x_pred_n1.detach(),
                    config, CFG, SC_CFG, cond_seq, cond_seq_mask,
                )

            # Conditional MSE loss (only on generation positions)
            loss = conditional_mse_loss(
                student_x_pred_n.float(), ema_x_pred_n1.float(), cond_seq_mask
            )

            # ── Backward ───────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # ── Update EMA ─────────────────────────────────────────────
            progress = global_step / max(1, total_steps)
            ema_decay = EMA_DECAY_START + (EMA_DECAY_END - EMA_DECAY_START) * progress
            update_ema(student_model, ema_model, ema_decay)

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # ── Validation ─────────────────────────────────────────────────
        student_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                bsz = batch["input_ids"].shape[0]
                input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
                encoder_attn_mask = torch.from_numpy(
                    np.array(batch["encoder_attention_mask"])).to(device).float()
                cond_seq_mask_v = torch.from_numpy(
                    np.array(batch["cond_seq_mask"])).to(device).float()

                cond_seq_v = encode_text(
                    input_ids=input_ids, attention_mask=encoder_attn_mask,
                    encoder=encoder, latent_mean=config.latent_mean,
                    latent_std=config.latent_std,
                ).to(param_dtype)

                t_steps_v = get_sampling_steps(
                    n_steps=NUM_STEPS_TEACHER, time_schedule=TIME_SCHEDULE,
                    P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                    device=device, dtype=param_dtype,
                )

                z_v = (torch.randn((bsz, config.max_length, d_model),
                                   dtype=param_dtype, device=device)
                       * config.denoiser_noise_scale)
                z_v = restore_cond(z_v, cond_seq_v, cond_seq_mask_v)
                x_prev_v = restore_cond(torch.zeros_like(z_v), cond_seq_v, cond_seq_mask_v)

                # Use middle step for validation consistency
                mid = NUM_STEPS_TEACHER // 2
                for i in range(mid):
                    z_v, x_prev_v = _ode_step(
                        model=teacher_model, z=z_v, t=t_steps_v[i].item(),
                        t_next=t_steps_v[i+1].item(), x_pred_prev=x_prev_v,
                        config=config, cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                        cond_seq=cond_seq_v, cond_seq_mask=cond_seq_mask_v,
                    )

                t_mid = t_steps_v[mid].item()
                t_mid_next = t_steps_v[mid+1].item()

                z_next_v, x_next_v = _ode_step(
                    model=teacher_model, z=z_v, t=t_mid, t_next=t_mid_next,
                    x_pred_prev=x_prev_v,
                    config=config, cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
                    cond_seq=cond_seq_v, cond_seq_mask=cond_seq_mask_v,
                )

                t_mid_b = torch.full((bsz,), t_mid, dtype=param_dtype, device=device)
                t_mid_next_b = torch.full((bsz,), t_mid_next, dtype=param_dtype, device=device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                    s_pred = get_consistency_function(
                        student_model, z_v, t_mid_b, x_prev_v,
                        config, CFG, SC_CFG, cond_seq_v, cond_seq_mask_v,
                    )
                    e_pred = get_consistency_function(
                        ema_model, z_next_v, t_mid_next_b, x_next_v,
                        config, CFG, SC_CFG, cond_seq_v, cond_seq_mask_v,
                    )

                vl = conditional_mse_loss(s_pred.float(), e_pred.float(), cond_seq_mask_v)
                val_loss += vl.item()
                val_batches += 1

        avg_val = val_loss / max(1, val_batches)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} — "
              f"Train: {avg_loss:.6f}, Val: {avg_val:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}, "
              f"EMA: {ema_decay:.5f}, "
              f"Time: {epoch_time:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val,
            "lr": scheduler.get_last_lr()[0],
            "ema_decay": ema_decay,
            "time_s": epoch_time,
        })

        # Save checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(student_model.state_dict(), OUT_DIR / "student_best.pt")
            torch.save(ema_model.state_dict(), OUT_DIR / "ema_best.pt")
            print(f"  ✓ New best val loss: {best_val_loss:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(student_model.state_dict(), OUT_DIR / f"student_ep{epoch+1}.pt")

    # ── Save final ─────────────────────────────────────────────────────
    torch.save(student_model.state_dict(), OUT_DIR / "student_final.pt")
    torch.save(ema_model.state_dict(), OUT_DIR / "ema_final.pt")

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"  Models saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
