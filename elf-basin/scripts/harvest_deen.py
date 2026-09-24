#!/usr/bin/env python3
"""Phase 2: Harvest De-En trajectories for adapter training.

Uses the SAME data pipeline as reproduce_deen.py (via data_utils.get_dataloader)
to ensure conditioning masks are dynamically computed from actual token lengths,
not hardcoded to 64.
"""
import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.harvest.harvester import TrajectoryHarvester
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch
from utils.encoder_utils import encode_text
from utils.data_utils import load_dataset_split, get_dataloader, get_pad_token_id
from configs.config import SamplingConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "ELF-B-de-en"
    NUM_SAMPLES = 3000
    BATCH_SIZE = 16
    NUM_STEPS = 64
    SDE_GAMMA = 0.0
    SC_CFG = 1.0
    CFG = 2.0
    SEED = 42
    TIME_SCHEDULE = "logit_normal"

    HARVEST_STEPS = [8, 16, 24, 32, 40, 48, 56, 62]

    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"

    print(f"Harvesting De-En trajectories to {RESULTS_DIR}")

    wrapper = ELFWrapper(CHECKPOINT, device=device)
    model = wrapper.model
    config = wrapper.config
    d_model = wrapper.d_model
    max_length = wrapper.max_length
    param_dtype = next(model.parameters()).dtype
    tokenizer = wrapper.tokenizer
    encoder = wrapper.encoder

    # Add missing config fields for data_utils compatibility
    config.max_input_length = getattr(config, "max_input_length", 64)
    config.pad_token = getattr(config, "pad_token", "eos")
    config.label_drop_prob = getattr(config, "label_drop_prob", 0.1)
    config.use_bf16 = True
    config.online_eval = True

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    sampling_config = SamplingConfig(
        sampling_method="ode",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        sde_gamma=SDE_GAMMA,
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )

    harvester = TrajectoryHarvester(
        output_dir=str(RESULTS_DIR),
        num_samples=NUM_SAMPLES,
        seq_len=max_length,
        d_model=d_model,
        steps=HARVEST_STEPS,
        max_gb=50.0
    )

    # ── Use the SAME data pipeline as reproduce_deen.py ──
    EVAL_DATA = "embedded-language-flows/wmt14_de-en_validation_t5"
    import pyarrow.ipc as ipc
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=EVAL_DATA, repo_type="dataset")
    arrow_file = os.path.join(local_dir, "data-00000-of-00001.arrow")
    with open(arrow_file, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
    eval_dataset = table.to_pandas().to_dict('records')
    actual_samples = min(NUM_SAMPLES, len(eval_dataset))
    print(f"Dataset size: {len(eval_dataset)}, using {actual_samples} samples")

    # Use the EXACT same dataloader as reproduce_deen.py
    # This correctly computes cond_seq_mask based on actual token lengths
    dataloader = get_dataloader(
        eval_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
    else:
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    samples_processed = 0
    from tqdm import tqdm
    pbar = tqdm(total=(actual_samples + BATCH_SIZE - 1) // BATCH_SIZE,
                desc="Harvesting")

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

        z_init = (torch.randn((bsz, config.max_length, d_model),
                              generator=generator, dtype=param_dtype, device=device)
                  * config.denoiser_noise_scale)

        # Compute GT tokens and GT encoder states
        # GT tokens are the full input_ids (condition + target, padded to max_length)
        gt_tokens = input_ids.clone()

        with torch.no_grad():
            gt_encoder_states = wrapper.encode(gt_tokens)

        start_idx = batch_idx * BATCH_SIZE
        harvester.harvest_batch(
            start_idx=start_idx,
            wrapper=wrapper,
            generator=generator,
            z_init=z_init,
            t_steps=t_steps,
            cond_seq=cond_seq,
            cond_seq_mask=cond_seq_mask_arr,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=CFG,
            self_cond_cfg_scale=SC_CFG,
            gt_tokens=gt_tokens,
            gt_encoder_states=gt_encoder_states
        )

        samples_processed += bsz
        pbar.update(1)

        if (batch_idx + 1) % 10 == 0:
            print(f"Harvested {samples_processed}/{actual_samples} samples")

    pbar.close()
    print(f"\nDone. Harvested {samples_processed} samples to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
