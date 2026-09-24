#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.harvest.harvester import TrajectoryHarvester
from utils.sampling_utils import get_sampling_steps
from configs.config import SamplingConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "ELF-B-owt"
    NUM_SAMPLES = 4096
    BATCH_SIZE = 8
    NUM_STEPS = 32
    SDE_GAMMA = 1.5
    SC_CFG = 3.0
    CFG = 1.0
    SEED = 42
    TIME_SCHEDULE = "logit_normal"
    
    # Subsampled timesteps
    HARVEST_STEPS = [8, 16, 24, 31]
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_owt"
    
    print(f"Harvesting OWT trajectories to {RESULTS_DIR}")
    
    wrapper = ELFWrapper(CHECKPOINT, device=device)
    model = wrapper.model
    config = wrapper.config
    d_model = wrapper.d_model
    max_length = wrapper.max_length
    param_dtype = next(model.parameters()).dtype

    sampling_config = SamplingConfig(
        sampling_method="sde",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        sde_gamma=SDE_GAMMA,
        self_cond_cfg_scales=[SC_CFG],
        time_schedule=TIME_SCHEDULE,
    )
    
    # Fixed random positions for subsampling (256 out of 1024)
    np.random.seed(SEED)
    SUB_POSITIONS = np.sort(np.random.choice(max_length, size=256, replace=False))
    
    harvester = TrajectoryHarvester(
        output_dir=str(RESULTS_DIR),
        num_samples=NUM_SAMPLES,
        seq_len=256,  # Used for disk budget check and memmap initialization
        d_model=d_model,
        steps=HARVEST_STEPS,
        max_gb=50.0
    )
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        current_batch = min(BATCH_SIZE, NUM_SAMPLES - start_idx)
        
        t_steps = get_sampling_steps(
            n_steps=NUM_STEPS, time_schedule=TIME_SCHEDULE,
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            device=device, dtype=param_dtype,
        )
        
        generator = torch.Generator(device=device).manual_seed(SEED + batch_idx)
        z_init = torch.randn(
            (current_batch, max_length, d_model),
            dtype=param_dtype, device=device, generator=generator,
        ) * config.denoiser_noise_scale
        
        harvester.harvest_batch(
            start_idx=start_idx,
            wrapper=wrapper,
            generator=generator,
            z_init=z_init,
            t_steps=t_steps,
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=CFG,
            self_cond_cfg_scale=SC_CFG,
            gt_tokens=None,
            gt_encoder_states=None,
            positions=SUB_POSITIONS
        )
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Harvested {start_idx + current_batch}/{NUM_SAMPLES} samples")
            
if __name__ == "__main__":
    main()
