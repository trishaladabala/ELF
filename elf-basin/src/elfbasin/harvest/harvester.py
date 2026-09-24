import torch
import numpy as np
from typing import List, Optional
from .memmap_store import TrajectoryStore
from ..sampling.instrumented_sampler import instrumented_generate
from utils.sampling_utils import get_sampling_steps

class TrajectoryHarvester:
    def __init__(self, output_dir: str, num_samples: int, seq_len: int, d_model: int, steps: List[int], max_gb: float = 50.0):
        self.store = TrajectoryStore(output_dir, num_samples, seq_len, d_model, steps, mode='w+')
        projected_gb = self.store.check_disk_budget()
        print(f"Projected disk usage: {projected_gb:.2f} GB")
        if projected_gb > max_gb:
            raise ValueError(f"Projected disk usage {projected_gb:.2f} GB exceeds maximum allowed {max_gb:.2f} GB")
        
    def harvest_batch(self, start_idx: int, wrapper, generator, z_init, t_steps, cond_seq, cond_seq_mask, config, sampling_config, cfg_scale, self_cond_cfg_scale, gt_tokens=None, gt_encoder_states=None, positions=None):
        # Run generation with diagnostics only at the target steps
        z_final, trajectory = instrumented_generate(
            model=wrapper.model,
            wrapper=wrapper,
            generator=generator,
            z=z_init,
            t_steps=t_steps,
            cond_seq=cond_seq,
            cond_seq_mask=cond_seq_mask,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=cfg_scale,
            self_cond_cfg_scale=self_cond_cfg_scale,
            compute_diagnostics=True,
            compute_L_i=False,
            store_states=True,
            diagnostic_steps=self.store.steps
        )
        
        # x_ref computation: final token IDs -> T5 encoder -> latent norm
        final_ids = trajectory.final_token_ids.to(z_final.device)
        with torch.no_grad():
            x_ref = wrapper.encode(final_ids)
            
        # Apply position subsampling if provided
        if positions is not None:
            final_ids = final_ids[:, positions]
            x_ref = x_ref[:, positions, :]
            if gt_tokens is not None:
                gt_tokens = gt_tokens[:, positions]
            if gt_encoder_states is not None:
                gt_encoder_states = gt_encoder_states[:, positions, :]
            
        # Store steps
        for step_diag in trajectory.steps:
            step_idx = step_diag.step_idx
            if step_idx in self.store.steps:
                x_hat = step_diag.x_hat_val.to(z_final.device)
                z_val = step_diag.z_val.to(z_final.device)
                
                if positions is not None:
                    x_hat = x_hat[:, positions, :]
                    z_val = z_val[:, positions, :]
                    
                r_val = x_hat - x_ref
                
                # Write to memmap
                self.store.maps[step_idx]["x_hat"].write(start_idx, x_hat.cpu().numpy())
                self.store.maps[step_idx]["z"].write(start_idx, z_val.cpu().numpy())
                self.store.maps[step_idx]["r"].write(start_idx, r_val.cpu().numpy())
                
        # Store globals
        self.store.final_tokens.write(start_idx, final_ids.cpu().numpy())
        if gt_tokens is not None:
            self.store.gt_tokens.write(start_idx, gt_tokens.cpu().numpy())
        if gt_encoder_states is not None:
            self.store.gt_encoder_states.write(start_idx, gt_encoder_states.cpu().numpy())
