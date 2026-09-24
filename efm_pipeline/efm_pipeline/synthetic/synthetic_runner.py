#!/usr/bin/env python3
"""Runner utilities for generating and evaluating synthetic EFM tasks."""
import torch
import torch.nn.functional as F
from tqdm import auto

from efm_pipeline.utils import get_device


@torch.no_grad()
def generate_synthetic_samples(
    model,
    dataset,
    device: torch.device,
    num_samples: int = 512,
    num_steps: int = 32,
    batch_size: int = 64,
    seed: int = 42,
    local_time_mode: str = "expansion",
    W: int = 4,
):
    """Generate raw token IDs from the model for synthetic evaluation.
    
    Uses standard ODE sampling with Euler method.
    """
    model.eval()
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)

    all_token_ids = []
    
    seq_len = dataset.token_ids.shape[1]
    embed_dim = dataset.embed_dim
    vocab_size = dataset.vocab_size
    
    # We need the ground truth wave assignments (assumed deterministic)
    wave_template = dataset.wave_ids[0].to(device)

    num_batches = (num_samples + batch_size - 1) // batch_size

    for b_idx in auto.tqdm(range(num_batches), desc=f"Sampling ({local_time_mode})"):
        bs = min(batch_size, num_samples - b_idx * batch_size)
        
        # Start from pure noise
        z = torch.randn(bs, seq_len, embed_dim, device=device, generator=generator)
        
        # Local time logic
        if local_time_mode == "expansion":
            # Oracle / true timing
            local_times = wave_template.unsqueeze(0).expand(bs, -1).float() / max(W - 1, 1)
        elif local_time_mode == "none":
            local_times = None
        else:
            # Fallback (e.g. constant 0)
            local_times = torch.zeros(bs, seq_len, device=device)

        # ODE integration (t from 0 to 1)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((bs,), step * dt, device=device)
            with torch.amp.autocast('cuda', enabled=False):
                # Predict clean x0
                x0_pred, _ = model(z, t, local_times=local_times, decoder_step_active=False)
                # Compute velocity: v = x0 - noise (assuming standard flow matching formulation)
                # Here z_t = t * x0 + (1-t) * eps, so v = x0 - eps.
                # v = (x0_pred - z) / (1 - t + 1e-5)
                v = (x0_pred - z) / max(1.0 - step * dt, 1e-4)
                z = z + v * dt

        # Final decoding
        t_final = torch.ones(bs, device=device)
        with torch.amp.autocast('cuda', enabled=False):
            _, logits = model(z, t_final, local_times=local_times, decoder_step_active=True)
            token_ids = logits.argmax(dim=-1)
        
        all_token_ids.append(token_ids.cpu())

    return torch.cat(all_token_ids, dim=0)


