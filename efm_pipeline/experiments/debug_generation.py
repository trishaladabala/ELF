#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.utils import get_device

def main():
    device = get_device()
    dataset = OrderedAssemblyDataset(num_samples=4, seq_len=64, vocab_size=256, embed_dim=512, W=4, seed=42)
    
    model = EFM_Tiny(vocab_size=256, local_time_mode='continuous').to(device)
    ckpt = torch.load('tiny_pilot_W4/ckpt/tiny_W4/checkpoint_5000.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    bs = 4
    seq_len = 64
    z = torch.randn(bs, seq_len, 512, device=device)
    
    times = dataset.wave_ids[:bs].to(device).float() / 3.0
    targets = dataset.token_ids[:bs].to(device)
    
    num_steps = 128  # finer schedule
    dt = 1.0 / num_steps
    
    print("t\tExact Match of x0_pred")
    print("-" * 30)
    
    M = seq_len // 4
    num_keys = (4 - 1) * M
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((bs,), t_val, device=device)
            
            # Teacher-force the keys: set the key positions to the true intermediate state (or just pure x0)
            # Actually, the true trajectory is z_t = t * x0 + (1-t) * eps. 
            # We can just overwrite the z for the keys with the exact ODE path.
            z[:, :num_keys] = t_val * dataset.embeddings[:bs, :num_keys].to(device) + (1 - t_val) * torch.randn(bs, num_keys, 512, device=device)
            
            x0_pred, logits = model(z, t, local_times=times, decoder_step_active=True)
            
            # ODE Step
            v = (x0_pred - z) / max(1.0 - t_val, 1e-4)
            z = z + v * dt

        # Final decode
        t_final = torch.ones(bs, device=device)
        # Teacher-force at t=1
        z[:, :num_keys] = dataset.embeddings[:bs, :num_keys].to(device)
        
        _, final_logits = model(z, t_final, local_times=times, decoder_step_active=True)
        final_preds = final_logits.argmax(dim=-1)
        final_acc = (final_preds[:, num_keys:] == targets[:, num_keys:]).float().mean().item()
        print(f"1.00\t{final_acc:.4f} (Final generated value accuracy with TF Keys)")

if __name__ == "__main__":
    main()
