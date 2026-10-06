#!/usr/bin/env python3
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.utils import get_device

def evaluate_internal_consistency():
    device = get_device()
    W = 4
    dataset = OrderedAssemblyDataset(num_samples=100, seq_len=64, vocab_size=256, embed_dim=512, W=W, seed=42)
    
    model = EFM_Tiny(vocab_size=256, local_time_mode='continuous').to(device)
    ckpt = torch.load('tiny_pilot_W4/ckpt/tiny_W4/checkpoint_5000.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    bs = 100
    seq_len = 64
    z = torch.randn(bs, seq_len, 512, device=device)
    times = dataset.wave_ids[:bs].to(device).float() / max(W - 1, 1)
    
    num_steps = 128
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((bs,), t_val, device=device)
            x0_pred, _ = model(z, t, local_times=times, decoder_step_active=False)
            v = (x0_pred - z) / max(1.0 - t_val, 1e-4)
            z = z + v * dt

        t_final = torch.ones(bs, device=device)
        _, final_logits = model(z, t_final, local_times=times, decoder_step_active=True)
        generated_ids = final_logits.argmax(dim=-1)
    
    M = seq_len // W
    num_keys = (W - 1) * M
    
    # Internal Consistency
    total_correct = 0
    total_count = 0
    for w in range(1, W):
        val_waves = dataset.wave_ids[:bs, num_keys:]
        mask_w = (val_waves == w)
        count = mask_w.sum().item()
        if count > 0:
            gen_keys = generated_ids[:, (w-1)*M : w*M]
            gen_vals = generated_ids[:, num_keys:]
            matches = (gen_vals == gen_keys) & mask_w.to(device)
            total_correct += matches.sum().item()
            total_count += count
    
    internal_acc = total_correct / total_count
    
    # Mode collapse check (entropy of generated keys)
    keys_flat = generated_ids[:, :num_keys].flatten()
    unique_tokens = torch.unique(keys_flat)
    
    print(f"Internal Consistency (gen_vals == gen_keys): {internal_acc:.4f}")
    print(f"Number of unique tokens generated in Keys: {len(unique_tokens)} / 256")
    print("Sample generated keys:", keys_flat[:16].tolist())

if __name__ == "__main__":
    evaluate_internal_consistency()

