#!/usr/bin/env python3
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from efm_pipeline.efm_model import EFM_Tiny
from efm_pipeline.synthetic.ordered_assembly import OrderedAssemblyDataset
from efm_pipeline.utils import get_device

def main():
    device = get_device()
    dataset = OrderedAssemblyDataset(num_samples=100, seq_len=64, vocab_size=256, embed_dim=512, W=4, seed=42)
    model = EFM_Tiny(vocab_size=256, local_time_mode='continuous').to(device)
    model.load_state_dict(torch.load('tiny_pilot_W4/ckpt/tiny_W4/checkpoint_5000.pt', map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    bs = 100
    x0 = dataset.embeddings[:bs].to(device)
    eps = torch.randn_like(x0)
    times = dataset.wave_ids[:bs].to(device).float() / 3.0
    targets = dataset.token_ids[:bs].to(device)

    print("t\tTF Acc")
    print("-" * 15)
    with torch.no_grad():
        for t_val in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
            t = torch.full((bs,), t_val, device=device)
            z_t = t_val * x0 + (1 - t_val) * eps
            _, logits = model(z_t, t, local_times=times, decoder_step_active=True)
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            print(f"{t_val:.2f}\t{acc:.4f}")

if __name__ == "__main__":
    main()

