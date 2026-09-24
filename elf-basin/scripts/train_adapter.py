#!/usr/bin/env python3
"""Phase 4: Train decoder adapter variants (LoRA-based).

Trains 6 adapter variants on harvested trajectory states, plus evaluates
the frozen native decoder as variant 0. Uses LoRA on attention projections
instead of a small bottleneck, for much more expressive adaptation.
"""
import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.train.adapter import DecoderAdapter
from elfbasin.train.loss import compute_loss


def train_variant(variant_id, wrapper, train_loader, val_loader, config, num_epochs=15, lr=3e-4):
    device = wrapper.device
    adapter = DecoderAdapter(wrapper, r=256).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine LR schedule
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    print(f"\n--- Training Variant {variant_id} ({adapter.count_parameters()} trainable params) ---")
    
    lambda_sens = 0.0
    lambda_cons = 0.0
    use_clean_state = False
    use_iso_noise = False
    use_hinge = False
    
    if variant_id == 1:
        use_clean_state = True
    elif variant_id == 2:
        use_iso_noise = True
    elif variant_id == 3:
        use_hinge = True
    elif variant_id == 4:
        pass # default is trajectory-residual CE
    elif variant_id == 5:
        lambda_sens = 0.1 # placeholder weight
    elif variant_id == 6:
        lambda_sens = 0.1
        lambda_cons = 0.5
        
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0
    
    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_hat, r, gt_enc, gt_toks = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Select base state and noise
            if use_clean_state:
                base_x = gt_enc
                noise1, noise2, v_dir = torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r)
            elif use_iso_noise:
                base_x = x_hat
                noise1 = torch.randn_like(r) * r.norm(dim=-1, keepdim=True).mean()
                noise2 = torch.randn_like(r) * r.norm(dim=-1, keepdim=True).mean()
                v_dir = torch.randn_like(r)
                v_dir = v_dir / (v_dir.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                base_x = x_hat
                noise1 = r
                # In practice we can just shuffle the batch for r_prime.
                noise2 = r[torch.randperm(r.size(0))]
                # v is drawn from the trajectory direction distribution (we use r)
                v_dir = r
                
            if use_hinge:
                # 3. Scalar hinge margin on m
                logits = adapter(base_x + noise1)
                top2 = logits.topk(2, dim=-1).values
                margin = top2[:, :, 0] - top2[:, :, 1]
                # maximize margin -> minimize max(0, tau - margin)
                loss = torch.nn.functional.relu(10.0 - margin).mean()
            else:
                # compute_loss
                loss, ce, sens, cons = compute_loss(
                    adapter, base_x, noise1, noise2, v_dir, gt_toks, 
                    lambda_sens, lambda_cons
                )
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Early stopping on val loss
        adapter.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_hat, r, gt_enc, gt_toks = [b.to(device) for b in batch]
                if use_clean_state:
                    base_x = gt_enc
                    noise1 = torch.zeros_like(r)
                elif use_iso_noise:
                    base_x = x_hat
                    noise1 = torch.randn_like(r) * r.norm(dim=-1, keepdim=True).mean()
                else:
                    base_x = x_hat
                    noise1 = r
                logits = adapter(base_x + noise1)
                import torch.nn.functional as F
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gt_toks.reshape(-1), ignore_index=0)
                val_loss += ce.item()
        val_loss /= max(1, len(val_loader))
        print(f"  Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
    return adapter

def main():
    print("="*70)
    print("  Phase 4: Adapter Training (LoRA)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = ELFWrapper("ELF-B-de-en", device=device)
    
    RESULTS_DIR = _SCRIPT_DIR.parent / "runs" / "harvest_deen"
    if not RESULTS_DIR.exists():
        print("Harvested data not found. Run harvest_deen.py first.")
        return 1
        
    # We will train on harvested states from a mid/late step, e.g., step 48
    step = 48
    step_dir = RESULTS_DIR / f"step_{step}"
    
    print(f"Loading harvested states for step {step}...")
    x_hat_data = np.lib.format.open_memmap(step_dir / "x_hat.npy", mode='r')
    r_data = np.lib.format.open_memmap(step_dir / "r.npy", mode='r')
    gt_enc_data = np.lib.format.open_memmap(RESULTS_DIR / "gt_encoder_states.npy", mode='r')
    gt_tokens_data = np.lib.format.open_memmap(RESULTS_DIR / "gt_tokens.npy", mode='r')
    
    # Use first 2000 samples for training, rest for validation
    train_end = 2000
    
    def to_tensor(data, start, end, dtype=torch.float32):
        return torch.from_numpy(np.array(data[start:end])).to(dtype)
        
    train_dataset = TensorDataset(
        to_tensor(x_hat_data, 0, train_end),
        to_tensor(r_data, 0, train_end),
        to_tensor(gt_enc_data, 0, train_end),
        to_tensor(gt_tokens_data, 0, train_end, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        to_tensor(x_hat_data, train_end, 3000),
        to_tensor(r_data, train_end, 3000),
        to_tensor(gt_enc_data, train_end, 3000),
        to_tensor(gt_tokens_data, train_end, 3000, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    out_dir = _SCRIPT_DIR.parent / "runs" / "train_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Variant 0 is frozen native decoder (no training)
    # We will train Variants 1 through 6
    # IMPORTANT: Each variant needs a fresh model since LoRA modifies in-place
    for v in range(1, 7):
        # Reload wrapper for each variant to get clean weights
        wrapper = ELFWrapper("ELF-B-de-en", device=device)
        adapter = train_variant(v, wrapper, train_loader, val_loader, wrapper.config)
        torch.save(adapter.state_dict(), out_dir / f"adapter_v{v}.pt")
        
    print("\nTraining complete. Models saved to", out_dir)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
