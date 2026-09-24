import sys
import os
import torch
import torch.nn.functional as F

# Add project root and efm_pipeline dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from efm_pipeline.efm_model import EFM_Small

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def check_gradients(conditioner, tau):
    # reset grads
    for p in conditioner.parameters():
        p.grad = None
        
    emb = conditioner(tau)
    loss = emb.sum()
    loss.backward()
    
    has_grad = False
    for p in conditioner.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            has_grad = True
            break
    return has_grad

def run_checks():
    print("# Validation Representation Checks\n")
    
    # 1. Initialize models
    models = {
        "Continuous": EFM_Small(local_time_mode="continuous"),
        "Quantized": EFM_Small(local_time_mode="quantized", local_time_K=4),
        "Lowrank": EFM_Small(local_time_mode="lowrank", lowrank_basis="learned", local_time_K=4)
    }
    
    tau_vals = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32)
    
    for name, model in models.items():
        print(f"## Model: {name}\n")
        
        conditioner = model.local_time_conditioner
        
        dims = conditioner.num_conditioning_dims()
        params = count_parameters(conditioner)
        print(f"- **Conditioning Dimensions:** {dims}")
        print(f"- **Trainable Parameters:** {params}")
        
        emb = conditioner(tau_vals)
        
        l2_norm = torch.norm(emb, p=2, dim=-1)
        
        # Round the L2 norms for cleaner printing
        l2_norms_list = [round(val, 4) for val in l2_norm.squeeze(0).detach().numpy().tolist()]
        print(f"- **L2 Norms:** {l2_norms_list}")
        
        emb_squeezed = emb.squeeze(0) # (5, D)
        # Compute cosine similarity
        emb_normalized = F.normalize(emb_squeezed, p=2, dim=1)
        cos_sim = torch.mm(emb_normalized, emb_normalized.t())
        
        print(f"- **Cosine Similarity Matrix:**")
        print("```")
        import numpy as np
        np.set_printoptions(precision=4, suppress=True)
        print(cos_sim.detach().numpy())
        print("```")
        
        grad_flows = check_gradients(conditioner, tau_vals)
        print(f"- **Gradient Flows:** {grad_flows}\n")
        
        if name == "Quantized":
            print(f"### Specific checks for {name}\n")
            # Test that tau=1.0 maps to correct bin K-1
            tau_one = torch.tensor([[1.0]], dtype=torch.float32)
            try:
                emb_one = conditioner(tau_one)
                print(f"- `tau=1.0` processed successfully without out-of-bounds error.")
            except Exception as e:
                print(f"- `tau=1.0` processing failed: {e}")
            print("\n")

if __name__ == "__main__":
    run_checks()
