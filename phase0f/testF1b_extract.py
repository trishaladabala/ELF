#!/usr/bin/env python3
import json
import sys
import numpy as np
import torch
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase0t.testT5_rseries_indistribution import load_shared_model
from phase0i.testI2_final_result import ode_sample

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_DIR = Path(__file__).resolve().parent.parent

def compute_turning_angles(trajectory):
    """
    trajectory: numpy array of shape (steps+1, N, seq_len, dim)
    Returns curvature array of shape (N,)
    """
    v = np.diff(trajectory, axis=0)
    steps, N, seq_len, dim = v.shape
    if steps < 2:
        return np.zeros(N)
    
    v_prev = v[:-1]
    v_curr = v[1:]
    
    dot = np.sum(v_prev * v_curr, axis=-1)
    norm_prev = np.linalg.norm(v_prev, axis=-1)
    norm_curr = np.linalg.norm(v_curr, axis=-1)
    
    cos_sim = dot / (norm_prev * norm_curr + 1e-12)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    angles = np.arccos(cos_sim)
    curvature = np.mean(angles, axis=(0, 2))
    return curvature

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load seed groups
    group_path = RESULTS_DIR / "testF1a_seed_groups.json"
    with open(group_path, "r") as f:
        groups = json.load(f)
        
    n_samples = 256
    seq_len = 16
    encoder_dim = 128
    batch_size = 64
    n_steps = 8
    
    ctrl_path = BASE_DIR / "phase0r" / "results" / "testR2_model_control.pt"
    ctrl_model = load_shared_model(ctrl_path)
    
    print("Generating trajectories...")
    traj_all = []
    
    for start_idx in range(0, n_samples, batch_size):
        bsz = min(batch_size, n_samples - start_idx)
        torch.manual_seed(42 + start_idx)
        noise = torch.randn(bsz, seq_len, encoder_dim, dtype=torch.float32, device=DEVICE)
        
        _, t_ode = ode_sample(ctrl_model, bsz, n_steps, encoder_dim, seq_len,
                              time_schedule="logit_normal", device=DEVICE, initial_noise=noise,
                              return_trajectory=True)
        traj_all.append(t_ode)
        
    # Combine trajectories: t_ode is (steps+1, batch, seq_len, dim)
    trajectory = np.concatenate(traj_all, axis=1)
    
    print("Computing curvatures...")
    curvatures = compute_turning_angles(trajectory)
    
    print("Computing noise magnitudes...")
    # initial_noise is at step 0: (N, seq_len, dim)
    initial_noise = trajectory[0]
    # compute L2 norm per sample
    noise_mags = np.linalg.norm(initial_noise.reshape(n_samples, -1), axis=1)
    
    # Map to groups
    out_data = {}
    for group_name in ["always_fails", "sometimes_fails", "never_fails"]:
        indices = groups[group_name]
        out_data[group_name] = {
            "indices": indices,
            "curvatures": curvatures[indices].tolist(),
            "noise_mags": noise_mags[indices].tolist()
        }
        
    out_path = RESULTS_DIR / "testF1b_curvature_by_group.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
