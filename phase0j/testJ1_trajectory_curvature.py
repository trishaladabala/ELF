#!/usr/bin/env python3
"""J1 — Extract Per-Sample Trajectory Curvature.

Reuses the trajectories saved from I2 (8 steps, logit_normal) and computes
a local curvature proxy (average turning angle between successive steps)
for each sample.

Also computes a global intrinsic dimensionality estimate of the denoising
trajectory manifold.
"""

import json
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
I_RESULTS = Path(__file__).resolve().parent.parent / "phase0i" / "results"

def compute_turning_angles(trajectory):
    """
    trajectory: numpy array of shape (steps+1, N, seq_len, dim)
    Returns curvature array of shape (N,)
    """
    # Compute step vectors v_t = z_{t+1} - z_t
    # Shape: (steps, N, seq_len, dim)
    v = np.diff(trajectory, axis=0)
    
    steps, N, seq_len, dim = v.shape
    if steps < 2:
        return np.zeros(N)
    
    # We want angle between v_{t} and v_{t-1}
    # For each t from 1 to steps-1
    v_prev = v[:-1] # (steps-1, N, seq_len, dim)
    v_curr = v[1:]  # (steps-1, N, seq_len, dim)
    
    # Dot product along dim
    dot = np.sum(v_prev * v_curr, axis=-1) # (steps-1, N, seq_len)
    
    norm_prev = np.linalg.norm(v_prev, axis=-1)
    norm_curr = np.linalg.norm(v_curr, axis=-1)
    
    # Cosine similarity
    cos_sim = dot / (norm_prev * norm_curr + 1e-12)
    # Clip for numerical stability
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Angle in radians
    angles = np.arccos(cos_sim) # (steps-1, N, seq_len)
    
    # Average across steps and seq_len
    # Resulting shape: (N,)
    curvature = np.mean(angles, axis=(0, 2))
    return curvature


def estimate_manifold_dim(trajectory):
    """
    Estimate intrinsic dimensionality of the trajectory manifold.
    Flatten trajectory to (steps+1 * N, seq_len * dim)
    """
    steps_plus_one, N, seq_len, dim = trajectory.shape
    x = trajectory.reshape(steps_plus_one * N, seq_len * dim)
    
    # Global PCA on the manifold
    centered = x - x.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    
    total_var = (S**2).sum()
    cumvar = np.cumsum(S**2) / total_var
    dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    
    return {
        "pca_dim_90pct": dim_90,
        "pca_dim_95pct": dim_95,
        "ambient_dim": seq_len * dim,
        "total_points": x.shape[0]
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("J1 — Extract Trajectory Curvature")
    print("=" * 60)

    try:
        ode_traj = np.load(I_RESULTS / "testI2_ode_trajectories.npy")
        sde_traj = np.load(I_RESULTS / "testI2_sde_trajectories.npy")
    except FileNotFoundError:
        print("Error: Could not find trajectory files from I2. Run testI2 first.")
        return 1

    print(f"Loaded ODE trajectory shape: {ode_traj.shape}")
    print(f"Loaded SDE trajectory shape: {sde_traj.shape}")
    
    N = ode_traj.shape[1]
    
    # Compute per-sample curvature
    # We primarily care about the ODE (deterministic path) curvature as a predictor
    # of where SDE helps.
    ode_curvature = compute_turning_angles(ode_traj)
    sde_curvature = compute_turning_angles(sde_traj)
    
    print(f"ODE Curvature Mean: {ode_curvature.mean():.4f}, Std: {ode_curvature.std():.4f}")
    print(f"SDE Curvature Mean: {sde_curvature.mean():.4f}, Std: {sde_curvature.std():.4f}")
    
    # Global manifold estimates
    ode_manifold = estimate_manifold_dim(ode_traj)
    print("ODE Trajectory Manifold PCA Dims:")
    print(f"  90% var: {ode_manifold['pca_dim_90pct']} / {ode_manifold['ambient_dim']}")
    print(f"  95% var: {ode_manifold['pca_dim_95pct']} / {ode_manifold['ambient_dim']}")
    
    sde_manifold = estimate_manifold_dim(sde_traj)
    
    results = {
        "ode_curvature": ode_curvature.tolist(),
        "sde_curvature": sde_curvature.tolist(),
        "ode_manifold": ode_manifold,
        "sde_manifold": sde_manifold
    }
    
    out_path = RESULTS_DIR / "testJ1_trajectory_curvature.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n  Saved {len(ode_curvature)} curvature values per method to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
