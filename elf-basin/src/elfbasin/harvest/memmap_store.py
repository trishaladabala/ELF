import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class MemmapStore:
    def __init__(self, output_dir: str, name: str, shape: Tuple[int, ...], dtype=np.float16, mode='w+'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{name}.npy"
        self.shape = shape
        self.dtype = dtype
        
        # Pre-allocate if writing
        if mode == 'w+':
            self.data = np.lib.format.open_memmap(self.path, mode='w+', dtype=dtype, shape=shape)
        else:
            self.data = np.lib.format.open_memmap(self.path, mode=mode)
            
    def write(self, start_idx: int, data: np.ndarray):
        end_idx = start_idx + data.shape[0]
        self.data[start_idx:end_idx] = data
        self.data.flush()
        
    def read(self, start_idx: int, end_idx: int) -> np.ndarray:
        return self.data[start_idx:end_idx]

class TrajectoryStore:
    """Manages memmaps for multiple steps and fields."""
    def __init__(self, output_dir: str, num_samples: int, seq_len: int, d_model: int, steps: List[int], mode='w+'):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.d_model = d_model
        self.steps = steps
        self.mode = mode
        self.maps = {}
        
        # Shape per step: (num_samples, seq_len, d_model)
        shape_fp16 = (num_samples, seq_len, d_model)
        
        for step in steps:
            step_dir = self.output_dir / f"step_{step}"
            self.maps[step] = {
                "x_hat": MemmapStore(step_dir, "x_hat", shape_fp16, np.float16, mode),
                "z": MemmapStore(step_dir, "z", shape_fp16, np.float16, mode),
                "r": MemmapStore(step_dir, "r", shape_fp16, np.float16, mode)
            }
            
        # Metadata / globals
        self.final_tokens = MemmapStore(self.output_dir, "final_tokens", (num_samples, seq_len), np.int16, mode)
        self.gt_tokens = MemmapStore(self.output_dir, "gt_tokens", (num_samples, seq_len), np.int16, mode)
        self.gt_encoder_states = MemmapStore(self.output_dir, "gt_encoder_states", shape_fp16, np.float16, mode)
        
    def check_disk_budget(self) -> float:
        """Returns projected size in GB."""
        fp16_bytes = 2
        int16_bytes = 2
        
        per_step_bytes = self.num_samples * self.seq_len * self.d_model * fp16_bytes * 3 # x_hat, z, r
        total_steps_bytes = len(self.steps) * per_step_bytes
        
        globals_bytes = (self.num_samples * self.seq_len * int16_bytes * 2) + \
                        (self.num_samples * self.seq_len * self.d_model * fp16_bytes)
                        
        total_gb = (total_steps_bytes + globals_bytes) / (1024**3)
        return total_gb
