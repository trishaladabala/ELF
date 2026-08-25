import torch
import sys
from pathlib import Path
from testT5_rseries_indistribution import load_shared_model, generate_texts
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent

model = load_shared_model(BASE_DIR / "phase0r" / "results" / "testR2_model_control.pt")
texts = generate_texts(model, n_samples=5, n_steps=8, seq_len=16, time_schedule="logit_normal")
for i, t in enumerate(texts):
    print(f"Sample {i+1}: '{t}'")
