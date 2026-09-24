"""Shared utilities: seed fixing, device detection, memory monitoring."""

import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: The random seed to use everywhere.
        deterministic: If True, enable PyTorch deterministic mode.  Note that
            some operations (e.g. certain CUDA kernels) may be slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some ops have no deterministic implementation; allow fallback.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't support warn_only.
            torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(prefer: Optional[str] = None) -> torch.device:
    """Auto-detect the best available device.

    Priority: CUDA → MPS → CPU  (unless *prefer* overrides).
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    """Return a human-readable device summary string."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"{name} ({mem:.1f} GB)"
    if device.type == "mps":
        return "Apple MPS (Metal Performance Shaders)"
    return "CPU"


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def gpu_memory_mb(device: Optional[torch.device] = None) -> dict:
    """Return current GPU memory usage in MB (CUDA only)."""
    if device is None:
        device = get_device()
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1e6,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
    }


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak memory stats."""
    if device is None:
        device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------

def amp_autocast(device: torch.device, dtype=torch.float16, enabled: bool = True):
    """Return the correct autocast context for the given device.

    - CUDA: use torch.amp.autocast('cuda', dtype=dtype)
    - MPS / CPU: autocast is either unsupported or counterproductive; return
      a no-op context manager.
    """
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextmanager(lambda: (yield))()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    def reset(self):
        self._start = None
        self._elapsed = 0.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> str:
    """Return the 7semminor project root directory."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def elf_src_root() -> str:
    """Return the ELF/src directory path."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
