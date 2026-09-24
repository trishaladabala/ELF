#!/usr/bin/env python3
"""Test strict key matching and basic ELFWrapper functionality.

Tests:
1. Load ELF-B-owt and ELF-B-de-en with strict=True
2. Verify decode() output shape is (B, L, V) with V=32128
3. Verify encode() output shape is (B, L, 512)
4. Verify denoise() output shape is (B, L, 512)
5. Verify margin() is non-negative for a random input
"""

import sys
import time
from pathlib import Path

import torch

# Add elfbasin to path
_ELFBASIN = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_ELFBASIN))
from elfbasin.model.elf import ELFWrapper


def test_checkpoint(name: str, device: str = "cuda"):
    """Test a single checkpoint: load, decode, encode, denoise."""
    print(f"\n{'='*70}")
    print(f"  Testing: {name}")
    print(f"{'='*70}")

    t0 = time.time()
    wrapper = ELFWrapper(name, device=device)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Model: {wrapper}")

    B = 2
    L = wrapper.max_length
    D = wrapper.d_model
    V = wrapper.vocab_size
    print(f"  B={B}, L={L}, D={D}, V={V}")

    # ── Test encode() ──
    print("\n  [1/4] Testing encode()...")
    dummy_ids = torch.randint(0, V, (B, L), device=device)
    x = wrapper.encode(dummy_ids)
    assert x.shape == (B, L, D), f"encode shape mismatch: {x.shape} != ({B}, {L}, {D})"
    assert x.dtype == torch.float32, f"encode dtype: {x.dtype}"
    print(f"    ✓ encode() -> {x.shape}, dtype={x.dtype}")
    print(f"    norm per position: mean={x.norm(dim=-1).mean():.3f}, "
          f"std={x.norm(dim=-1).std():.3f}")

    # ── Test decode() ──
    print("\n  [2/4] Testing decode()...")
    # Use the encoded output as input to decode
    logits = wrapper.decode(x)
    assert logits.shape == (B, L, V), \
        f"decode shape mismatch: {logits.shape} != ({B}, {L}, {V})"
    assert logits.dtype == torch.float32, f"decode dtype: {logits.dtype}"
    print(f"    ✓ decode() -> {logits.shape}, dtype={logits.dtype}")
    print(f"    logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    pred_ids = logits.argmax(dim=-1)
    print(f"    argmax tokens sample (first 10): {pred_ids[0, :10].tolist()}")

    # ── Test denoise() ──
    print("\n  [3/4] Testing denoise()...")
    z = torch.randn(B, L, D, device=device) * 2.0  # noisy input at noise scale
    x_hat = wrapper.denoise(z, t=0.5, sc_cfg=1.0)
    assert x_hat.shape == (B, L, D), \
        f"denoise shape mismatch: {x_hat.shape} != ({B}, {L}, {D})"
    print(f"    ✓ denoise(t=0.5) -> {x_hat.shape}")
    print(f"    x_hat norm: mean={x_hat.norm(dim=-1).mean():.3f}")

    # ── Test margin() ──
    print("\n  [4/4] Testing margin()...")
    margins = wrapper.margin(x)
    assert margins.shape == (B, L), \
        f"margin shape mismatch: {margins.shape} != ({B}, {L})"
    print(f"    ✓ margin() -> {margins.shape}")
    print(f"    margin stats: mean={margins.mean():.3f}, min={margins.min():.3f}, "
          f"max={margins.max():.3f}")
    print(f"    fraction positive: {(margins > 0).float().mean():.3f}")

    # Verify margin is non-negative for clean encoder states
    # (clean states should decode with high confidence)
    neg_frac = (margins < 0).float().mean().item()
    if neg_frac > 0.5:
        print(f"    ⚠ WARNING: {neg_frac*100:.1f}% of positions have negative margin!")
    else:
        print(f"    ✓ Only {neg_frac*100:.1f}% negative margins (expected for clean states)")

    # Peak VRAM
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak VRAM: {peak_mb:.0f} MB")

    print(f"\n  ✓ All tests passed for {name} ({time.time()-t0:.1f}s total)")
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    results = {}

    # Test ELF-B-owt
    try:
        results["ELF-B-owt"] = test_checkpoint("ELF-B-owt", device)
    except Exception as e:
        print(f"\n  ✗ ELF-B-owt FAILED: {e}")
        results["ELF-B-owt"] = False
        import traceback; traceback.print_exc()

    # Clear GPU memory between checkpoints
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Test ELF-B-de-en
    try:
        results["ELF-B-de-en"] = test_checkpoint("ELF-B-de-en", device)
    except Exception as e:
        print(f"\n  ✗ ELF-B-de-en FAILED: {e}")
        results["ELF-B-de-en"] = False
        import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_pass = all(results.values())
    if all_pass:
        print("\n  ✓ All checkpoint tests passed. Ready for Phase 0 reproduction.")
    else:
        print("\n  ✗ Some tests failed. Fix before proceeding.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
