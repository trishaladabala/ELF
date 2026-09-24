#!/usr/bin/env python3
"""Unit test for margin and ρ computations against hand-computed toy cases.

Tests:
1. Known logits → correct margin, top1/top2 IDs
2. FD estimator on a linear decode function → matches analytical Jacobian
3. ρ = m / L computation
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ELFBASIN = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_ELFBASIN))

from elfbasin.sampling.sensitivity import FDEstimator, ProbeDirectionGenerator


def test_margin_basic():
    """Test margin computation on known logits."""
    print("  [1/3] Testing margin on known logits...")

    # 3-token vocabulary, known logits
    logits = torch.tensor([[[5.0, 3.0, 1.0],    # position 0: margin = 2.0
                            [10.0, 9.5, 0.0],   # position 1: margin = 0.5
                            [1.0, 1.0, 5.0]]])   # position 2: margin = 4.0 (top1=2, top2 tie)
    # B=1, L=3, V=3

    top2 = logits.topk(2, dim=-1)
    margin = top2.values[:, :, 0] - top2.values[:, :, 1]

    expected_margins = torch.tensor([[2.0, 0.5, 4.0]])
    expected_top1 = torch.tensor([[0, 0, 2]])
    expected_top2 = torch.tensor([[1, 1, 0]])  # second token of position 2 could be 0 or 1

    assert torch.allclose(margin, expected_margins), \
        f"Margin mismatch: {margin} != {expected_margins}"
    assert torch.equal(top2.indices[:, :, 0], expected_top1), \
        f"Top1 mismatch: {top2.indices[:,:,0]} != {expected_top1}"

    # Position 2: top2 could be 0 or 1 (both have logit 1.0)
    assert top2.indices[0, 2, 1] in [0, 1], \
        f"Top2 at position 2 should be 0 or 1, got {top2.indices[0, 2, 1]}"

    print(f"    Margins: {margin.squeeze().tolist()}")
    print(f"    Top1 IDs: {top2.indices[:,:,0].squeeze().tolist()}")
    print(f"    ✓ Margin computation correct")
    return True


def test_fd_estimator_linear():
    """Test FD estimator on a linear decode function.

    For a linear decoder g(x) = W @ x, the true Jacobian is W.
    The margin m(x) = (W[top1] - W[top2]) @ x
    The directional derivative dm/dv = (W[top1] - W[top2]) @ v
    So L_i = |(W[top1] - W[top2]) @ v|

    FD should approximate this exactly (up to numerical precision).
    """
    print("\n  [2/3] Testing FD estimator on linear decode...")

    B, L, D, V = 1, 2, 4, 3
    torch.manual_seed(42)

    # Fixed linear decode: g(x) = x @ W.T
    W = torch.randn(V, D)  # (V, D)

    def linear_decode(x):
        # x: (B, L, D) -> logits: (B, L, V)
        return x @ W.T

    # Input
    x_hat = torch.randn(B, L, D)
    logits = linear_decode(x_hat)
    top2 = logits.topk(2, dim=-1)
    top1_ids = top2.indices[:, :, 0]
    top2_ids = top2.indices[:, :, 1]

    # Analytical L_i for a specific direction
    v = torch.randn(B, L, D)
    v = F.normalize(v, dim=-1)

    # True directional derivative of margin
    # dm/dv = (W[top1] - W[top2]) @ v, per position
    analytical_L = torch.zeros(B, L)
    for b in range(B):
        for l in range(L):
            w_diff = W[top1_ids[b, l]] - W[top2_ids[b, l]]
            analytical_L[b, l] = abs(float(w_diff @ v[b, l]))

    # FD estimate
    estimator = FDEstimator(h=1e-4)  # small h for accuracy
    fd_L = estimator.estimate(x_hat, linear_decode, top1_ids, top2_ids, [v])

    error = (fd_L - analytical_L).abs()
    max_error = error.max().item()
    print(f"    Analytical L_i: {analytical_L.squeeze().tolist()}")
    print(f"    FD L_i:         {fd_L.squeeze().tolist()}")
    print(f"    Max error:      {max_error:.6e}")

    assert max_error < 1e-2, f"FD error too large: {max_error:.6e}"
    print(f"    ✓ FD estimator matches analytical Jacobian (error < 1e-2)")
    return True


def test_rho_computation():
    """Test ρ = m / L computation."""
    print("\n  [3/3] Testing ρ = m/L computation...")

    margin = torch.tensor([[2.0, 0.5, 4.0]])
    L_i = torch.tensor([[1.0, 0.25, 2.0]])

    rho = margin / L_i.clamp(min=1e-6)
    expected_rho = torch.tensor([[2.0, 2.0, 2.0]])

    assert torch.allclose(rho, expected_rho), \
        f"ρ mismatch: {rho} != {expected_rho}"

    # Test with near-zero L_i (should be clamped)
    L_i_zero = torch.tensor([[0.0, 1e-7, 1.0]])
    rho_safe = margin / L_i_zero.clamp(min=1e-6)
    assert torch.isfinite(rho_safe).all(), "ρ should be finite with clamping"

    print(f"    ρ values: {rho.squeeze().tolist()}")
    print(f"    ✓ ρ computation correct (including zero-L_i safety)")
    return True


def main():
    print("="*70)
    print("  Margin and ρ Unit Tests")
    print("="*70)

    results = []
    results.append(("margin_basic", test_margin_basic()))
    results.append(("fd_linear", test_fd_estimator_linear()))
    results.append(("rho_computation", test_rho_computation()))

    print(f"\n{'='*70}")
    all_pass = all(r[1] for r in results)
    for name, passed in results:
        print(f"  {'✓' if passed else '✗'} {name}")

    if all_pass:
        print("\n  ✓ All margin/ρ tests passed.")
    else:
        print("\n  ✗ Some tests failed.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
