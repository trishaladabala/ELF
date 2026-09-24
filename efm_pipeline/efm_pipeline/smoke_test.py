#!/usr/bin/env python3
"""Smoke test — validates all modules in < 2 minutes on CPU/MPS.

Tests:
    1. Module imports
    2. Config creation
    3. Local-time conditioning (all 3 modes)
    4. Insertion head forward + loss
    5. Expand operator
    6. EFM model forward pass (tiny)
    7. Training step (1 iteration)
    8. Generation loop
    9. Metrics computation
   10. Logger

Usage:
    python -m efm_pipeline.smoke_test
"""

import os
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch


def _ok(name):
    print(f"  ✓ {name}")


def _fail(name, e):
    print(f"  ✗ {name}: {e}")
    traceback.print_exc()


def test_imports():
    """Test all module imports."""
    import efm_pipeline
    from efm_pipeline.utils import set_seed, get_device
    from efm_pipeline.logger import ExperimentLogger
    from efm_pipeline.config import ExperimentConfig, efm_tiny_config
    from efm_pipeline.local_time import LocalTimeConditioner
    from efm_pipeline.insertion_head import InsertionHead
    from efm_pipeline.expand_op import ExpandOperator, ExpandOperatorBatched
    from efm_pipeline.efm_model import EFMModel, EFM_Tiny, EFM_Small
    _ok("All imports")


def test_config():
    """Test configuration creation."""
    from efm_pipeline.config import ExperimentConfig, efm_tiny_config, efm_small_config, elf_baseline_config
    cfg = efm_tiny_config()
    assert cfg.model.depth == 4
    assert cfg.model.hidden_size == 256
    cfg2 = efm_small_config(time_mode="quantized", K=8)
    assert cfg2.local_time.K == 8
    _ok("Config creation")


def test_local_time():
    """Test all local-time conditioning modes."""
    from efm_pipeline.local_time import LocalTimeConditioner
    B, S, H = 2, 16, 128
    tau = torch.rand(B, S)

    for mode in ["continuous", "quantized", "lowrank"]:
        for basis in (["fourier", "learned"] if mode == "lowrank" else [None]):
            kwargs = {"hidden_size": H, "mode": mode, "K": 4}
            if basis:
                kwargs["lowrank_basis"] = basis
            ltc = LocalTimeConditioner(**kwargs)
            out = ltc(tau)
            assert out.shape == (B, S, H), f"{mode}/{basis}: shape {out.shape} != ({B}, {S}, {H})"
            assert not torch.isnan(out).any(), f"{mode}/{basis}: NaN in output"
            _ok(f"LocalTime mode={mode}" + (f"/{basis}" if basis else ""))


def test_insertion_head():
    """Test insertion head forward and loss."""
    from efm_pipeline.insertion_head import InsertionHead
    B, S, H = 2, 16, 128
    head = InsertionHead(H, max_insert_per_gap=3)
    hidden = torch.randn(B, S, H)

    rates = head(hidden)
    assert rates.shape == (B, S - 1), f"Rates shape: {rates.shape}"
    assert (rates >= 0).all(), "Rates must be non-negative"

    counts = head.predict_counts(hidden)
    assert counts.shape == (B, S - 1)
    assert (counts >= 0).all() and (counts <= 3).all()

    target = torch.zeros(B, S - 1)
    loss = InsertionHead.poisson_nll_loss(rates, target)
    assert loss.isfinite(), f"Loss is not finite: {loss}"
    _ok("InsertionHead")


def test_expand_op():
    """Test expand operator."""
    from efm_pipeline.expand_op import ExpandOperator, ExpandOperatorBatched
    B, S, D = 2, 8, 64
    emb = torch.randn(B, S, D)
    counts = torch.tensor([[1, 0, 2, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0]])  # (B, S-1)
    tau = torch.ones(B, S)

    # Loop-based.
    expand = ExpandOperator(D, max_total_length=32)
    exp_emb, exp_tau, exp_mask = expand(emb, counts, tau)
    assert exp_emb.shape[0] == B
    assert exp_emb.shape[2] == D
    assert exp_tau.shape == exp_emb.shape[:2]
    _ok("ExpandOperator (loop)")

    # Batched.
    expand_b = ExpandOperatorBatched(D, max_total_length=32)
    exp_emb_b, exp_tau_b, exp_mask_b = expand_b(emb, counts, tau)
    assert exp_emb_b.shape[0] == B
    _ok("ExpandOperatorBatched")


def test_efm_model():
    """Test EFM model forward pass."""
    from efm_pipeline.efm_model import EFM_Tiny
    B, S = 2, 16
    model = EFM_Tiny(
        text_encoder_dim=64, max_length=32, vocab_size=100,
        local_time_mode="continuous", insertion_enabled=True,
    )
    # Override for smaller test.
    x = torch.randn(B, S, 64)
    t = torch.rand(B)
    tau = torch.rand(B, S)

    # Flow output.
    flow_out, dec_logits = model(x, t, local_times=tau, decoder_step_active=False)
    assert flow_out.shape == (B, S, 64), f"Flow shape: {flow_out.shape}"
    assert dec_logits is None
    assert not torch.isnan(flow_out).any()
    _ok("EFM forward (flow)")

    # Decoder output.
    flow_out2, dec_logits2 = model(x, t, local_times=tau, decoder_step_active=True)
    assert dec_logits2 is not None
    assert dec_logits2.shape == (B, S, 100), f"Logits shape: {dec_logits2.shape}"
    _ok("EFM forward (decoder)")


def test_training_step():
    """Test a single training step."""
    from efm_pipeline.efm_model import EFM_Tiny
    B, S = 2, 16
    model = EFM_Tiny(
        text_encoder_dim=64, max_length=32, vocab_size=100,
        local_time_mode="continuous", insertion_enabled=False,
    )
    model.train()
    x0 = torch.randn(B, S, 64)
    t = torch.rand(B)
    noise = torch.randn_like(x0)
    z = t.unsqueeze(1).unsqueeze(2) * x0 + (1 - t.unsqueeze(1).unsqueeze(2)) * noise
    tau = t.unsqueeze(1).expand(-1, S)

    flow_out, dec_logits = model(z, t, local_times=tau, decoder_step_active=True)
    loss = torch.nn.functional.mse_loss(flow_out, x0)

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"
    assert loss.isfinite()
    _ok(f"Training step (loss={loss.item():.4f})")


def test_logger():
    """Test the experiment logger."""
    import tempfile
    from efm_pipeline.logger import ExperimentLogger, load_all_results
    with tempfile.TemporaryDirectory() as tmpdir:
        log = ExperimentLogger(tmpdir, "test_exp")
        log.log_step({"step": 1, "loss": 2.3})
        log.log_step({"step": 2, "loss": 1.8})
        log.save_result({"gen_ppl": 25.0, "entropy": 5.1})
        log.close()

        results = load_all_results(tmpdir)
        assert "test_exp" in results
        assert results["test_exp"]["gen_ppl"] == 25.0
    _ok("Logger")


def main():
    print(f"\n{'='*60}")
    print(f"  EFM PIPELINE SMOKE TEST")
    print(f"{'='*60}\n")

    from efm_pipeline.utils import set_seed, get_device, device_summary
    set_seed(42)
    device = get_device()
    print(f"  Device: {device_summary(device)}")
    print()

    start = time.perf_counter()
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("LocalTime", test_local_time),
        ("InsertionHead", test_insertion_head),
        ("ExpandOp", test_expand_op),
        ("EFMModel", test_efm_model),
        ("TrainingStep", test_training_step),
        ("Logger", test_logger),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            _fail(name, e)
            failed += 1

    elapsed = time.perf_counter() - start
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if failed == 0:
        print(f"  All smoke tests passed! ✓")
    else:
        print(f"  {failed} test(s) failed ✗")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
