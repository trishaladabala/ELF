#!/usr/bin/env python3
"""Test that the instrumented sampler produces bit-comparable output
to the existing (Phase 0) generation pipeline.

Verifies:
- Same seed + config → same final latent states (within bf16 tolerance)
- Same decoded token IDs (exact match)
- Run on a small sample (4 samples, 32 steps) for speed
"""

import sys
import time
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELFBASIN = _SCRIPT_DIR.parent / "src"
_ELF_SRC = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_ELFBASIN))
sys.path.insert(0, str(_ELF_SRC))

from elfbasin.model.elf import ELFWrapper
from elfbasin.sampling.instrumented_sampler import instrumented_generate
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch
from configs.config import SamplingConfig


def test_equivalence_owt():
    """Test OWT SDE-32 equivalence."""
    print("="*70)
    print("  Sampler Equivalence Test: ELF-B-owt, SDE-32, 4 samples")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 12345
    BATCH = 4
    NUM_STEPS = 32
    SDE_GAMMA = 1.5
    SC_CFG = 3.0
    CFG = 1.0

    wrapper = ELFWrapper("ELF-B-owt", device=device)
    model = wrapper.model
    config = wrapper.config
    d_model = wrapper.d_model
    max_length = wrapper.max_length
    param_dtype = next(model.parameters()).dtype

    sampling_config = SamplingConfig(
        sampling_method="sde",
        num_sampling_steps=[NUM_STEPS],
        cfgs=[CFG],
        sde_gamma=SDE_GAMMA,
        self_cond_cfg_scales=[SC_CFG],
        time_schedule="logit_normal",
    )

    # ── Run 1: Original pipeline ──
    print("\n  [1/3] Running original pipeline...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Generate deterministic time steps
    t_steps = get_sampling_steps(
        n_steps=NUM_STEPS, time_schedule="logit_normal",
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=param_dtype,
    )

    generator = torch.Generator(device=device).manual_seed(SEED)
    z_init = torch.randn(
        (BATCH, max_length, d_model),
        dtype=param_dtype, device=device,
        generator=generator,
    ) * config.denoiser_noise_scale

    # Must re-seed generator for the SDE noise inside the loop
    generator_orig = torch.Generator(device=device).manual_seed(SEED + 1)

    t0 = time.time()
    z_orig = _generate_samples_single_batch(
        model=model, generator=generator_orig, z=z_init.clone(),
        t_steps=t_steps,
        cond_seq=None, cond_seq_mask=None,
        config=config, sampling_config=sampling_config,
        cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
    )
    t_orig = time.time() - t0

    # Decode
    t_final_val = t_steps[-1].item()
    ids_orig = _dlm_decode_batch(
        z=z_orig, model=model, t_final_val=t_final_val,
        config=config, self_cond_cfg_scale=SC_CFG,
    )
    print(f"    Original: {t_orig:.2f}s, z_norm={z_orig.norm():.4f}")

    # ── Run 2: Instrumented pipeline (no diagnostics) ──
    print("  [2/3] Running instrumented pipeline (no diagnostics)...")
    generator_inst = torch.Generator(device=device).manual_seed(SEED + 1)

    t0 = time.time()
    z_inst, trajectory = instrumented_generate(
        model=model, wrapper=wrapper,
        generator=generator_inst, z=z_init.clone(),
        t_steps=t_steps,
        cond_seq=None, cond_seq_mask=None,
        config=config, sampling_config=sampling_config,
        cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
        compute_diagnostics=False,
        compute_L_i=False,
    )
    t_inst = time.time() - t0
    print(f"    Instrumented: {t_inst:.2f}s, z_norm={z_inst.norm():.4f}")

    # ── Compare ──
    print("\n  [3/3] Comparing outputs...")

    # Latent comparison
    diff = (z_orig - z_inst).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"    Max |z_orig - z_inst|: {max_diff:.6e}")
    print(f"    Mean |z_orig - z_inst|: {mean_diff:.6e}")

    # Token comparison
    ids_inst = trajectory.final_token_ids.to(device)
    ids_orig_cpu = ids_orig.cpu()
    ids_inst_cpu = ids_inst.cpu()
    token_match = (ids_orig_cpu == ids_inst_cpu).float().mean().item()
    print(f"    Token agreement: {token_match*100:.2f}%")

    # Tolerance check
    ATOL = 1e-3
    RTOL = 1e-3
    latent_ok = torch.allclose(z_orig, z_inst, atol=ATOL, rtol=RTOL)
    tokens_ok = token_match == 1.0

    if latent_ok:
        print(f"\n    ✓ Latent states match (atol={ATOL}, rtol={RTOL})")
    else:
        print(f"\n    ✗ Latent states DIFFER (max_diff={max_diff:.6e}, "
              f"atol={ATOL})")

    if tokens_ok:
        print(f"    ✓ Token IDs match exactly (100%)")
    else:
        print(f"    ✗ Token IDs DIFFER ({token_match*100:.2f}% agreement)")

    # ── Run 3: With diagnostics (verify it doesn't change output) ──
    print("\n  [Bonus] Running with diagnostics enabled...")
    generator_diag = torch.Generator(device=device).manual_seed(SEED + 1)

    t0 = time.time()
    z_diag, traj_diag = instrumented_generate(
        model=model, wrapper=wrapper,
        generator=generator_diag, z=z_init.clone(),
        t_steps=t_steps,
        cond_seq=None, cond_seq_mask=None,
        config=config, sampling_config=sampling_config,
        cfg_scale=CFG, self_cond_cfg_scale=SC_CFG,
        compute_diagnostics=True,
        compute_L_i=False,
    )
    t_diag = time.time() - t0

    diff_diag = (z_orig - z_diag).abs().max().item()
    ids_diag = traj_diag.final_token_ids.to(device)
    token_match_diag = (ids_orig.cpu() == ids_diag.cpu()).float().mean().item()

    print(f"    With diagnostics: {t_diag:.2f}s")
    print(f"    Max |z_orig - z_diag|: {diff_diag:.6e}")
    print(f"    Token agreement: {token_match_diag*100:.2f}%")
    print(f"    Steps recorded: {len(traj_diag.steps)}")
    if traj_diag.steps:
        s = traj_diag.steps[-1]
        print(f"    Final step margin: mean={s.margin.mean():.3f}, "
              f"min={s.margin.min():.3f}, max={s.margin.max():.3f}")
        print(f"    Final step entropy: mean={s.entropy.mean():.3f}")

    diag_latent_ok = torch.allclose(z_orig, z_diag, atol=ATOL, rtol=RTOL)
    diag_tokens_ok = token_match_diag == 1.0

    # Summary
    all_pass = latent_ok and tokens_ok and diag_latent_ok and diag_tokens_ok
    print(f"\n{'='*70}")
    if all_pass:
        print("  ✓ EQUIVALENCE TEST PASSED")
        print("    Instrumented sampler produces identical output to original.")
    else:
        print("  ✗ EQUIVALENCE TEST FAILED")
        if not latent_ok:
            print("    - Latent states differ")
        if not tokens_ok:
            print("    - Token IDs differ")
        if not diag_latent_ok:
            print("    - Diagnostics mode changes latent output")
        if not diag_tokens_ok:
            print("    - Diagnostics mode changes token output")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(test_equivalence_owt())
