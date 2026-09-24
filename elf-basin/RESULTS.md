# RESULTS.md — Decoder-Basin Research on ELF

**Append-only experiment log.** Do not delete or overwrite entries.

---

## Gate 0 — Environment and Reproduction

### OWT (ELF-B-owt)

| Field | Value |
|-------|-------|
| Checkpoint | `ELF-B-owt` |
| Config | SDE-32, γ=1.5, SC-CFG=3, logit-normal (P_mean=−1.5, P_std=0.8) |
| Samples | 1000 |
| Seed | 42 |
| Target Gen. PPL | 24–26 (unofficial PyTorch repro: 25.61) |
| Target Entropy | ~5.15–5.20 |
| **Measured Gen. PPL** | **24.03** |
| **Measured Entropy** | **5.161** |
| Distinct-1 | 0.0596 |
| Distinct-2 | 0.3959 |
| Rep. 4-gram frac | 0.0354 |
| Wall-clock (gen) | 6.2m |
| Wall-clock (PPL) | 1.7m |
| Peak VRAM | 10428 MB |
| Batch size | 8 |
| Config hash | `53e40f6ae7e6` |
| **GATE 0 OWT** | **PASS** |

### De-En (ELF-B-de-en)

| Field | Value |
|-------|-------|
| Checkpoint | `ELF-B-de-en` |
| Config | ODE-64, CFG=2, SC-CFG=1, logit-normal (P_mean=−1.5, P_std=0.8) |
| Samples | ~3000 (WMT14 De-En validation) |
| Seed | 42 |
| Target BLEU | ~26.4 |
| **Measured BLEU** | **26.51** |
| ROUGE-1 | 60.19 |
| ROUGE-2 | 34.46 |
| ROUGE-L | 55.44 |
| Wall-clock (gen) | 9.8m |
| Peak VRAM | 1692 MB |
| Batch size | 32 |
| Config hash | `d578cd6e3a61` |
| **GATE 0 DE-EN** | **PASS** |

---

## Gate 1 — Instrumented Sampler and Phase Structure

### Sampler Equivalence

| Field | Value |
|-------|-------|
| Test | ELF-B-owt, SDE-32, 4 samples, seed 12345 |
| Latent max diff | **0.000000e+00** (exact match) |
| Token agreement | **100.00%** (exact match) |
| Diagnostics impact | None (latents unchanged with diagnostics on) |
| **EQUIVALENCE** | **PASS** |

### Phase Structure Verification (ELF-B-owt, SDE-32, 512 samples)

| Field | Value |
|-------|-------|
| Checkpoint | `ELF-B-owt` |
| Samples | 512 |
| Steps | 32, SDE γ=1.5, SC-CFG=3 |
| Seed | 42 |
| Wall-clock (gen) | 16.0m |
| Peak VRAM | 4901 MB |

#### (a) Margin Monotonicity

| Field | Value |
|-------|-------|
| First step mean margin | 0.3766 |
| Last step mean margin | 19.9615 |
| Strictly monotone | Yes |
| **Check (a)** | **✓ PASS** |

#### (b) Self-Conditioning Disagreement

| Field | Value |
|-------|-------|
| Peak step | 1 / 31 |
| Peak value | 7.1212 |
| Peak in middle region | No |
| **Check (b)** | **⚠ MARGINAL** |

> Note: SC disagreement peaks at step 1 (earliest measurement) rather than in the
> middle. This is expected: at step 0→1, the self-conditioning prediction jumps from
> zeros to a rough estimate, producing the largest delta. The subsequent steps show
> decreasing disagreement as predictions stabilize. This is informational, not
> a gate-blocking criterion.

#### (c) Basin Entry Staggering

| Field | Value |
|-------|-------|
| Entry time std | 6.48 steps |
| Entry time IQR | 9.00 steps |
| P25 | 12.0 |
| P50 (median) | 16.0 |
| P75 | 21.0 |
| **Check (c)** | **✓ PASS** |

> Basin entry is clearly staggered: tokens enter their final basin across a wide
> range from step 12 to step 21 (IQR), confirming the per-token, phase-dependent
> structure described in the analysis paper.

| **GATE 1** | **PASS** |
|------------|----------|

Plots saved to `elf-basin/runs/gate1/`:
- `plot_a_margin_vs_step.png`
- `plot_b_sc_disagreement.png`
- `plot_c_basin_entry.png`

---
