# Magus Implementation Plan — Phase 1: Real-Scale Replication

**Date:** 2026-08-25
**Cluster:** Magus GPU Cluster, Shiv Nadar University
**Objective:** Validate Claims A, B, C, D from the Pre-Registration on the production-scale ELF-B checkpoint (105M parameters).

---

## 0. Prerequisites (Before You Get Cluster Access)

### 0.1 Code Repository
Push the entire `ELF/` directory to a private Git repo so you can clone it on Magus:
```bash
cd /Users/adabalatrishal/7semminor/ELF
git remote add magus <your-private-repo-url>
git push magus main
```

### 0.2 Checkpoint Access
The ELF-B checkpoint is downloaded from HuggingFace. The loading code already exists in [`phase0l/testL1_checkpoint_sanity.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0l/testL1_checkpoint_sanity.py):
```python
HF_REPOS = {
    "ELF-B": "embedded-language-flows/ELF-B-owt-torch",
    "ELF-M": "embedded-language-flows/ELF-M-owt-torch",
    "ELF-L": "embedded-language-flows/ELF-L-owt-torch",
}
```
Download once to a shared cache directory on Magus to avoid re-downloading per job.

### 0.3 Environment Setup
```bash
# On Magus login node
module load cuda/12.x python/3.11
python -m venv ~/.venvs/elf
source ~/.venvs/elf/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub scipy matplotlib numpy
pip install sentencepiece  # for T5 tokenizer used by ELF
```

### 0.4 Key Adaptation: Device Change
All local scripts use `DEVICE = "mps"`. Every Magus script must change to:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # ELF-B fits in fp16 on a single A100/V100
```

---

## 1. Stage 1: Checkpoint Sanity (Day 1, ~1 GPU-hour)

> **Goal:** Confirm the ELF-B checkpoint loads correctly and produces coherent text on Magus before running any experiments.

### What to Run
Adapt [`phase0l/testL1_checkpoint_sanity.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0l/testL1_checkpoint_sanity.py) for CUDA:

| Parameter | Local Value | Magus Value |
|-----------|-------------|-------------|
| `DEVICE` | `"mps"` | `"cuda"` |
| `DTYPE` | `torch.float32` | `torch.float16` |
| `N_SAMPLES` | 32 | 32 (same — this is just a sanity check) |
| `BATCH_SIZE` | 4 | 16 (A100 has much more VRAM) |
| `max_length` | 128 | **1024** (production sequence length) |

### Pass/Fail Gate
- **PASS:** Generated text is coherent English. GPT-2 Large perplexity is within ±20% of the paper's reported value (24.1 for ELF-B).
- **FAIL:** Stop. Debug the checkpoint loading / CUDA compatibility before proceeding.

### SLURM Script Template
```bash
#!/bin/bash
#SBATCH --job-name=elf_sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sanity_%j.out

source ~/.venvs/elf/bin/activate
python phase1/stage1_sanity.py
```

---

## 2. Stage 2: Claims A + B — Degeneracy Rate & SDE Quality Penalty (Day 1–2, ~4 GPU-hours)

> **Goal:** Run the core ODE vs SDE comparison at production scale. This is the single most important experiment.

### What to Run
Create `phase1/stage2_claim_ab.py` by adapting [`phase0i/testI2_final_result.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0i/testI2_final_result.py) with the following changes:

| Parameter | Local (Phase 0I) | Magus (Phase 1) |
|-----------|-------------------|-----------------|
| Model | Toy ELF (~275K params) | ELF-B (105M params) |
| `N_SAMPLES` | 512 | **1024** (more power for small effects) |
| `BATCH_SIZE` | 64 | 32 (larger model needs more VRAM per sample) |
| `N_STEPS` | 8 (confirmatory) | **32** (ELF paper's default) |
| `max_length` | 16 | **1024** |
| `SDE_GAMMA` | 0.5 | **1.5** (ELF paper's default) |
| Scorer | DistilGPT2 | **GPT-2 Large** (stronger evaluator) |

### Procedure
1. Generate 1024 sequences with the **ODE sampler** (deterministic, no noise). Save all outputs + trajectories.
2. Generate 1024 sequences with the **Corrected-SDE sampler** (γ=1.5). Use the **same initial noise** (same random seeds) so each pair is directly comparable.
3. For each output:
   - Detect degeneracy (repetition collapse) using the existing 3-gram repeat detector.
   - Score non-degenerate outputs with GPT-2 Large perplexity.
4. Compute:
   - **Claim A:** ODE degeneracy rate. Fisher's exact test comparing ODE vs SDE rates.
   - **Claim B:** Paired bootstrap CI on log-PPL difference (ODE − SDE) for jointly non-degenerate samples.

### Pass/Fail Bars (From Pre-Registration)
- **Claim A Replicates:** ODE degeneracy > 1.0%, SDE significantly reduces it.
- **Claim A Fails:** ODE degeneracy < 1.0%.
- **Claim B Replicates:** SDE log-PPL is significantly worse than ODE on normal paths (CI strictly negative).
- **Claim B Fails:** SDE matches or beats ODE quality.

### Save Everything
```
phase1/results/stage2/
├── ode_outputs.json          # All 1024 ODE generations
├── sde_outputs.json          # All 1024 SDE generations
├── ode_trajectories.pt       # z_t at each step (needed for Claims C & D)
├── sde_trajectories.pt
├── degeneracy_report.json    # Per-seed degeneracy flags
├── ppl_comparison.json       # Paired PPL scores
└── claim_ab_verdict.md       # Automated pass/fail report
```

> [!IMPORTANT]
> **You MUST save the raw trajectories (z_t tensors at each ODE step).** Claims C and D both require them, and re-generating 1024 sequences at 1024 length is expensive. Save once, analyze twice.

---

## 3. Stage 3: Claim C — Curvature-Driven Overconfidence (Day 2–3, ~2 GPU-hours)

> **Goal:** Test whether high-curvature trajectory regions cause SDE overconfidence at production scale.

### What to Run
Create `phase1/stage3_claim_c.py` by adapting [`phase0j/testJ2_curvature_quality_correlation.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0j/testJ2_curvature_quality_correlation.py):

### Procedure (No New Generation — Uses Stage 2 Trajectories)
1. Load `ode_trajectories.pt` and `sde_trajectories.pt` from Stage 2.
2. For each sequence, compute **turning-angle curvature** at each timestep (the angle between consecutive velocity vectors in embedding space).
3. For each sequence, compute the **token-level entropy** from both samplers.
4. Compute the **Spearman correlation** between per-token curvature and the SDE's entropy advantage (SDE entropy − ODE entropy).

### Pass/Fail Bars
- **Replicates:** Spearman $r \ge +0.10$, $p < 0.01$.
- **Does Not Replicate:** Correlation is flat or negative.

### Scale Trend Context
At toy scale (275K params), $r = +0.38$. At 1M, $r = +0.13$. At 5M, $r = -0.08$. The trend strongly suggests this will **not** replicate. Document the result honestly regardless.

---

## 4. Stage 4: Claim D — Hard-Seed Geometric Signature (Day 3, ~1 GPU-hour)

> **Goal:** Test whether degenerate seeds have measurably different trajectory geometry at production scale.

### What to Run
Create `phase1/stage4_claim_d.py` by adapting [`phase0f/testF1a_partition.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0f/testF1a_partition.py), [`phase0f/testF1b_extract.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0f/testF1b_extract.py), and [`phase0f/testF1c_stats.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0f/testF1c_stats.py):

### Procedure (No New Generation — Uses Stage 2 Data)
1. Load `degeneracy_report.json` from Stage 2. Partition seeds into "Degenerate" vs "Non-Degenerate" groups based on ODE output.
2. Load `ode_trajectories.pt`. Compute mean turning-angle curvature per seed.
3. Run **Mann-Whitney U test** (Degenerate vs Non-Degenerate curvatures).
4. Compute **Cohen's d** and **rank-biserial r**.
5. Run the **confound check**: compare L2 norms of initial noise ($z_0$) across groups.

### Pass/Fail Bars
- **Replicates:** Mann-Whitney $p < 0.01$ **AND** Cohen's $d \ge 0.50$.
- **Does Not Replicate:** $p > 0.05$ or $d < 0.30$.

### Dependency
> [!WARNING]
> **Claim D is auto-voided if Claim A fails.** If the ODE produces < 1% degeneracy at ELF-B scale, there are not enough degenerate seeds to form a meaningful partition. In this case, report: *"Claim D is untestable because the failure mode it characterizes does not occur at production scale."*

---

## 5. Stage 5: Conditional Task Generalization (Day 3–4, ~4 GPU-hours)

> **Goal:** Test whether the phenomena from Stages 2–4 extend to conditional generation tasks (translation, summarization), or are exclusive to unconditional generation.

### What to Run
Adapt [`phase0m/testM1_wmt.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0m/testM1_wmt.py) and [`phase0m/testM2_xsum.py`](file:///Users/adabalatrishal/7semminor/ELF/phase0m/testM2_xsum.py):

| Task | Dataset | Metric | N_SAMPLES |
|------|---------|--------|-----------|
| Translation | WMT14 EN→DE | BLEU + Degeneracy Rate | 512 |
| Summarization | XSum | ROUGE-L + Degeneracy Rate | 512 |

### Procedure
For each task:
1. Run ODE generation on 512 conditional inputs.
2. Run SDE generation on the same 512 inputs with the same noise.
3. Measure degeneracy rate, task-specific quality metric, and trajectory curvature.

### Expected Outcome
Based on local Phase 0M results, conditional tasks are expected to show **zero degeneracy** for both samplers, because the input conditioning tightly constrains the diffusion manifold. This is the "generalization boundary" of the paper.

---

## 6. Timeline Summary

| Day | Stage | GPU-Hours | Description |
|-----|-------|-----------|-------------|
| 1 | Stage 1 | ~1h | Checkpoint sanity — confirm ELF-B loads and generates |
| 1–2 | Stage 2 | ~4h | **Core experiment:** ODE vs SDE on 1024 unconditional samples (Claims A+B) |
| 2–3 | Stage 3 | ~2h | Curvature–entropy correlation (Claim C) — analysis only, no new generation |
| 3 | Stage 4 | ~1h | Hard-seed geometric partition (Claim D) — analysis only, no new generation |
| 3–4 | Stage 5 | ~4h | Conditional tasks: WMT14 + XSum |
| **Total** | | **~12h** | **All primary experiments** |

> [!TIP]
> Stages 3 and 4 require **zero additional GPU generation** — they analyze the trajectories saved in Stage 2. If GPU time is limited, prioritize Stage 2 above everything else.

---

## 7. Decision Flowchart After Results

```
Stage 2 complete
    │
    ├── ODE degeneracy > 1%? ──── YES ──→ Claim A REPLICATES
    │                                      │
    │                                      ├── SDE PPL worse? → Claim B REPLICATES
    │                                      │                    → Run Stage 3 (Claim C)
    │                                      │                    → Run Stage 4 (Claim D)
    │                                      │
    │                                      └── SDE PPL same/better? → Claim B FAILS
    │                                                                 → SDE is a pure win!
    │                                                                 → Paper Shape: Method paper
    │
    └── ODE degeneracy < 1%? ──── YES ──→ Claim A FAILS
                                          → Claim D AUTO-VOIDED
                                          → Still run Stage 3 (Claim C may independently hold)
                                          → Paper Shape: Negative result / Scaling paper
                                          → Consider pivot directions (see discussion doc)
```

---

## 8. File Structure on Magus

```
~/elf-phase1/
├── src/                          # ELF source code (cloned from repo)
├── checkpoints/                  # Downloaded ELF-B/M/L weights
├── phase1/
│   ├── stage1_sanity.py
│   ├── stage2_claim_ab.py
│   ├── stage3_claim_c.py
│   ├── stage4_claim_d.py
│   ├── stage5_wmt.py
│   ├── stage5_xsum.py
│   └── results/
│       ├── stage1/
│       ├── stage2/               # ← Most important: trajectories saved here
│       ├── stage3/
│       ├── stage4/
│       └── stage5/
├── slurm/                        # SLURM job scripts
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   ├── run_stage4.sh
│   └── run_stage5.sh
└── logs/                         # SLURM output logs
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ELF-B checkpoint doesn't load on CUDA | Stage 1 catches this before any expensive jobs run |
| OOM at seq_len=1024 | Reduce `BATCH_SIZE` to 8 or 4. Use gradient checkpointing if available. Fall back to `max_length=512` and document the limitation. |
| Degeneracy detector tuned to toy scale | Run the detector on 10 hand-inspected samples first. Adjust the 3-gram threshold if needed for longer sequences. |
| Cluster queue is slow / allocation is limited | Stages 3+4 are free (pure analysis). Prioritize Stage 2 generation above all else. Stage 5 (conditional) can be dropped if time is critically short — the unconditional results are the primary contribution. |
| Results are ambiguous (borderline pass/fail) | The pre-registration defines exact numeric bars. Report the exact numbers honestly, do not p-hack. If borderline, frame it as "suggestive but not conclusive" and note that higher N would be needed. |
