# Baseline Matched-Compute Experiment

## Purpose

This experiment trains `EFM_Small` (36M parameters) on the synthetic Ordered Assembly 
task for 50,000 steps across **all local-time conditioning modes**, establishing whether 
the U-shaped compression curve appears on the synthetic task at full compute budget.

This rules out the possibility that the U-shape observed in WikiText-2 experiments 
is an artifact of model size, training dynamics, or the natural language domain.

## Conditions Tested

| Condition | Mode | K | Expected Behavior |
|---|---|---|---|
| `global_time_only` | none | — | Moderate accuracy (can't distinguish waves) |
| `continuous` | continuous | — | Overfits at high W, poor generalization |
| `quantized_K2` | quantized | 2 | Under-compressed for W>3 |
| `quantized_K4` | quantized | 4 | Near-optimal for W=4 |
| `lowrank_K2` | lowrank | 2 | Under-compressed for W>3 |
| `lowrank_K4` | lowrank | 4 | Near-optimal for W=4 |

## Usage

```bash
cd /home/snu/trishal_sanjay_proj/ELF/efm_pipeline

# Default: W=4, 50k steps
python -m experiments.baseline_matched_compute.run

# Higher complexity
python -m experiments.baseline_matched_compute.run --W 16 --steps 50000

# Quick test
python -m experiments.baseline_matched_compute.run --steps 5000
```

## Metrics

- **TF Accuracy**: Teacher-forced exact match (should be ~99%+ for all conditions)
- **Internal Dep Acc**: Do generated values copy from the model's own generated keys? 
  *(PRIMARY metric — measures wave-routing consistency)*
- **External Dep Acc**: Do generated values match ground-truth keys? 
  *(Only meaningful for teacher-forced; will be ~1/256 for unconditional generation)*

## Expected Result

A U-shaped curve in Internal Dependency Accuracy as a function of conditioning 
capacity K, with the minimum matching the task complexity W.
