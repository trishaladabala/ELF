# Audit Report
Date: 2026-09-24

## Repo state vs. Deliverables 1-4

- **Deliverable 1** (anchor_finding.md): **missing** → created at `docs/anchor_finding.md`
- **Deliverable 2** (theory_sketch.md): **missing** → created at `docs/theory_sketch.md`
- **Deliverable 3** (second_synthetic_task.md): **missing** → created at `docs/second_synthetic_task.md`
- **Deliverable 4** (baseline_matched_compute/): **missing** → created at `experiments/baseline_matched_compute/` with `README.md` and `run.py` stub

## Existing files that overlap with these deliverables

| File | Overlap | Notes |
|:---|:---|:---|
| `efm_pipeline/experiments/scaleup_matrix.py` | Partial overlap with anchor finding validation | Scales to 105M params / 200k steps — **explicitly a non-goal** per the user's instructions. Created in prior turn; should not be run. |
| `efm_pipeline/experiments/validation_matrix.py` | Produced the anchor finding data | This is the script that generated the 3-seed, 7-condition results at 36M params. The anchor finding in Deliverable 1 is derived from its output. |
| `efm_pipeline/experiments/validation_summary.py` | Aggregates anchor finding data | Prints the summary table that includes the Gen-PPL numbers cited in the anchor paragraph. |
| `efm_pipeline/validation_phase/*.json` | Raw data behind anchor finding | 46 JSON result files from the 3-seed validation matrix. These are the source-of-truth numbers. |
| `efm_pipeline/efm_pipeline/local_time.py` | Core implementation for Deliverables 3-4 | Contains the `LocalTimeConditioner` with continuous/quantized/lowrank/none modes. Any synthetic task will use this directly. |

## Blockers

1. **Deliverable 3 (second synthetic task)**: The spec lists three candidate designs but none is selected. **Decision needed**: which option (A, B, or C) to implement first?
2. **Deliverable 4 (baseline script)**: The `run.py` stub raises `NotImplementedError` because it depends on the synthetic task dataset from Deliverable 3. This is intentional — Deliverable 3 must be implemented first.
3. **No first synthetic task exists yet**: The implementation plan (prior conversation) references a "first synthetic task" (ordered assembly sequences) but no code for it exists in the repo. Deliverable 3 is spec'd as a *second* synthetic task, implying a first must also be built. **Decision needed**: should the first synthetic task (ordered assembly) be created as a prerequisite, or should Deliverable 3 be treated as the *only* synthetic task?

