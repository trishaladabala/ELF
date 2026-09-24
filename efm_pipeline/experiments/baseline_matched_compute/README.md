# Matched-Compute Baseline

## Purpose
Show that a plain autoregressive model (or single-time-step flow 
model) at matched parameter count does NOT reproduce the U-shaped 
relationship on the synthetic task. This rules out "it's a quirk 
of our codebase / our specific model."

## Setup
- Parameter count: matched to 36M
- Training budget: matched to 50k steps
- Task: same synthetic task as Deliverable 3
- Conditioning: single global time only (no local time)

## Deliverable
- A single curve: Gen-PPL vs. training steps.
- Compare against the EFM curves from Deliverable 3.
- Claim: no U-shape appears.

## Script
See `run.py` in this directory. It imports the existing training 
loop and overrides the conditioning module to be global-time-only.

