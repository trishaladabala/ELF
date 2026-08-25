# Pre-Registration: Final Idea 6 Confirmatory Test (I2)

This document locks in the single confirmatory statistical test for the Idea 6 (Principled SDE Sampler) empirical claim, written before any new data for I2 is generated.

## Rationale
In Part G1, the Corrected SDE showed statistically significant improvements over the ODE sampler under a uniform time schedule (at 16 and 32 steps). However, ELF defaults to a `logit_normal` schedule. Under `logit_normal`, the difference at 8 steps was large in magnitude (Δ = -3178.2) but statistically non-significant due to wide confidence intervals. 

To determine if this is a true effect or just noise, we will run a properly powered, pre-registered test specifically targeting this gap, without the multiple-comparisons inflation of a full sweep.

## The Confirmatory Comparison
We pre-register the following exact comparison as the **single confirmatory claim** for Idea 6:

- **Samplers:** Corrected-SDE (g=0.5) vs. ODE
- **Schedule:** `logit_normal` (ELF's default)
- **Step Count:** 8 steps
- **Metric:** External Generative Perplexity (via `distilgpt2`) on detokenized sequences.

## Statistical Test
- **Method:** Two-sided paired bootstrap on the mean PPL difference (`PPL_ODE - PPL_SDE`), using paired sequences generated from the same initial noise seeds.
- **Significance Threshold:** α = 0.05 (95% Confidence Interval)
- **Sample Size:** N=512 generated sequences (increased from G1's 256 to ensure adequate power for the wide variance seen under `logit_normal`).
- **Resamples:** 1000 bootstrap iterations.

## Decision Rule
- **If the 95% CI excludes zero (favoring Corrected-SDE):** The quality advantage is confirmed under realistic settings. Proceed to Magus with this as the central empirical claim.
- **If the 95% CI includes zero or favors ODE:** There is no detectable quality advantage under the realistic schedule at toy scale. Idea 6 will be framed as a supporting/negative result rather than the headline claim.

*(Any other step counts or schedules run alongside this will be explicitly reported as purely descriptive context.)*
