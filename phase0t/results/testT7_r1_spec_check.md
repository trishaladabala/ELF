# Part T7: R1 Training Density Spec Check

**Finding:** The R1 `logit_normal_oversampled` schedule in `src/utils/sampling_utils.py` generates 50% of its timesteps from a `uniform(0.9, 1.0)` distribution. Because the ODE formulation defines $t=1$ as the clean data and $t=0$ as pure noise, this confirms that R1 **did correctly oversample the near-boundary region ($t \to 1$)** precisely as originally designed. 

**Correction:** The theoretical explanation proposed in T6 (claiming that R1 failed because it oversampled $t \to 0$ and starved the boundary) was factually incorrect and contradicts the codebase's literal implementation. R1 worsened degeneracy for reasons not fully explained. The density-reweighting hypothesis—that simply adding more training capacity at the unstable boundary would stabilize generation—is definitively falsified by this result, but the exact mechanism for *why* it backfired and created 52 new degenerate failures remains an open question.
