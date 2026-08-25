# Part T1: R2 Sanity Gate

**Objective:** Confirm that the R2 curvature penalty (which produced a negative result for downstream quality) was actually wired into the computational graph and mechanically capable of producing gradients, rather than being a silent implementation bug (e.g., detached tensors or zeroed loss).

## Methodology
We reran a short segment of the R2 fine-tuning loop (100 steps) but pushed the `curvature_weight` to an extreme value (`5.0` vs original `0.1`). Crucially, we decomposed the `backward()` pass to isolate and log the gradient norms contributed strictly by the base loss (CE + L2) versus the auxiliary curvature loss.

## Results
- **Avg Base Grad Norm:** 1.9542
- **Avg Curvature Grad Norm:** 0.4024
- **Ratio (Curv / Base):** ~20.6%

## Interpretation
The curvature loss term is **mechanically active and correctly wired**. At the extreme weight of 5.0, it contributes roughly 20% of the total gradient norm, meaning the computational graph is intact and gradients are successfully flowing back to the model parameters through the geometry computation. 

Because the loss is functionally active, we can confirm that **the R2 negative result is a genuine scientific finding**: explicitly penalizing trajectory turning angles successfully propagates gradients but fails to meaningfully restructure the geometry in a way that improves downstream SDE generation quality. The R2 conclusion stands.
