# Part F1c: Hard-Seed Statistical Analysis

## Curvature Comparison (Turning Angle)
- **Always-Fails (N=15):** Mean = 0.0123, Median = 0.0113
- **Sometimes-Fails (N=20):** Mean = 0.0137, Median = 0.0112
- **Never-Fails (N=200):** Mean = 0.0345, Median = 0.0369

**Kruskal-Wallis H-test (Across all 3):** H=71.3251, p=3.25e-16
**Mann-Whitney U-test (Always > Never):** U=123.0, p=3.15e-09

## Initial Noise Magnitude Comparison (L2 Norm)
- **Always-Fails (N=15):** Mean = 45.3494, Median = 45.2522
- **Sometimes-Fails (N=20):** Mean = 45.3734, Median = 45.3625
- **Never-Fails (N=200):** Mean = 45.2416, Median = 45.2131

**Kruskal-Wallis H-test (Across all 3):** H=1.0579, p=5.89e-01
**Mann-Whitney U-test (Always vs Never):** U=1650.0, p=5.20e-01

## Interpretation
**GO:** Always-fails seeds show significantly lower curvature than never-fails seeds, and this is NOT explained by a difference in initial noise magnitude. This provides strong, independent support for the trajectory geometry playing a key role in failure (hard seeds have lower curvature).