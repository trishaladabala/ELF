#!/usr/bin/env python3
"""Stage 1 — Train continuous-time EFM baseline.

Trains the EFM-style model with continuous (uncompressed) local-time
conditioning. This is the "EFM continuous baseline" that the compression
experiments (Stages 3-4) are compared against.

Usage:
    # Short validation run
    python -m efm_pipeline.experiments.stage1_train_efm --max_steps 1000

    # Full A4000 run
    python -m efm_pipeline.experiments.stage1_train_efm --model_size small --max_steps 50000 --use_amp
"""

import sys
import os

# Run the training script with continuous time mode.
if __name__ == "__main__":
    sys.argv = [
        "train_efm",
        "--time_mode", "continuous",
        "--experiment_name", "stage1_continuous_baseline",
    ] + sys.argv[1:]

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
    if _PROJ_ROOT not in sys.path:
        sys.path.insert(0, _PROJ_ROOT)

    from efm_pipeline.train_efm import main
    main()
