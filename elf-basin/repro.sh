#!/bin/bash
set -e
echo "========================================"
echo "  Reproducing all ELF Basin experiments "
echo "========================================"

echo "Running Phase 0 (OWT)"
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_owt.py

echo "Running Phase 0 (De-En)"
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_deen.py

echo "Running Phase 1 (Equivalence)"
CUDA_VISIBLE_DEVICES=0 python tests/test_sampler_equivalence.py

echo "Running Phase 1 (Gate 1 Verification)"
CUDA_VISIBLE_DEVICES=0 python scripts/gate1_verify.py

echo "Running Phase 2 (De-En Harvest)"
CUDA_VISIBLE_DEVICES=0 python scripts/harvest_deen.py

echo "Running Phase 2 (Gate 2 Verification)"
CUDA_VISIBLE_DEVICES=0 python scripts/gate2_verify.py

echo "Running Phase 3 (Headroom)"
CUDA_VISIBLE_DEVICES=0 python scripts/phase3_headroom.py

echo "Running Phase 4 (Adapter Training)"
CUDA_VISIBLE_DEVICES=0 python scripts/train_adapter.py

echo "Running Phase 4 (Gate 4 Verification)"
CUDA_VISIBLE_DEVICES=0 python scripts/gate4_verify.py

echo "Running Phase 5 (Pilot)"
CUDA_VISIBLE_DEVICES=0 python scripts/phase5_pilot.py

echo "Running Phase 5 (Control)"
CUDA_VISIBLE_DEVICES=0 python scripts/phase5_control.py

echo "Running Phase 6 (Generate Assets)"
python scripts/generate_assets.py

echo "Done!"
