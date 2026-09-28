#!/bin/bash
# Reruns one experiment from scratch and checks it against the stored result: the trunk-architecture and
# scale-anchoring study for seed 42 (three trunks, two heads and the anchored head; about 4 minutes on one core).
# The rerun is written to a temporary directory, so the stored results are never touched.
set -e
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
echo "Rerunning experiments/worker_arch.py 42 into $TMP ..."
RESULTS_DIR="$TMP" CHECKPOINT_DIR="$TMP/checkpoints" TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 python3 experiments/worker_arch.py 42 > "$TMP/worker_arch_42.log" 2>&1
python3 scripts/compare_results.py results/runs_arch/arch_s42.json "$TMP/runs_arch/arch_s42.json"
