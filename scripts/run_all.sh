#!/bin/bash
# Reruns every experiment in the article from scratch, then recomputes all statistics, tables and numbers and
# checks them against the stored values.
# About four hours on a 10-core Apple M4 CPU. Each job uses one thread; set PAR to the number of cores to use.
# Results are written to results/ and trained weights to checkpoints/ (override with RESULTS_DIR, CHECKPOINT_DIR).
set -e
cd "$(dirname "$0")/.."
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PAR=${PAR:-10}
LOGS=${RESULTS_DIR:-results}/logs; mkdir -p "$LOGS"; export LOGS
run() { printf '%s\n' "$@" | tr '\n' '\0' | xargs -0 -P "$PAR" -I{} sh -c 'python3 experiments/{} > "$LOGS/$(echo "{}" | tr " /" "__").log" 2>&1'; }

python3 experiments/preflight.py > "${RESULTS_DIR:-results}/preflight.log"          # amplitude bound (Sec. 3.5)
run "worker.py 42 1.0 full" "worker.py 123 1.0 full" "worker.py 456 1.0 full" "worker.py 789 1.0 full" \
    "worker.py 999 1.0 full" "worker.py 7 1.0 core" "worker.py 13 1.0 core" "worker.py 21 1.0 core" \
    "worker.py 33 1.0 core" "worker.py 55 1.0 core"                               # main study, n=10 (saves checkpoints)
run $(for s in 42 123 456; do echo "worker_gen.py $s 4 5.0 g_strong"; echo "worker_gen.py $s 4 50.0 g_weak"; done)
run "worker_gen.py 42 6 15.0 q6" "worker_gen.py 123 6 15.0 q6" "worker_gen.py 456 6 15.0 q6"   # six qubits, reduced budget
export GEN_NSAMP=120 GEN_EPOCHS=120
run "worker_gen.py 42 6 15.0 q6m" "worker_gen.py 123 6 15.0 q6m" "worker_gen.py 456 6 15.0 q6m" # six qubits, matched budget
unset GEN_NSAMP GEN_EPOCHS
run "worker_ens.py 42" "worker_ens.py 123" "worker_ens.py 456" "worker_ens.py 789" "worker_ens.py 999"   # deep ensembles
run $(for s in 42 123 456 789 999 7 13 21 33 55; do echo "worker_arch.py $s"; done)                      # trunks, remedy
run $(for s in 42 123 456 789 999; do echo "worker_ext.py $s sys2"; done)                                # XY ring
run $(for s in 42 123 456 789 999 7 13 21 33 55; do echo "worker_ext.py $s curetrunks"; done)            # remedy, other trunks
run $(sed 's/^/worker_rev.py /' experiments/jobs_rev.txt)                                                 # CMA-ES, lambda sweep
bash scripts/reproduce_numbers.sh
