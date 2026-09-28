# Diagnosing and repairing miscalibration in physically constrained neural-operator surrogates for quantum control

Code and raw results for the article by **R. G. Bahumanya** and **Rushikesh Anil Padaki** (Department of Computer
Science and Engineering – Artificial Intelligence and Machine Learning, R. V. College of Engineering, Bengaluru,
India). Manuscript under review. Every statistic, table and number reported in the article is recomputed from the raw
per-seed results in this repository, and each can be checked against the stored value with one command.

## What the study shows

- When a neural-operator surrogate of four-qubit Lindblad dynamics is used as the objective of an optimiser, a
  Cholesky output head, whose density matrices are positive semidefinite (PSD) by construction, predicts a fidelity of
  0.404 for its own optimised pulse, 86% of the GRAPE optimum (0.471). The pulse delivers 0.004. A standard
  Hermitian + trace head predicts 0.086 and delivers 0.066.
- At the optimised pulse the two heads are about equally inaccurate (trajectory-mean trace distance 0.768 against
  0.741, Wilcoxon p = 0.13). The failure is miscalibration, not prediction error.
- It replicates in a second Hamiltonian family (XY ring, gap +0.476), across T1 = 5–50 µs and under a derivative-free
  optimiser (CMA-ES, gap +0.394). It is absent for a Transformer trunk and at six qubits, where the optimiser cannot
  exploit the surrogate.
- Scale anchoring, one extra loss term on the normalisation constant the head discards, removes 94% of the gap
  (0.400 to 0.025), restores the delivered fidelity and keeps the output PSD by construction.

## Repository layout

```
src/            physics.py   Lindblad model and RK4 integrator
                models.py    FNO, Transformer and MLP-Mixer trunks; the five output heads, including scale anchoring
                paths.py     where results and checkpoints live
experiments/    experiment workers, one seed per call; write raw results to results/
analysis/       raw results -> summary statistics, the article's tables and every number quoted in the text
results/        runs*/            raw per-seed results (JSON), one file per seed and experiment
                summary_*.json    summary statistics, including all paired tests and bootstrap intervals
                tables/           the article's tables as CSV (tab01_... is Table 1, and so on)
                numbers.csv       every number quoted in the text of the article
                logs/             logs of the original runs
scripts/        reproduce_numbers.sh, verify_one_run.sh, run_all.sh
config.json     every hyperparameter and seed
```

## Requirements

Python 3.13 (tested with 3.13.7) and four packages: PyTorch 2.9.1, NumPy, SciPy and pycma. Everything runs on a
CPU; no GPU is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

**1. Recompute every statistic, table and number from the stored raw results** (a few seconds, no training):

```bash
bash scripts/reproduce_numbers.sh
```

This reruns the analysis and checks each output against the stored value: the 656 numbers in
`results/numbers.csv`, the 12 tables and the summary statistics. The only values not compared are two timings of the
solver used for the eight-qubit cost estimate (`cost.q6`, `cost.q8`), which depend on the machine.

**2. Rerun one experiment and compare it with the stored result** (about 4 minutes on one core):

```bash
bash scripts/verify_one_run.sh
```

This retrains the three trunks, both heads and the anchored head for seed 42, reruns the optimisation and the exact
evaluation, and checks every number against `results/runs_arch/arch_s42.json`.

**3. Rerun every experiment from scratch** (about four hours on a 10-core CPU):

```bash
PAR=10 bash scripts/run_all.sh
```

The script ends by recomputing all numbers and checking them against the stored values.

Individual experiments, one seed per call:

| Experiment | Command | Seeds |
|---|---|---|
| Main study: all heads, GRAPE, random search, ablations (`full`) | `python3 experiments/worker.py <seed> 1.0 full\|core` | 42 123 456 789 999 (full), 7 13 21 33 55 (core) |
| Dissipation variants | `python3 experiments/worker_gen.py <seed> 4 5.0 g_strong` and `... 4 50.0 g_weak` | 42 123 456 |
| Six qubits, reduced and matched budget | `python3 experiments/worker_gen.py <seed> 6 15.0 q6`; with `GEN_NSAMP=120 GEN_EPOCHS=120`: `... q6m` | 42 123 456 |
| Deep ensembles | `python3 experiments/worker_ens.py <seed>` | 42 123 456 789 999 |
| Trunk architectures and scale anchoring | `python3 experiments/worker_arch.py <seed>` | the 10 main seeds |
| Second Hamiltonian family (XY ring) | `python3 experiments/worker_ext.py <seed> sys2` | 42 123 456 789 999 |
| Anchoring on the MLP-Mixer and Transformer trunks | `python3 experiments/worker_ext.py <seed> curetrunks` | the 10 main seeds |
| Derivative-free optimisation (CMA-ES) | `python3 experiments/worker_rev.py cma <seed>` (needs the checkpoints) | the 10 main seeds |
| Anchoring coefficient | `python3 experiments/worker_rev.py lam <seed> <lambda>` (needs the checkpoints) | the 10 main seeds; lambda 0.01 0.03 0.3 1 |

Runs are deterministic for a given seed on one platform: independent reruns, including reloaded checkpoints and
retrained models, reproduce every reported control gap exactly (50 comparisons, largest difference 0). A different
CPU, BLAS library or PyTorch build can change the last digits of a training run.

## Trained checkpoints

The 50 trained models of the main study are attached to the
[v1.0 release](https://github.com/bahumanyarg11/constrained-surrogate-calibration/releases/tag/v1.0)
(`checkpoints.zip`, 118 MB). They are needed only to rerun the CMA-ES and anchoring-coefficient experiments without
first rerunning the main study. Unzip in the repository root:

```bash
gh release download v1.0 --pattern checkpoints.zip && unzip checkpoints.zip
```

## Where each result comes from

| Script | Output | In the article |
|---|---|---|
| `analysis/aggregate.py` | `summary_stats.json`, `tables/tab01_*`, `tab02_*`, `tab03_*`, `tab10_*`, `tab11_*`, `tab12_*` | the ten planned comparisons; Tables 1–3 and 10–12 |
| `analysis/aggregate_rev.py` | `summary_rev.json`, `tables/tab04_*`, `tab08_*` | optimisers, anchoring coefficient, control distributions; Tables 4 and 8 |
| `analysis/aggregate_gen.py` | `tables/tab05_*` | system variants; Table 5 |
| `analysis/aggregate_arch.py` | `summary_arch.json`, `tables/tab06_*`, `tab07_*` | trunk architectures and scale anchoring; Tables 6 and 7 |
| `analysis/aggregate_ens.py` | `summary_ens.json`, `tables/tab09_*` | deep ensembles; Table 9 |
| `analysis/paper_numbers.py` | `numbers.csv` | every number in the text |

## Citation

```bibtex
@unpublished{bahumanya2026miscalibration,
  author = {Bahumanya, R. G. and Padaki, Rushikesh Anil},
  title  = {Diagnosing and repairing miscalibration in physically constrained neural-operator surrogates for quantum control},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
