"""Repository layout. Every script reads and writes through these paths, so it can be run from any directory.

RESULTS_DIR and CHECKPOINT_DIR can be set in the environment to write elsewhere, for example when rerunning an
experiment to compare it with the stored result (scripts/verify_one_run.sh).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results'))              # raw per-seed results, summaries
CHECKPOINTS = os.environ.get('CHECKPOINT_DIR', os.path.join(ROOT, 'checkpoints'))   # trained weights (not in git)
TABLES = os.path.join(RESULTS, 'tables')                                            # the article's tables as CSV
