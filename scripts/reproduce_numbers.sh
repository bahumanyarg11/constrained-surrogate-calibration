#!/bin/bash
# Recomputes every statistic, table and number in the article from the raw per-seed results in results/
# (about a minute, no training) and checks each one against the stored value.
set -e
cd "$(dirname "$0")/.."
python3 -c "import numpy, scipy" 2> /dev/null || { echo "Install the requirements first: pip install -r requirements.txt"; exit 1; }

REF=$(mktemp -d); mkdir -p "$REF/tables"                  # the stored values, to compare against
cp results/summary_*.json results/numbers.csv "$REF/"; cp results/tables/*.csv "$REF/tables/"
for a in aggregate aggregate_gen aggregate_ens aggregate_arch aggregate_rev; do
  echo "== analysis/$a.py"; python3 analysis/$a.py > /dev/null
done
echo "== analysis/paper_numbers.py"; python3 analysis/paper_numbers.py > /dev/null
echo
python3 scripts/check_outputs.py "$REF" results
