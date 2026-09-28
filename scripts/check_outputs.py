"""Check recomputed outputs against stored ones: the summary statistics (JSON), the tables (CSV) and every number
quoted in the article (numbers.csv). The two live timings of the solver (cost.q6, cost.q8) are reported, not compared.

usage: python3 scripts/check_outputs.py STORED_DIR NEW_DIR
"""
import csv, glob, json, os, sys

ref, new = sys.argv[1:3]
bad = []
for f in sorted(glob.glob(os.path.join(ref, 'summary_*.json'))):
    name = os.path.basename(f)
    if json.load(open(f)) != json.load(open(os.path.join(new, name))): bad.append(name)
tables = sorted(glob.glob(os.path.join(ref, 'tables', '*.csv')))
for f in tables:
    name = os.path.join('tables', os.path.basename(f))
    if open(f).read() != open(os.path.join(new, name)).read(): bad.append(name)
read = lambda p: dict(csv.reader(open(p)))
a, b = read(os.path.join(ref, 'numbers.csv')), read(os.path.join(new, 'numbers.csv'))
timing = {k for k in a if k.startswith('cost.')}
diff = [k for k in a if k not in timing and a[k] != b.get(k)] + [k for k in b if k not in a]
n_json = len(glob.glob(os.path.join(ref, 'summary_*.json')))
if not bad and not diff:
    print(f'Reproduced: all {len(a) - 1 - len(timing)} numbers in results/numbers.csv, {len(tables)} tables and '
          f'{n_json} files of summary statistics agree with the stored values.')
    print('Solver timing (machine-dependent, not compared): ' +
          ', '.join(f'{k} = {b.get(k)} s (stored {a[k]} s)' for k in sorted(timing)))
    sys.exit(0)
for x in bad: print(f'differs: {x}')
for k in diff[:20]: print(f'differs: {k}: stored {a.get(k)}, new {b.get(k)}')
sys.exit(1)
