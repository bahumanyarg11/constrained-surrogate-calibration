"""Compare a rerun experiment with the stored result, number by number. Wall-clock timings are ignored.

usage: python3 scripts/compare_results.py STORED.json RERUN.json
Exit status 0 if every number agrees exactly, 1 otherwise.
"""
import json, math, sys

TIMING = ('_min', '_ms', '_sec', 'time', 'elapsed', 'speedup')


def walk(a, b, path=''):
    if isinstance(a, dict):
        for k in a:
            if any(t in k for t in TIMING): continue
            if not isinstance(b, dict) or k not in b: yield path + '/' + k, None; continue
            yield from walk(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b): yield path, None; return
        for i, (x, y) in enumerate(zip(a, b)): yield from walk(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b): yield path, 0.0
        else: yield path, abs(a - b)
    elif a != b:
        yield path, None


stored, rerun = (json.load(open(p)) for p in sys.argv[1:3])
res = list(walk(stored, rerun))
diffs = [(p, d) for p, d in res if d is None or d > 0]
if not diffs:
    print(f'Reproduced exactly: all {len(res)} numbers in {sys.argv[1]} agree (timings ignored).')
    sys.exit(0)
num = [d for _, d in diffs if d is not None]
print(f'{len(diffs)} of {len(res)} numbers differ' + (f', largest difference {max(num):.3g}' if num else '') + ':')
for p, d in diffs[:15]: print(f'  {p}: ' + ('missing or structurally different' if d is None else f'{d:.3g}'))
print('Runs are deterministic for a given seed on one platform; a different CPU, BLAS library or PyTorch build can\n'
      'change the last digits during training. The paper reports means and tests over seeds.')
sys.exit(1)
