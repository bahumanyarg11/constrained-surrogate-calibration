"""System variants: dissipation, XY ring and six qubits (results/runs_gen, runs_ext): Table 5 (CSV)."""
import json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, TABLES
TAB = TABLES
os.makedirs(TAB, exist_ok=True)

R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs_gen', '*.json')))]
# second Hamiltonian family (runs_ext/sys2_*): adapt to the runs_gen layout
for f in sorted(glob.glob(os.path.join(RESULTS, 'runs_ext', 'sys2_s*.json'))):
    x = json.load(open(f))
    if not all(f'fno_{h}' in x['runs'] for h in ['herm_trace', 'cholesky']): continue
    R.append({'tag': 'xy', 'seed': x['seed'], 'grape_ceiling': x['grape_ceiling'],
              'heads': {h: {'fid_traj': x['runs'][f'fno_{h}']['fid_traj'], 'surrogate_fid': x['runs'][f'fno_{h}']['claims'],
                            'true_fid': x['runs'][f'fno_{h}']['true_fid'], 'gap': x['runs'][f'fno_{h}']['gap'],
                            'err_at_uopt': x['runs'][f'fno_{h}']['err_at_uopt'], 'neg_mass': x['runs'][f'fno_{h}']['neg_mass'],
                            'calib_err': float('nan')} for h in ['herm_trace', 'cholesky']}})
MAIN = json.load(open(os.path.join(RESULTS, 'summary_stats.json')))
ORDER = [t for t in ['g_strong', 'base', 'g_weak', 'xy', 'q6', 'q6m'] if t == 'base' or any(r['tag'] == t for r in R)]
def cell(tag, head, key):
    if tag == 'base':
        m = MAIN['heads'][head]
        k = {'fid_traj': 'fid_traj', 'surrogate_fid': 'surrogate_fid', 'true_fid': 'true_fid',
             'gap': 'gap', 'err_at_uopt': 'err_at_uopt', 'calib_err': 'calib_err', 'neg_mass': 'neg_mass'}[key]
        return np.array([m[k]['mean']]), m[k]['sd'], MAIN['n_seeds']
    S = [r for r in R if r['tag'] == tag]
    v = np.array([r['heads'][head][key] for r in S])
    return v, (v.std(ddof=1) if len(v) > 1 else 0.0), len(S)
def ceil_(tag):
    if tag == 'base': return MAIN['grape_ceiling']['mean']
    return float(np.mean([r['grape_ceiling'] for r in R if r['tag'] == tag]))

csv = []
for t in ORDER:
    n = cell(t, 'herm_trace', 'gap')[2]
    for h in ['herm_trace', 'cholesky']:
        csv.append([t, h, n, ceil_(t)] + [cell(t, h, k)[0].mean() for k in
                                          ['fid_traj', 'surrogate_fid', 'true_fid', 'gap', 'err_at_uopt', 'calib_err']])
open(f'{TAB}/tab05_system_variants.csv', 'w').write('\n'.join(
    [','.join(['tag', 'head', 'n', 'grape_ceiling', 'fid_traj', 'claims', 'true_fid', 'gap', 'err_at_uopt', 'calib_err'])] +
    [','.join(map(str, r)) for r in csv]))
print('configs:', ORDER)
for t in ORDER:
    for h in ['herm_trace', 'cholesky']:
        print(f'{t:9s} {h:12s} gap={cell(t,h,"gap")[0].mean():+.3f} claims={cell(t,h,"surrogate_fid")[0].mean():.3f} '
              f'delivers={cell(t,h,"true_fid")[0].mean():.3f} err={cell(t,h,"err_at_uopt")[0].mean():.3f}')
print('-> tab05_system_variants.csv')
