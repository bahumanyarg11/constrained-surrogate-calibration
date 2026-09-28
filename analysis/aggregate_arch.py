"""Trunk architectures and scale anchoring (results/runs_arch, runs_ext): statistics and Tables 6-7 (CSV)."""
import json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as st
import csv as _csv
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, TABLES
TAB = TABLES
os.makedirs(TAB, exist_ok=True)
TRUNKS = ['fno', 'transformer', 'mixer']
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs_arch', 'arch_s*.json')))]
N = len(R); print(f'{N} seeds')
A = lambda tk, h, m: np.array([r['arch'][f'{tk}_{h}'][m] for r in R], float)
C = lambda *k: np.array([(r['cure'][k[0]] if len(k) == 1 else r['cure'][k[0]][k[1]]) for r in R], float)
def bci(a):
    a = np.asarray(a, float)
    if len(a) < 3 or np.allclose(a, a[0]): return a.mean(), a.mean()
    try:
        c = st.bootstrap((a,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa',
                         random_state=0).confidence_interval
        return float(c.low), float(c.high)
    except Exception:
        h = st.t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)); return a.mean()-h, a.mean()+h
def pr(a, b):
    d = np.asarray(a, float)-np.asarray(b, float)
    lo, hi = bci(d)
    return dict(diff=float(d.mean()), ci=[lo, hi], p_t=float(st.ttest_rel(a, b).pvalue),
                p_w=float(st.wilcoxon(a, b).pvalue), dz=float(d.mean()/d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan)

S = {'n': N, 'arch': {}, 'cure': {}, 'tests': {}}
for tk in TRUNKS:
    for h in ['herm_trace', 'cholesky']:
        S['arch'][f'{tk}_{h}'] = {m: float(A(tk, h, m).mean()) for m in
                                  ['fid_traj', 'neg_mass', 'claims', 'true_fid', 'gap', 'err_at_uopt']}
        S['arch'][f'{tk}_{h}']['gap_ci'] = list(bci(A(tk, h, 'gap')))
    S['tests'][f'{tk}_gap'] = pr(A(tk, 'cholesky', 'gap'), A(tk, 'herm_trace', 'gap'))
    S['tests'][f'{tk}_claims'] = pr(A(tk, 'cholesky', 'claims'), A(tk, 'herm_trace', 'claims'))
    S['tests'][f'{tk}_err'] = pr(A(tk, 'cholesky', 'err_at_uopt'), A(tk, 'herm_trace', 'err_at_uopt'))
S['cure'] = {'fid_traj': float(C('fid_traj').mean()), 'neg_mass': float(C('neg_mass').mean()),
             'logmass_in_dist': float(C('logmass_in_dist').mean())}
for mo in ['plain', 'gated']:
    S['cure'][mo] = {m: float(C(mo, m).mean()) for m in ['claims', 'true_fid', 'gap', 'logmass_uopt']}
    S['cure'][mo]['gap_ci'] = list(bci(C(mo, 'gap')))
    S['cure'][mo]['ratio'] = float(C('ratio_'+mo).mean())
S['tests']['cure_vs_chol_gap'] = pr(C('plain', 'gap'), A('fno', 'cholesky', 'gap'))
S['tests']['cure_vs_chol_true'] = pr(C('plain', 'true_fid'), A('fno', 'cholesky', 'true_fid'))
S['tests']['cure_vs_chol_acc'] = pr(C('fid_traj'), A('fno', 'cholesky', 'fid_traj'))
S['tests']['cure_vs_herm_true'] = pr(C('plain', 'true_fid'), A('fno', 'herm_trace', 'true_fid'))
S['tests']['gated_obj_vs_plain'] = pr(C('gated', 'gap'), C('plain', 'gap'))
json.dump(S, open(os.path.join(RESULTS, 'summary_arch.json'), 'w'), indent=2)

# ---- scale anchoring across systems and trunks
XR = {}
for f in glob.glob(os.path.join(RESULTS, 'runs_ext', '*.json')):
    x = json.load(open(f)); XR.setdefault(x['mode'], {})[x['seed']] = x
COND = []   # (label, n, {head: {metric: array}})
def pack(get, seeds):
    return {h: {m: np.array([get(sd, h, m) for sd in seeds], float) for m in ['gap', 'true_fid', 'fid_traj', 'claims', 'neg_mass']}
            for h in ['herm_trace', 'cholesky', 'anchored']}
sd_a = [r['seed'] for r in R]; byseed = {r['seed']: r for r in R}
def g_fno(sd, h, m):
    r = byseed[sd]
    if h == 'anchored':
        return r['cure'][m] if m in ('fid_traj', 'neg_mass') else r['cure']['plain'][m]
    return r['arch'][f'fno_{h}'][m]
COND.append(('FNO, ZZ chain', len(sd_a), pack(g_fno, sd_a)))
if 'sys2' in XR:
    s2 = sorted(sd for sd, x in XR['sys2'].items() if 'fno_cholesky_gated' in x['runs'])
    g_s2 = lambda sd, h, m: XR['sys2'][sd]['runs'][f'fno_{"cholesky_gated" if h == "anchored" else h}'][m]
    if s2: COND.append(('FNO, XY ring', len(s2), pack(g_s2, s2)))
if 'curetrunks' in XR:
    for tk, lab in [('mixer', 'MLP-Mixer, ZZ chain'), ('transformer', 'Transformer, ZZ chain')]:
        sc = sorted(sd for sd in XR['curetrunks'] if sd in byseed)
        g_ct = (lambda tk: lambda sd, h, m: XR['curetrunks'][sd]['runs'][f'{tk}_cholesky_gated'][m] if h == 'anchored'
                else byseed[sd]['arch'][f'{tk}_{h}'][m])(tk)
        if sc: COND.append((lab, len(sc), pack(g_ct, sc)))
S['cure_conditions'] = {lab.replace('\n', ' '): {'n': n, **{h: {m: float(v.mean()) for m, v in d.items()} for h, d in hd.items()}}
                        for lab, n, hd in COND}
json.dump(S, open(os.path.join(RESULTS, 'summary_arch.json'), 'w'), indent=2)
# ================================ TABLES (CSV) ================================
def csv(name, header, rows):
    with open(f'{TAB}/{name}.csv', 'w', newline='') as fh:
        w = _csv.writer(fh, lineterminator='\n'); w.writerow(header); w.writerows(rows)
    print(f'  {name}.csv')

# Table 6: both heads for each trunk, then the paired difference (Cholesky minus reference) with its t-test p value
rows = []
for tk in TRUNKS:
    for h in ['herm_trace', 'cholesky']:
        d = S['arch'][f'{tk}_{h}']
        rows.append([tk, h, d['fid_traj'], d['claims'], d['true_fid'], d['gap'], d['gap_ci'][0], d['gap_ci'][1],
                     d['err_at_uopt'], ''])
    t = S['tests']
    rows.append([tk, 'cholesky_minus_herm_trace', '', t[f'{tk}_claims']['diff'], '', t[f'{tk}_gap']['diff'],
                 t[f'{tk}_gap']['ci'][0], t[f'{tk}_gap']['ci'][1], t[f'{tk}_err']['diff'], t[f'{tk}_gap']['p_t']])
csv('tab06_architectures',
    ['trunk', 'head', 'fid_traj', 'claims', 'true_fid', 'gap', 'gap_ci_lo', 'gap_ci_hi', 'err_at_uopt', 'p_gap'], rows)

# Table 7: the three heads for each condition, then the paired difference in gap (anchored minus Cholesky)
rows = []
for lab, n, hd in COND:
    for h in ['herm_trace', 'cholesky', 'anchored']:
        d = hd[h]
        rows.append([lab, n, h, d['fid_traj'].mean(), d['neg_mass'].mean(), d['claims'].mean(), d['true_fid'].mean(),
                     d['gap'].mean(), ''])
    t = pr(hd['anchored']['gap'], hd['cholesky']['gap'])
    rows.append([lab, n, 'anchored_minus_cholesky', '', '', '', '', t['diff'], t['p_t']])
csv('tab07_scale_anchoring', ['condition', 'n', 'head', 'fid_traj', 'neg_mass', 'claims', 'true_fid', 'gap', 'p_gap'], rows)

for lab, n, hd in COND:
    t = pr(hd['anchored']['gap'], hd['cholesky']['gap']); u = pr(hd['anchored']['true_fid'], hd['herm_trace']['true_fid'])
    print(f'{lab.replace(chr(10)," "):24s} n={n:2d} gap: herm {hd["herm_trace"]["gap"].mean():+.3f} chol {hd["cholesky"]["gap"].mean():+.3f} '
          f'anch {hd["anchored"]["gap"].mean():+.3f} (anch-chol {t["diff"]:+.3f}, p={t["p_t"]:.3g}) | true: herm '
          f'{hd["herm_trace"]["true_fid"].mean():.3f} anch {hd["anchored"]["true_fid"].mean():.3f} (p={u["p_t"]:.3g}) | '
          f'acc chol {hd["cholesky"]["fid_traj"].mean():.3f} anch {hd["anchored"]["fid_traj"].mean():.3f}')

for tk in TRUNKS:
    t = S['tests'][f'{tk}_gap']; print(f'{tk:12s} gap diff={t["diff"]:+.3f} CI=[{t["ci"][0]:+.3f},{t["ci"][1]:+.3f}] p={t["p_t"]:.4f} dz={t["dz"]:+.2f}')
