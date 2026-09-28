"""Main study (results/runs): statistics for the ten planned comparisons and Tables 1-3 and 10-12 (CSV)."""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as st
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, TABLES

TAB = TABLES
os.makedirs(TAB, exist_ok=True)
LBL = {'unconstrained': 'Unconstrained', 'herm_trace': 'Hermitian + trace', 'eigh_clip': 'Eigenvalue clipping',
       'cholesky': 'Cholesky (PSD)', 'linear': 'Linear (ridge)'}
HEADS = ['unconstrained', 'herm_trace', 'eigh_clip', 'cholesky', 'linear']
NN = ['unconstrained', 'herm_trace', 'eigh_clip', 'cholesky']

runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs', 'seed_*.json')))]
assert runs, 'no runs'
FULL = [r for r in runs if r.get('mode') == 'full']
N = len(runs); print(f'{N} seeds ({len(FULL)} with ablations): {sorted(r["seed"] for r in runs)}')

def dig(d, ks):
    for k in ks: d = d[k]
    return d
g = lambda h, *k: np.array([dig(r['heads'][h], k) for r in runs], float)

# ------------------------------ statistics ----------------------------------
def boot_ci(a, B=10000):
    a = np.asarray(a, float)
    if len(a) < 3 or np.allclose(a, a[0]): return float(a.mean()), float(a.mean())
    try:
        r = st.bootstrap((a,), np.mean, confidence_level=0.95, n_resamples=B,
                         method='BCa', random_state=0).confidence_interval
        return float(r.low), float(r.high)
    except Exception:
        h = st.t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        return float(a.mean() - h), float(a.mean() + h)

def desc(a):
    a = np.asarray(a, float); lo, hi = boot_ci(a)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                ci_lo=lo, ci_hi=hi, n=int(len(a)))

def paired(a, b, alt='two-sided'):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    o = dict(mean_diff=float(d.mean()), n=int(len(d)))
    o['ci_lo'], o['ci_hi'] = boot_ci(d)
    if len(d) < 3 or np.allclose(d, 0):
        o.update(p_t=np.nan, p_w=np.nan, dz=np.nan, shapiro_p=np.nan); return o
    o['dz'] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    o['p_t'] = float(st.ttest_rel(a, b, alternative=alt).pvalue)
    try: o['p_w'] = float(st.wilcoxon(a, b, alternative=alt).pvalue)
    except Exception: o['p_w'] = np.nan
    try: o['shapiro_p'] = float(st.shapiro(d).pvalue)
    except Exception: o['shapiro_p'] = np.nan
    return o

def holm(pairs):
    """pairs: [(name,p)] -> {name: (p_adj, reject@0.05)}"""
    v = sorted([(p, n) for n, p in pairs if np.isfinite(p)]); m = len(v); out = {}; run = 0.0
    for i, (p, n) in enumerate(v):
        run = max(run, min(1.0, (m - i) * p)); out[n] = (run, run < 0.05)
    for n, p in pairs:
        if not np.isfinite(p): out[n] = (np.nan, False)
    return out

def calib(h):
    """mean |F_sur - F_true| along the in-distribution -> u* path."""
    return np.array([np.mean([abs(q['f_sur'] - q['f_true']) for q in r['heads'][h]['control']['path']])
                     for r in runs], float)

S = {'n_seeds': N, 'seeds': sorted(r['seed'] for r in runs), 'ctrl_max': runs[0]['ctrl_max'],
     'n_full': len(FULL), 'env': runs[0]['env'], 'heads': {}}
S['grape_ceiling'] = desc([r['grape_ceiling'] for r in runs])
S['grape_at_30'] = desc([r['grape_at_30'] for r in runs])
S['grape_calls'] = int(runs[0]['grape_calls'])
S['random_search'] = desc([r['random_search']['best'] for r in runs])
for h in HEADS:
    S['heads'][h] = {k: desc(g(h, k)) for k in
                     ['fid_final', 'fid_traj', 'trace_dist', 'neg_mass', 'frac_nonpsd', 'trace_err',
                      'train_min', 'speedup', 'solver_ms', 'fno_ms']}
    S['heads'][h].update({k: desc(g(h, 'control', k)) for k in
                          ['surrogate_fid', 'true_fid', 'gap', 'err_at_uopt', 'err_in_dist',
                           'ood_std_ratio', 'sat_frac']})
    S['heads'][h].update({'extrap_' + k: desc(g(h, 'extrap', k)) for k in
                          ['fno_in', 'fno_ex', 'freeze', 'mixed', 'eq']})
    S['heads'][h]['calib_err'] = desc(calib(h))
S['mitigation'] = {h: {k: desc([r['mitigation'][h][k] for r in runs])
                       for k in ['before', 'after', 'ctrl_after', 'true_before', 'true_after',
                                 'ctrl_true_after', 'sur_after', 'err_after']}
                   for h in ['herm_trace', 'cholesky']}
S['warmstart'] = {h: {k: desc([r['warmstart'][h][k] for r in runs])
                      for k in ['warm_best', 'cold_at_same_budget', 'init_fid']}
                  for h in ['herm_trace', 'cholesky']}
if FULL:
    S['ablation'] = {k: {m: desc([r['ablation'][k][m] for r in FULL])
                         for m in ['fid_traj', 'gap', 'true_fid', 'surrogate_fid'] if m in FULL[0]['ablation'][k]}
                     for k in FULL[0]['ablation']}

# ---- the ten planned comparisons ----
T = {}
T['H1_final_vs_traj'] = paired(g('herm_trace', 'fid_final'), g('herm_trace', 'fid_traj'))
T['H2_chol_vs_herm_acc'] = paired(g('cholesky', 'fid_traj'), g('herm_trace', 'fid_traj'))
T['H3_chol_vs_herm_gap'] = paired(g('cholesky', 'control', 'gap'), g('herm_trace', 'control', 'gap'))
T['H4_extrap_vs_freeze'] = paired(g('herm_trace', 'extrap', 'fno_ex'), g('herm_trace', 'extrap', 'freeze'))
T['H5_err_chol_vs_herm'] = paired(g('cholesky', 'control', 'err_at_uopt'), g('herm_trace', 'control', 'err_at_uopt'))
T['H6_claim_chol_vs_herm'] = paired(g('cholesky', 'control', 'surrogate_fid'), g('herm_trace', 'control', 'surrogate_fid'))
T['H7_mitigation'] = paired([r['mitigation']['cholesky']['after'] for r in runs],
                            [r['mitigation']['cholesky']['ctrl_after'] for r in runs])
T['H8_fno_vs_linear'] = paired(g('herm_trace', 'fid_traj'), g('linear', 'fid_traj'))
T['H9_warm_vs_cold'] = paired([r['warmstart']['cholesky']['warm_best'] for r in runs],
                              [r['grape_at_30'] for r in runs])
T['H10_calib_chol_vs_herm'] = paired(calib('cholesky'), calib('herm_trace'))
HH = holm([(k, v['p_t']) for k, v in T.items()])
for k in T: T[k]['p_holm'], T[k]['reject_holm'] = HH[k]
S['tests'] = T
S['correction'] = f'Holm-Bonferroni over {len(T)} pre-registered hypotheses, family alpha=0.05'
json.dump(S, open(os.path.join(RESULTS, 'summary_stats.json'), 'w'), indent=2)

M = lambda h, k: S['heads'][h][k]['mean']

# ================================ TABLES (CSV) ================================
gc, rsv, mh = S['grape_ceiling'], S['random_search'], ['herm_trace', 'cholesky']
def csv(name, header, rows):
    open(f'{TAB}/{name}.csv', 'w').write('\n'.join([','.join(map(str, header))] + [','.join(map(str, r)) for r in rows]))
    print(f'  {name}.csv')

csv('tab01_planned_comparisons',
    ['hypothesis', 'mean_diff', 'ci_lo', 'ci_hi', 'p_t', 'p_wilcoxon', 'p_holm', 'dz', 'shapiro_p', 'reject'],
    [[k, T[k]['mean_diff'], T[k]['ci_lo'], T[k]['ci_hi'], T[k]['p_t'], T[k]['p_w'], T[k]['p_holm'],
      T[k]['dz'], T[k].get('shapiro_p'), T[k]['reject_holm']] for k in T])
csv('tab02_accuracy',
    ['model', 'fid_final', 'fid_final_ci_lo', 'fid_final_ci_hi', 'fid_traj', 'fid_traj_ci_lo', 'fid_traj_ci_hi',
     'trace_dist', 'neg_mass', 'frac_nonpsd', 'trace_err'],
    [[h, M(h, 'fid_final'), S['heads'][h]['fid_final']['ci_lo'], S['heads'][h]['fid_final']['ci_hi'],
      M(h, 'fid_traj'), S['heads'][h]['fid_traj']['ci_lo'], S['heads'][h]['fid_traj']['ci_hi'],
      M(h, 'trace_dist'), M(h, 'neg_mass'), M(h, 'frac_nonpsd'), M(h, 'trace_err')] for h in HEADS])
csv('tab03_control',
    ['model', 'claim', 'claim_ci_lo', 'claim_ci_hi', 'true', 'true_ci_lo', 'true_ci_hi', 'gap', 'gap_ci_lo', 'gap_ci_hi',
     'err_at_uopt', 'err_in_dist', 'calib_err', 'sat_frac'],
    [[h, M(h, 'surrogate_fid'), S['heads'][h]['surrogate_fid']['ci_lo'], S['heads'][h]['surrogate_fid']['ci_hi'],
      M(h, 'true_fid'), S['heads'][h]['true_fid']['ci_lo'], S['heads'][h]['true_fid']['ci_hi'],
      M(h, 'gap'), S['heads'][h]['gap']['ci_lo'], S['heads'][h]['gap']['ci_hi'],
      M(h, 'err_at_uopt'), M(h, 'err_in_dist'), M(h, 'calib_err'), M(h, 'sat_frac')] for h in HEADS])
csv('tab10_extrapolation', ['model', 'fno_in', 'fno_ex', 'freeze', 'mixed', 'eq'],
    [[h] + [M(h, 'extrap_' + k) for k in ['fno_in', 'fno_ex', 'freeze', 'mixed', 'eq']] for h in NN])
if FULL and 'ablation' in S:
    A = S['ablation']
    csv('tab11_ablations', ['config', 'fid_traj', 'gap', 'true_fid'],
        [[f'modes_{m}', A[f'modes_{m}']['fid_traj']['mean'], A[f'modes_{m}']['gap']['mean'], A[f'modes_{m}']['true_fid']['mean']]
         for m in [0, 4, 8, 16, 32]]
        + [[f'ntrain_{h}_{n}', A[f'ntrain_{h}_{n}']['fid_traj']['mean'], A[f'ntrain_{h}_{n}']['gap']['mean'],
            A[f'ntrain_{h}_{n}']['true_fid']['mean']] for h in ['herm_trace', 'cholesky'] for n in [30, 60, 90]])
MG, WS = S['mitigation'], S['warmstart']
csv('tab12_remedies', ['model', 'gap_before', 'gap_ctrl', 'gap_after', 'true_before', 'true_after', 'warm', 'cold'],
    [[h, MG[h]['before']['mean'], MG[h]['ctrl_after']['mean'], MG[h]['after']['mean'],
      MG[h]['true_before']['mean'], MG[h]['true_after']['mean'], WS[h]['warm_best']['mean'],
      WS[h]['cold_at_same_budget']['mean']] for h in mh])

print(f'\n=== HEADLINES (n={N}) ===')
for h in HEADS:
    print(f'{LBL[h]:22s} F_tj={M(h,"fid_traj"):.3f} neg={M(h,"neg_mass"):.1e} claims={M(h,"surrogate_fid"):.3f} '
          f'delivers={M(h,"true_fid"):.3f} err@u*={M(h,"err_at_uopt"):.3f} calib={M(h,"calib_err"):.3f}')
print(f'GRAPE {gc["mean"]:.3f} [{gc["ci_lo"]:.3f},{gc["ci_hi"]:.3f}] | random search {rsv["mean"]:.3f}')
for k in T:
    t = T[k]
    print(f'{k:24s} diff={t["mean_diff"]:+.4f} p_t={t["p_t"]:.2g} p_holm={t["p_holm"]:.2g} dz={t["dz"]:+.2f} '
          f'{"SIG" if t["reject_holm"] else "ns"}')
