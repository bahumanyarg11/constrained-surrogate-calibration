"""Deep ensembles (results/runs_ens): statistics and Table 9 (CSV)."""
import json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as st
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, TABLES
TAB = TABLES
os.makedirs(TAB, exist_ok=True)
MODES = ['single', 'mean', 'lcb']
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs_ens', '*.json')))]
N = len(R); print(f'{N} ensemble seeds, K={R[0]["K"]}, kappa={R[0]["kappa"]}')
v = lambda h, mo, k: np.array([r['heads'][h][f'opt_{mo}'][k] for r in R], float)
def ci(a):
    a = np.asarray(a, float)
    if len(a) < 3 or np.allclose(a, a[0]): return a.mean(), a.mean()
    try:
        c = st.bootstrap((a,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa', random_state=0).confidence_interval
        return float(c.low), float(c.high)
    except Exception:
        h = st.t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)); return a.mean()-h, a.mean()+h
S = {'n': N, 'K': R[0]['K'], 'kappa': R[0]['kappa'], 'heads': {}}
for h in ['herm_trace', 'cholesky']:
    S['heads'][h] = {'fid_traj_ens': float(np.mean([r['heads'][h]['fid_traj_ens'] for r in R])),
                     'ind_disagree': float(np.mean([r['heads'][h]['ind_disagree'] for r in R]))}
    for mo in MODES:
        lo, hi = ci(v(h, mo, 'gap'))
        S['heads'][h][mo] = dict(gap=float(v(h, mo, 'gap').mean()), gap_ci=[lo, hi],
                                 claims=float(v(h, mo, 'claims').mean()), true=float(v(h, mo, 'true_fid').mean()),
                                 dis_ratio=float(v(h, mo, 'disagree_ratio').mean()))
# hypotheses
T = {}
def pr(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(mean_diff=float(d.mean()), p_t=float(st.ttest_rel(a, b).pvalue),
                p_w=float(st.wilcoxon(a, b).pvalue), dz=float(d.mean()/d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan)
T['E1_disagree_amplified_chol'] = pr(v('cholesky', 'single', 'disagree_ratio'), v('herm_trace', 'single', 'disagree_ratio'))
T['E2_lcb_vs_single_gap_herm'] = pr(v('herm_trace', 'lcb', 'gap'), v('herm_trace', 'single', 'gap'))
T['E3_lcb_vs_single_gap_chol'] = pr(v('cholesky', 'lcb', 'gap'), v('cholesky', 'single', 'gap'))
T['E4_lcb_vs_single_true_herm'] = pr(v('herm_trace', 'lcb', 'true_fid'), v('herm_trace', 'single', 'true_fid'))
T['E5_lcb_vs_single_true_chol'] = pr(v('cholesky', 'lcb', 'true_fid'), v('cholesky', 'single', 'true_fid'))
T['E6_mean_vs_single_gap_chol'] = pr(v('cholesky', 'mean', 'gap'), v('cholesky', 'single', 'gap'))
T['E7_chol_vs_herm_gap_single'] = pr(v('cholesky', 'single', 'gap'), v('herm_trace', 'single', 'gap'))
S['tests'] = T
json.dump(S, open(os.path.join(RESULTS, 'summary_ens.json'), 'w'), indent=2)

open(f'{TAB}/tab09_ensembles.csv', 'w').write('\n'.join(
    [','.join(['head', 'objective', 'claims', 'true_fid', 'gap', 'gap_ci_lo', 'gap_ci_hi', 'dis_ratio'])] +
    [','.join(map(str, [h, mo, S['heads'][h][mo]['claims'], S['heads'][h][mo]['true'], S['heads'][h][mo]['gap'],
                        *S['heads'][h][mo]['gap_ci'], S['heads'][h][mo]['dis_ratio']]))
     for h in ['herm_trace', 'cholesky'] for mo in MODES]))
for h in ['herm_trace', 'cholesky']:
    print(f'{h}: F_ens={S["heads"][h]["fid_traj_ens"]:.3f} ind_dis={S["heads"][h]["ind_disagree"]:.4f}')
    for mo in MODES:
        d = S['heads'][h][mo]
        print(f'   {mo:7s} claims={d["claims"]:.3f} true={d["true"]:.3f} gap={d["gap"]:+.3f} '
              f'[{d["gap_ci"][0]:+.3f},{d["gap_ci"][1]:+.3f}] dis_ratio={d["dis_ratio"]:.1f}')
print()
for k, t in T.items():
    print(f'{k:30s} diff={t["mean_diff"]:+.4f} p_t={t["p_t"]:.4f} p_w={t["p_w"]:.4f} dz={t["dz"]:+.2f}')
print('-> tab09_ensembles.csv')
