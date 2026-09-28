"""Optimisers, anchoring coefficient (results/runs_rev) and control distributions: statistics and Tables 4, 8 (CSV)."""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as st
import csv as _csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, TABLES
TAB = TABLES
os.makedirs(TAB, exist_ok=True)
AR = {json.load(open(f))['seed']: json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, 'runs_arch', 'arch_s*.json'))}
L = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, 'runs_rev', 'lam_*.json'))]
C = {json.load(open(f))['seed']: json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, 'runs_rev', 'cma_s*.json'))}
seeds = sorted(C)
def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    return dict(diff=float(d.mean()), p=float(st.ttest_rel(a, b).pvalue), dz=float(d.mean() / d.std(ddof=1)))
S = {'lambda': {}, 'cma': {}, 'tests': {}}
# ---- lambda sweep: lambda=0 is the plain Cholesky head, lambda=0.1 the anchored head of the main remedy runs
LAMS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
def lam_vals(lam, key):
    if lam == 0.0:
        m = {'gap': 'gap', 'true_fid': 'true_fid', 'fid_traj': 'fid_traj', 'claims': 'claims'}[key]
        return np.array([AR[s]['arch']['fno_cholesky'][m] for s in seeds])
    if lam == 0.1:
        return np.array([AR[s]['cure']['fid_traj'] if key == 'fid_traj' else AR[s]['cure']['plain'][key] for s in seeds])
    R = {r['seed']: r['run'] for r in L if abs(r['lam'] - lam) < 1e-12}
    return np.array([R[s][key] for s in seeds])
for lam in LAMS:
    S['lambda'][f'{lam:g}'] = {k: float(lam_vals(lam, k).mean()) for k in ['gap', 'true_fid', 'fid_traj', 'claims']}
    S['lambda'][f'{lam:g}'].update({k + '_sd': float(lam_vals(lam, k).std(ddof=1)) for k in ['gap', 'true_fid', 'fid_traj']})
    if lam > 0:
        S['tests'][f'lam{lam:g}_vs_chol_gap'] = pr(lam_vals(lam, 'gap'), lam_vals(0.0, 'gap'))
        S['tests'][f'lam{lam:g}_vs_ref_true'] = pr(lam_vals(lam, 'true_fid'), np.array([AR[s]['arch']['fno_herm_trace']['true_fid'] for s in seeds]))
# ---- CMA-ES vs Adam
for h in ['herm_trace', 'cholesky', 'anchored']:
    g = lambda k, sub=None: np.array([(C[s]['runs'][h][sub][k] if sub else C[s]['runs'][h][k]) for s in seeds])
    S['cma'][h] = {'claims': float(g('claims').mean()), 'true_fid': float(g('true_fid').mean()), 'gap': float(g('gap').mean()),
                   'gap_sd': float(g('gap').std(ddof=1)), 'err': float(g('err_at_uopt').mean()),
                   'adam_gap': float(g('gap', 'adam').mean()), 'adam_true': float(g('true_fid', 'adam').mean()),
                   'adam_claims': float(g('claims', 'adam').mean()), 'evals': int(g('evals').mean()),
                   'sat': float(g('sat_frac').mean()), 'ood': float(g('ood_std_ratio').mean())}
gc = lambda h, k='gap': np.array([C[s]['runs'][h][k] for s in seeds])
S['tests']['cma_chol_vs_ref_gap'] = pr(gc('cholesky'), gc('herm_trace'))
S['tests']['cma_anch_vs_chol_gap'] = pr(gc('anchored'), gc('cholesky'))
S['tests']['cma_anch_vs_ref_true'] = pr(gc('anchored', 'true_fid'), gc('herm_trace', 'true_fid'))
S['tests']['cma_vs_adam_chol_gap'] = pr(gc('cholesky'), np.array([C[s]['runs']['cholesky']['adam']['gap'] for s in seeds]))
S['n'] = len(seeds)
# ---- control distribution: training controls regenerated exactly from the seeds; optimised controls from the main study
CTRL_MAX, NSEG, NQ = 1.0, 20, 4
train_amp = []
for s in seeds:
    rng = np.random.default_rng(s)
    for i in range(120):
        rng.standard_normal(16); rng.standard_normal(16)          # random_rho draws two length-16 normal vectors
        sg = CTRL_MAX * np.tanh(rng.standard_normal((NSEG, NQ))); sg[1:-1] = (sg[:-2] + sg[1:-1] + sg[2:]) / 3
        if i < 90: train_amp.append(sg.ravel())
train_amp = np.concatenate(train_amp)
lo, hi = np.quantile(train_amp, [0.025, 0.975])
main = {json.load(open(f))['seed']: json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, 'runs', 'seed_*.json'))}
U = {h: np.concatenate([np.array(main[s]['heads'][h]['control']['u'])[::5].ravel() for s in seeds]) for h in ['herm_trace', 'cholesky']}
S['dist'] = {'train_sd': float(train_amp.std()), 'train_lo': float(lo), 'train_hi': float(hi),
             'train_outside': 0.05, 'train_sat': float(np.mean(np.abs(train_amp) > 0.9 * CTRL_MAX))}
for h in U:
    S['dist'][h] = {'sd_ratio': float(U[h].std() / train_amp.std()), 'outside': float(np.mean((U[h] < lo) | (U[h] > hi))),
                    'sat': float(np.mean(np.abs(U[h]) > 0.9 * CTRL_MAX))}
json.dump(S, open(os.path.join(RESULTS, 'summary_rev.json'), 'w'), indent=2)

hs = ['herm_trace', 'cholesky', 'anchored']
# ================================ TABLES (CSV) ================================
def csv(name, header, rows):
    with open(f'{TAB}/{name}.csv', 'w', newline='') as fh:
        w = _csv.writer(fh, lineterminator='\n'); w.writerow(header); w.writerows(rows)
    print(f'  {name}.csv')

# Table 4: Adam and CMA-ES against the same trained surrogates
csv('tab04_optimisers',
    ['head', 'adam_claims', 'adam_true', 'adam_gap', 'cmaes_claims', 'cmaes_true', 'cmaes_gap', 'cmaes_evals'],
    [[h, c['adam_claims'], c['adam_true'], c['adam_gap'], c['claims'], c['true_fid'], c['gap'], c['evals']]
     for h in hs for c in [S['cma'][h]]])
# Table 8: anchoring coefficient; lambda = 0 is the plain Cholesky head, 0.1 the value used throughout
csv('tab08_anchoring_coefficient',
    ['lambda', 'fid_traj', 'fid_traj_sd', 'claims', 'true_fid', 'true_fid_sd', 'gap', 'gap_sd'],
    [[f'{lam:g}', d['fid_traj'], d['fid_traj_sd'], d['claims'], d['true_fid'], d['true_fid_sd'], d['gap'], d['gap_sd']]
     for lam in LAMS for d in [S['lambda'][f'{lam:g}']]])
for k, t in S['tests'].items(): print(f'{k:28s} diff={t["diff"]:+.3f} p={t["p"]:.3g} dz={t["dz"]:+.2f}')
d = S['dist']; print(f"controls: train sd={d['train_sd']:.3f} 95%=[{d['train_lo']:.2f},{d['train_hi']:.2f}] sat={d['train_sat']:.3f} | "
                    f"herm sd x{d['herm_trace']['sd_ratio']:.2f} outside={d['herm_trace']['outside']:.2f} sat={d['herm_trace']['sat']:.2f} | "
                    f"chol sd x{d['cholesky']['sd_ratio']:.2f} outside={d['cholesky']['outside']:.2f} sat={d['cholesky']['sat']:.2f}")
