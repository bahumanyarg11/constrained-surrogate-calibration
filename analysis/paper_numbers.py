r"""Every number quoted in the text of the article, recomputed from the raw per-seed results and the aggregated
statistics, written to results/numbers.csv (key, value). Keys are named after the quantity: for example gap.ch is the
control gap of the Cholesky head and h3.ph the Holm-corrected p value of planned comparison H3.
The eight-qubit cost estimate (cost.q6, cost.q8) is a live timing of the solver and varies with the machine.
"""
import os, sys, json, glob, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats as st
V = {}
def put(k, v): V[k] = v
def fx(x, d=3): return f'{x:.{d}f}'
def sx(x, d=3): return f'{x:+.{d}f}'
def sci(x, d=1):
    if x == 0: return '0'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10: m /= 10; e += 1
    return f'{m:.{d}f}\\times10^{{{e}}}'
def pf(p):
    if not np.isfinite(p): return 'n/a'
    if p >= 0.1: return f'{p:.2f}'
    if p >= 0.01: return f'{p:.3f}'
    if p >= 0.001: return f'{p:.4f}'
    return sci(p)
def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    out = dict(diff=d.mean(), p=st.ttest_rel(a, b).pvalue if len(d) > 1 and d.std() > 0 else np.nan,
               dz=d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan)
    try: out['pw'] = st.wilcoxon(a, b).pvalue
    except Exception: out['pw'] = np.nan
    return out
def put_test(prefix, t, d=3):
    put(f'{prefix}.diff', sx(t['diff'], d)); put(f'{prefix}.p', pf(t['p']))
    put(f'{prefix}.dz', sx(t['dz'], 2) if np.isfinite(t['dz']) else 'n/a'); put(f'{prefix}.pw', pf(t['pw']))

# ---------------------------------------------------------------- main study
S = json.load(open(os.path.join(RESULTS, 'summary_stats.json')))
H = S['heads']; SH = dict(unconstrained='un', herm_trace='ht', eigh_clip='ec', cholesky='ch', linear='li')
put('main.n', str(S['n_seeds']))
for h, k in SH.items():
    for m, key in [('fid_final', 'fidfinal'), ('fid_traj', 'fidtraj'), ('surrogate_fid', 'claim'),
                   ('true_fid', 'true'), ('gap', 'gap'), ('err_at_uopt', 'err'), ('calib_err', 'calib'),
                   ('err_in_dist', 'errind')]:
        put(f'{key}.{k}', fx(H[h][m]['mean']))
    put(f'neg.{k}', sci(H[h]['neg_mass']['mean']) if H[h]['neg_mass']['mean'] > 0 else '0')
    put(f'speed.{k}', f'{H[h]["speedup"]["mean"]:.0f}')
    for m in ['fno_in', 'fno_ex', 'freeze', 'mixed', 'eq']:
        put(f'extrap.{m}.{k}', fx(H[h]['extrap_' + m]['mean']))
put('grape.mean', fx(S['grape_ceiling']['mean'])); put('grape.lo', fx(S['grape_ceiling']['ci_lo']))
put('grape.hi', fx(S['grape_ceiling']['ci_hi'])); put('grape.calls', str(S['grape_calls']))
put('rs.mean', fx(S['random_search']['mean']))
put('ratio.claim', f'{H["cholesky"]["surrogate_fid"]["mean"] / H["herm_trace"]["surrogate_fid"]["mean"]:.1f}')
put('ratio.calib', f'{H["cholesky"]["calib_err"]["mean"] / H["herm_trace"]["calib_err"]["mean"]:.1f}')
put('ratio.ceiling', f'{100 * H["cholesky"]["surrogate_fid"]["mean"] / S["grape_ceiling"]["mean"]:.0f}')
negs = [H[h]['neg_mass']['mean'] for h in ['unconstrained', 'herm_trace', 'cholesky']]
put('neg.span', f'{math.log10(max(negs)) - math.log10(min(negs)):.0f}')
T = S['tests']
for i, key in enumerate(['H1_final_vs_traj', 'H2_chol_vs_herm_acc', 'H3_chol_vs_herm_gap', 'H4_extrap_vs_freeze',
                         'H5_err_chol_vs_herm', 'H6_claim_chol_vs_herm', 'H7_mitigation', 'H8_fno_vs_linear',
                         'H9_warm_vs_cold', 'H10_calib_chol_vs_herm'], 1):
    t = T[key]; d = 4 if i == 4 else 3
    put(f'h{i}.diff', sx(t['mean_diff'], d)); put(f'h{i}.lo', sx(t['ci_lo'], d)); put(f'h{i}.hi', sx(t['ci_hi'], d))
    put(f'h{i}.pt', pf(t['p_t'])); put(f'h{i}.pw', pf(t['p_w'])); put(f'h{i}.ph', pf(t['p_holm']))
    put(f'h{i}.dz', sx(t['dz'], 2) if t['dz'] == t['dz'] else 'n/a')
    put(f'h{i}.sw', pf(t['shapiro_p']) if t.get('shapiro_p') == t.get('shapiro_p') else 'n/a')
put('h.nsig', str(sum(1 for k, t in T.items() if isinstance(t, dict) and t.get('reject_holm'))))
A = S['ablation']
for m in [0, 4, 8, 16, 32]: put(f'abl.m{m}', fx(A[f'modes_{m}']['fid_traj']['mean']))
for n in [30, 60, 90]:
    put(f'abl.n{n}.ht', fx(A[f'ntrain_herm_trace_{n}']['fid_traj']['mean']))
    put(f'abl.n{n}.ch', fx(A[f'ntrain_cholesky_{n}']['fid_traj']['mean']))
MG, WS = S['mitigation']['cholesky'], S['warmstart']['cholesky']
put('mit.before', fx(MG['before']['mean'])); put('mit.ctrl', fx(MG['ctrl_after']['mean'])); put('mit.after', fx(MG['after']['mean']))
put('warm.warm', fx(WS['warm_best']['mean'])); put('warm.cold', fx(WS['cold_at_same_budget']['mean']))
put('warm.init', fx(WS['init_fid']['mean']))
# per-seed audit: does any surrogate-guided pulse beat that seed's own random-search result?
main_runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs', 'seed_*.json')))]
beats = sum(1 for r in main_runs for h in SH if r['heads'][h]['control']['true_fid'] > r['random_search']['best'])
put('audit.beatsrs', str(beats)); put('audit.pairs', str(len(main_runs) * len(SH)))
best_true = max(r['heads'][h]['control']['true_fid'] for r in main_runs for h in SH)
put('audit.besttrue', fx(best_true))

# ---------------------------------------------------------------- ensembles
E = json.load(open(os.path.join(RESULTS, 'summary_ens.json')))
put('ens.n', str(E['n'])); put('ens.K', str(E['K'])); put('ens.kappa', f'{E["kappa"]:g}')
for h, k in [('herm_trace', 'ht'), ('cholesky', 'ch')]:
    for mo in ['single', 'mean', 'lcb']:
        d = E['heads'][h][mo]
        put(f'ens.{mo}.gap.{k}', sx(d['gap'])); put(f'ens.{mo}.true.{k}', fx(d['true']))
        put(f'ens.{mo}.claim.{k}', fx(d['claims'])); put(f'ens.{mo}.dis.{k}', f'{d["dis_ratio"]:.1f}')
        put(f'ens.{mo}.gaplo.{k}', sx(d['gap_ci'][0])); put(f'ens.{mo}.gaphi.{k}', sx(d['gap_ci'][1]))
for key, t in E['tests'].items():
    k = key.split('_')[0].lower()
    put(f'{k}.diff', sx(t['mean_diff'], 1 if k == 'e1' else 3)); put(f'{k}.p', pf(t['p_t']))
    put(f'{k}.pw', pf(t['p_w'])); put(f'{k}.dz', sx(t['dz'], 2))

# ---------------------------------------------------------------- generalisation (raw runs)
G = {}
for f in glob.glob(os.path.join(RESULTS, 'runs_gen', '*.json')):
    r = json.load(open(f)); G.setdefault(r['tag'], []).append(r)
for tag, key in [('g_strong', 'gs'), ('g_weak', 'gw'), ('q6', 'qr'), ('q6m', 'qm')]:
    R_ = G[tag]; put(f'{key}.n', str(len(R_))); put(f'{key}.ceil', fx(np.mean([r['grape_ceiling'] for r in R_])))
    g = lambda h, m: np.array([r['heads'][h][m] for r in R_])
    for h, hk in [('herm_trace', 'ht'), ('cholesky', 'ch')]:
        for m, mk in [('gap', 'gap'), ('surrogate_fid', 'claim'), ('true_fid', 'true'), ('err_at_uopt', 'err'),
                      ('fid_traj', 'fidtraj')]:
            put(f'{key}.{mk}.{hk}', sx(g(h, m).mean()) if m == 'gap' else fx(g(h, m).mean()))
        put(f'{key}.neg.{hk}', sci(g(h, 'neg_mass').mean()) if g(h, 'neg_mass').mean() > 0 else '0')
    for m, mk in [('gap', 'gap'), ('surrogate_fid', 'claim'), ('err_at_uopt', 'err'), ('fid_traj', 'acc')]:
        put_test(f'{key}.d{mk}', paired(g('cholesky', m), g('herm_trace', m)))
put('chance.q6', fx(1 / 64, 4))

# ---------------------------------------------------------------- architectures + remedy (raw runs)
AR = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, 'runs_arch', 'arch_s*.json')))]
seeds_arch = [r['seed'] for r in AR]; put('arch.n', str(len(AR)))
a = lambda tk, h, m: np.array([r['arch'][f'{tk}_{h}'][m] for r in AR])
for tk, tkk in [('fno', 'fno'), ('transformer', 'tf'), ('mixer', 'mx')]:
    for h, hk in [('herm_trace', 'ht'), ('cholesky', 'ch')]:
        for m, mk in [('fid_traj', 'fidtraj'), ('claims', 'claim'), ('true_fid', 'true'), ('gap', 'gap'),
                      ('err_at_uopt', 'err')]:
            put(f'arch.{tkk}.{mk}.{hk}', sx(a(tk, h, m).mean()) if m == 'gap' else fx(a(tk, h, m).mean()))
    for m, mk in [('gap', 'gap'), ('claims', 'claim'), ('err_at_uopt', 'err'), ('fid_traj', 'acc')]:
        put_test(f'arch.{tkk}.d{mk}', paired(a(tk, 'cholesky', m), a(tk, 'herm_trace', m)))
c = lambda *k: np.array([(r['cure'][k[0]] if len(k) == 1 else r['cure'][k[0]][k[1]]) for r in AR])
put('cure.fno.gap', sx(c('plain', 'gap').mean())); put('cure.fno.true', fx(c('plain', 'true_fid').mean()))
put('cure.fno.claim', fx(c('plain', 'claims').mean())); put('cure.fno.acc', fx(c('fid_traj').mean()))
put('cure.fno.neg', sci(c('neg_mass').mean())); put('cure.fno.ratio', f'{c("ratio_plain").mean():.1f}')
put('cure.fno.gatedgap', sx(c('gated', 'gap').mean())); put('cure.fno.gatedtrue', fx(c('gated', 'true_fid').mean()))
put_test('cure.fno.dgap', paired(c('plain', 'gap'), a('fno', 'cholesky', 'gap')))
put_test('cure.fno.dtrue', paired(c('plain', 'true_fid'), a('fno', 'cholesky', 'true_fid')))
put_test('cure.fno.dacc', paired(c('fid_traj'), a('fno', 'cholesky', 'fid_traj')))
put_test('cure.fno.vsht', paired(c('plain', 'true_fid'), a('fno', 'herm_trace', 'true_fid')))
put_test('cure.fno.dgated', paired(c('gated', 'gap'), c('plain', 'gap')))
put('cure.fno.reduction', f'{100 * (1 - c("plain", "gap").mean() / a("fno", "cholesky", "gap").mean()):.0f}')

# ---------------------------------------------------------------- extension: second system + remedy on other trunks
X = {}
for f in glob.glob(os.path.join(RESULTS, 'runs_ext', '*.json')):
    r = json.load(open(f)); X.setdefault(r['mode'], []).append(r)
if 'sys2' in X:
    R2 = X['sys2']; put('sys2.n', str(len(R2)))
    put('sys2.ceil', fx(np.mean([r['grape_ceiling'] for r in R2]))); put('sys2.rs', fx(np.mean([r['random_search'] for r in R2])))
    s2 = lambda h, m: np.array([r['runs'][f'fno_{h}'][m] for r in R2])
    for h, hk in [('herm_trace', 'ht'), ('cholesky', 'ch'), ('cholesky_gated', 'an')]:
        for m, mk in [('fid_traj', 'fidtraj'), ('claims', 'claim'), ('true_fid', 'true'), ('gap', 'gap'),
                      ('err_at_uopt', 'err')]:
            put(f'sys2.{mk}.{hk}', sx(s2(h, m).mean()) if m == 'gap' else fx(s2(h, m).mean()))
    for m, mk in [('gap', 'gap'), ('claims', 'claim'), ('err_at_uopt', 'err'), ('fid_traj', 'acc')]:
        put_test(f'sys2.d{mk}', paired(s2('cholesky', m), s2('herm_trace', m)))
    for m, mk in [('gap', 'gap'), ('true_fid', 'true'), ('fid_traj', 'acc')]:
        put_test(f'sys2.cure.d{mk}', paired(s2('cholesky_gated', m), s2('cholesky', m)))
    put_test('sys2.cure.vsht', paired(s2('cholesky_gated', 'true_fid'), s2('herm_trace', 'true_fid')))
if 'curetrunks' in X:
    CT = {r['seed']: r for r in X['curetrunks']}
    common = [s for s in seeds_arch if s in CT]; put('ct.n', str(len(common)))
    for tk, tkk in [('mixer', 'mx'), ('transformer', 'tf')]:
        an = lambda m: np.array([CT[s]['runs'][f'{tk}_cholesky_gated'][m] for s in common])
        ch = lambda m: np.array([next(r for r in AR if r['seed'] == s)['arch'][f'{tk}_cholesky'][m] for s in common])
        ht = lambda m: np.array([next(r for r in AR if r['seed'] == s)['arch'][f'{tk}_herm_trace'][m] for s in common])
        for m, mk in [('gap', 'gap'), ('true_fid', 'true'), ('fid_traj', 'acc'), ('claims', 'claim')]:
            put(f'ct.{tkk}.{mk}.an', sx(an(m).mean()) if m == 'gap' else fx(an(m).mean()))
            put_test(f'ct.{tkk}.d{mk}', paired(an(m), ch(m)))
        put_test(f'ct.{tkk}.vsht', paired(an('true_fid'), ht('true_fid')))

# ---------------------------------------------------------------- revision: lambda sweep, CMA-ES, control distribution, budgets
RV = json.load(open(os.path.join(RESULTS, 'summary_rev.json')))
put('rev.n', str(RV['n']))
for lam, k in [('0', '0'), ('0.01', '001'), ('0.03', '003'), ('0.1', '01'), ('0.3', '03'), ('1', '1')]:
    d = RV['lambda'][lam]
    put(f'lam.{k}.gap', sx(d['gap'])); put(f'lam.{k}.true', fx(d['true_fid'])); put(f'lam.{k}.acc', fx(d['fid_traj']))
lp = [RV['tests'][f'lam{l}_vs_chol_gap']['p'] for l in ['0.01', '0.03', '0.1', '0.3', '1']]
put('lam.maxp', pf(max(lp)))
put('lam.gapmin', sx(min(RV['lambda'][l]['gap'] for l in ['0.01', '0.03', '0.1', '0.3', '1'])))
put('lam.gapmax', sx(max(RV['lambda'][l]['gap'] for l in ['0.01', '0.03', '0.1', '0.3', '1'])))
for l, k in [('0.3', '03'), ('1', '1')]:
    t = RV['tests'][f'lam{l}_vs_ref_true']; put(f'lam.{k}.vsref.diff', sx(t['diff'])); put(f'lam.{k}.vsref.p', pf(t['p']))
for h, k in [('herm_trace', 'ht'), ('cholesky', 'ch'), ('anchored', 'an')]:
    c = RV['cma'][h]
    put(f'cma.{k}.claim', fx(c['claims'])); put(f'cma.{k}.true', fx(c['true_fid'])); put(f'cma.{k}.gap', sx(c['gap']))
    put(f'cma.{k}.adamgap', sx(c['adam_gap']))
put('cma.evals', str(RV['cma']['cholesky']['evals']))
for key, k in [('cma_chol_vs_ref_gap', 'dgap'), ('cma_anch_vs_chol_gap', 'dcure'), ('cma_anch_vs_ref_true', 'vsref'),
               ('cma_vs_adam_chol_gap', 'vsadam')]:
    t = RV['tests'][key]; put(f'cma.{k}.diff', sx(t['diff'])); put(f'cma.{k}.p', pf(t['p'])); put(f'cma.{k}.dz', sx(t['dz'], 2))
D = RV['dist']
put('dist.sd', fx(D['train_sd'], 2)); put('dist.hi', fx(D['train_hi'], 2)); put('dist.sat', fx(100 * D['train_sat'], 1))
for h, k in [('herm_trace', 'ht'), ('cholesky', 'ch')]:
    put(f'dist.{k}.sdratio', fx(D[h]['sd_ratio'], 2)); put(f'dist.{k}.out', f'{100 * D[h]["outside"]:.0f}')
    put(f'dist.{k}.sat', f'{100 * D[h]["sat"]:.0f}')
smS, trM = H['herm_trace']['solver_ms']['mean'] / 1000, H['herm_trace']['train_min']['mean']
put('budget.solver_ms', f'{1000 * smS:.0f}'); put('budget.train_min', f'{trM:.1f}')
put('budget.data_s', f'{120 * smS:.0f}'); put('budget.rs_s', f'{300 * smS:.0f}'); put('budget.grape_s', f'{S["grape_calls"] * smS:.0f}')
put('budget.breakeven', f'{(120 * smS + 60 * trM) / (300 * smS):.0f}')

# ---------------------------------------------------------------- reproducibility: independent reruns of the same seeds
byseed_a = {r['seed']: r for r in AR}
diffs = [abs(r['heads'][h]['control']['gap'] - byseed_a[r['seed']]['arch'][f'fno_{h}']['gap'])
         for r in main_runs if r['seed'] in byseed_a for h in ['herm_trace', 'cholesky']]
# the CMA-ES runs reload the saved checkpoints and retrain the anchored head: both must reproduce the earlier values
diffs += [abs(RV_c['runs'][h]['adam']['gap'] - (byseed_a[sd]['cure']['plain']['gap'] if h == 'anchored' else
          next(r for r in main_runs if r['seed'] == sd)['heads'][h]['control']['gap']))
          for sd, RV_c in ((json.load(open(f))['seed'], json.load(open(f))) for f in glob.glob(os.path.join(RESULTS, 'runs_rev', 'cma_s*.json')))
          for h in ['herm_trace', 'cholesky', 'anchored']]
put('repro.n', str(len(diffs))); put('repro.maxdiff', sci(max(diffs)) if max(diffs) > 0 else '0')

# ---------------------------------------------------------------- physics facts, recomputed here
from physics import QSys, random_rho, expand_segments
s05, s01 = QSys(n_qubits=4, total_time=5.0, dt=0.05), QSys(n_qubits=4, total_time=5.0, dt=0.01)
rng = np.random.default_rng(1); r0 = random_rho(16, rng); sg = 0.3 * np.tanh(rng.standard_normal((20, 4)))
a05 = s05.evolve(r0, expand_segments(sg, 100))[-1]; a01 = s01.evolve(r0, expand_segments(sg, 500))[-1]
put('phys.dtconv', sci(0.5 * np.abs(np.linalg.eigvalsh(a05 - a01)).sum()))
g1, nth, T1, T2 = 1 / 15.0, 0.01, 15.0, 10.0
gphi = max(1 / T2 - 0.5 / T1, 0.0)
put('phys.gamma1', fx(g1, 3)); put('phys.gphi', fx(gphi, 3))
put('phys.t2eff', f'{1 / (g1 * (1 + 2 * nth) / 2 + 2 * gphi):.1f}')
import re as _re
pl = open(os.path.join(RESULTS, 'preflight.log')).read()
put('phys.ceil05', _re.search(r'CTRL_MAX=0\.5: GRAPE ceiling\(40it\) fid=([0-9.]+)', pl).group(1)[:4])
for n in [7, 8]: put(f'mem.q{n}', f'{101 * (2 ** n) ** 2 * 16 * 120 / 1e9:.1f}')
import time
s6 = QSys(n_qubits=6, total_time=5.0, dt=0.05); r6 = random_rho(64, rng); c6 = expand_segments(0.3 * np.tanh(rng.standard_normal((20, 6))), 100)
t0 = time.time(); s6.evolve(r6, c6); t6 = time.time() - t0
put('cost.q6', f'{t6:.1f}'); put('cost.q8', f'{t6 * 64:.0f}')

# ---------------------------------------------------------------- write
plain = lambda v: _re.sub(r'\\times10\^\{(-?\d+)\}', r'e\1', str(v))      # 7.6\times10^{-6} -> 7.6e-6
with open(os.path.join(RESULTS, 'numbers.csv'), 'w') as f:
    f.write('key,value\n'); [f.write(f'{k},"{plain(V[k])}"\n') for k in sorted(V)]
print(f'{len(V)} values written | arch n={len(AR)} | ext modes={sorted(X)}')
