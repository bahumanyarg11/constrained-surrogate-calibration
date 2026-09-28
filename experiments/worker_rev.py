"""Revision experiments requested by the reviewer (the professor).

usage: worker_rev.py lam <seed> <lambda>   sensitivity of scale anchoring to the coefficient lambda (FNO, main system)
       worker_rev.py cma <seed>            derivative-free optimisation (CMA-ES) against the reference, Cholesky and
                                           anchored heads; reference and Cholesky models are the saved main-study
                                           checkpoints, the anchored model is retrained deterministically
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
np.seterr(all='ignore')
import torch
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS, CHECKPOINTS
from physics import QSys, fidelity, trace_dist, random_rho, expand_segments
from models import FNO

MODE, SEED = sys.argv[1], int(sys.argv[2])
LAM = float(sys.argv[3]) if MODE == 'lam' else 0.1
D, NQ, T_TOT, DT, NST, NSEG = 16, 4, 5.0, 0.05, 100, 20
NSAMP, NTR, NVA, EPOCHS, BATCH = 120, 90, 15, 120, 8
CTRL_MAX, OPT_ITERS, CMA_EVALS, CMA_SIGMA = 1.0, 300, 4000, 0.3
OUT = os.path.join(RESULTS, 'runs_rev'); os.makedirs(OUT, exist_ok=True)
CK = CHECKPOINTS; os.makedirs(CK, exist_ok=True)
log = lambda *a: print(f'[{MODE}/s{SEED}' + (f'/lam{LAM:g}' if MODE == 'lam' else '') + ']', *a, flush=True)

sysq = QSys(n_qubits=NQ, total_time=T_TOT, dt=DT)
exc = np.zeros(D, dtype=complex); exc[-1] = 1.0
gnd = np.zeros(D, dtype=complex); gnd[0] = 1.0
G0 = np.outer(gnd, gnd.conj())
tc = lambda a: torch.tensor(np.asarray(a), dtype=torch.complex128)
tf_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
pure = lambda r: float(np.clip(np.real(exc.conj() @ r @ exc), 0, 1))
tpsi = torch.tensor(exc, dtype=torch.complex128)
def fid_phys(p, g):
    w, V = np.linalg.eigh((p + p.conj().T) / 2); w = np.clip(w, 0, None); s = w.sum()
    return 0.0 if s <= 0 else fidelity((V * (w / s)) @ V.conj().T, g)

rng = np.random.default_rng(SEED); R0, CU, RT = [], [], []
for i in range(NSAMP):
    r0 = random_rho(D, rng)
    sg = CTRL_MAX * np.tanh(rng.standard_normal((NSEG, NQ)))
    sg[1:-1] = (sg[:-2] + sg[1:-1] + sg[2:]) / 3
    c = expand_segments(sg, NST)
    R0.append(r0); CU.append(c); RT.append(sysq.evolve(r0, c))
R0, CU, RT = np.stack(R0), np.stack(CU), np.stack(RT)
TR = (R0[:NTR], CU[:NTR], RT[:NTR]); TE = (R0[NTR + NVA:], CU[NTR + NVA:], RT[NTR + NVA:])

def train(head, lam):
    torch.manual_seed(SEED)
    m = FNO(head=head, trunk='fno', dim=D, n_ch=NQ, modes=16, width=64, depth=4, T=NST + 1)
    op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    for ep in range(EPOCHS):
        m.train(); idx = np.random.default_rng(SEED * 1000 + ep).permutation(NTR)
        for s in range(0, NTR, BATCH):
            ii = idx[s:s + BATCH]; op.zero_grad()
            pr = m(tc(TR[0][ii]), tf_(TR[1][ii])); tg = tc(TR[2][ii])
            loss = (pr.real.float() - tg.real.float()).pow(2).mean() + (pr.imag.float() - tg.imag.float()).pow(2).mean()
            loss = loss + .1 * (pr.diagonal(dim1=-2, dim2=-1).sum(-1).real - 1).pow(2).mean()
            if head == 'cholesky_gated':
                loss = loss + lam * m.aux_logmass.pow(2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()
        sc.step()
    m.eval(); return m

def load(head):
    m = FNO(head=head, trunk='fno', dim=D, n_ch=NQ, modes=16, width=64, depth=4, T=NST + 1)
    m.load_state_dict(torch.load(f'{CK}/s{SEED}_{head}.pt')); m.eval(); return m

def accuracy(m):
    ftj = []
    with torch.no_grad():
        for i in range(len(TE[0])):
            pr = m(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy()
            ftj.append(float(np.mean([fid_phys(pr[t], TE[2][i][t]) for t in range(0, NST + 1, 5)])))
    return float(np.mean(ftj))

def logmass(m, u):
    with torch.no_grad():
        m(tc(G0[None]), tf_(u[None]))
        return float(m.aux_logmass.abs().mean()) if m.aux_logmass is not None else float('nan')

def score(m, p):
    u = expand_segments(np.clip(p, -CTRL_MAX, CTRL_MAX).reshape(NSEG, NQ), NST)
    with torch.no_grad(): pr = m(tc(G0[None]), tf_(u[None]))[0].numpy()
    gt = sysq.evolve(G0, u)
    return dict(claims=pure(pr[-1]), true_fid=pure(gt[-1]), gap=pure(pr[-1]) - pure(gt[-1]),
                err_at_uopt=float(np.mean([trace_dist(pr[t], gt[t]) for t in range(0, NST + 1, 10)])),
                ood_std_ratio=float(u.std() / CU[:NTR].std()), sat_frac=float(np.mean(np.abs(u) > 0.9 * CTRL_MAX)))

P0 = .05 * np.random.default_rng(7).standard_normal(NSEG * NQ)
def adam(m):
    b0 = tc(G0[None]); rep = NST // NSEG; p = P0.copy(); m1, v1 = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, OPT_ITERS + 1):
        pt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        c = pt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        ov = torch.real(torch.conj(tpsi) @ m(b0, c)[0, -1].to(torch.complex128) @ tpsi)
        g = torch.autograd.grad(-ov, pt)[0].numpy()
        if not np.isfinite(g).all(): break
        m1 = .9 * m1 + .1 * g; v1 = .999 * v1 + .001 * g * g
        p = np.clip(p - 3e-2 * (m1 / (1 - .9 ** t)) / (np.sqrt(v1 / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    return p

def cmaes(m):
    import cma
    rep = NST // NSEG
    def batch_claims(X):
        X = np.clip(np.asarray(X), -CTRL_MAX, CTRL_MAX)
        c = torch.tensor(X, dtype=torch.float32).reshape(len(X), NSEG, NQ).repeat_interleave(rep, 1)[:, :NST]
        with torch.no_grad():
            pr = m(tc(np.repeat(G0[None], len(X), 0)), c)[:, -1]
            return np.clip(np.real(np.einsum('i,bij,j->b', exc.conj(), pr.numpy(), exc)), 0, 1)
    es = cma.CMAEvolutionStrategy(P0.copy(), CMA_SIGMA, {'bounds': [-CTRL_MAX, CTRL_MAX], 'maxfevals': CMA_EVALS,
                                                         'seed': SEED + 1, 'verbose': -9})
    while not es.stop():
        X = es.ask(); es.tell(X, list(-batch_claims(X)))
    return np.asarray(es.result.xbest), int(es.result.evaluations)

R = {'seed': SEED, 'mode': MODE, 'lam': LAM}
T0 = time.time()
if MODE == 'lam':
    m = train('cholesky_gated', LAM)
    lm_ind = float(np.mean([logmass(m, CU[105 + j]) for j in range(5)]))
    r = score(m, adam(m)); r['fid_traj'] = accuracy(m)
    r['logmass_in_dist'] = lm_ind; r['ratio'] = logmass(m, expand_segments(adam(m).reshape(NSEG, NQ), NST)) / max(lm_ind, 1e-9)
    R['run'] = r
    log(f'F_tj={r["fid_traj"]:.3f} claims={r["claims"]:.3f} true={r["true_fid"]:.3f} gap={r["gap"]:+.3f} ratio={r["ratio"]:.1f}')
else:
    models = {'herm_trace': load('herm_trace'), 'cholesky': load('cholesky'), 'anchored': train('cholesky_gated', 0.1)}
    torch.save(models['anchored'].state_dict(), f'{CK}/s{SEED}_anchored.pt')
    R['runs'] = {}
    for h, m in models.items():
        t0 = time.time(); pb, ne = cmaes(m); r = score(m, pb); r['evals'] = ne; r['cma_sec'] = time.time() - t0
        r['adam'] = score(m, adam(m))
        R['runs'][h] = r
        log(f'{h:11s} CMA-ES: claims={r["claims"]:.3f} true={r["true_fid"]:.3f} gap={r["gap"]:+.3f} ({ne} evals) | '
            f'Adam: claims={r["adam"]["claims"]:.3f} true={r["adam"]["true_fid"]:.3f} gap={r["adam"]["gap"]:+.3f}')
R['total_min'] = (time.time() - T0) / 60
json.dump(R, open(f'{OUT}/{MODE}_s{SEED}' + (f'_lam{LAM:g}' if MODE == 'lam' else '') + '.json', 'w'))
log(f'DONE [{R["total_min"]:.1f}m]')
