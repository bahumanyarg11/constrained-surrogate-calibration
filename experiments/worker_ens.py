"""Uncertainty-aware surrogates: can ensemble disagreement restore the 'I don't know' signal
that a structural physicality constraint destroys?

usage: worker_ens.py <seed>
Per head in {herm_trace, cholesky}: train K=5 members, then optimise control against
  (i) a single member, (ii) the ensemble mean, (iii) a pessimistic LCB objective mean - kappa*std.
Records disagreement in-distribution vs at u*, and claims/delivers for each objective.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
np.seterr(all='ignore')
import torch
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from paths import RESULTS
from physics import QSys, fidelity, trace_dist, random_rho, expand_segments
from models import FNO

SEED = int(sys.argv[1]); K = 5; KAPPA = 1.0
D, NQ, T_TOT, DT, NST, NSEG = 16, 4, 5.0, 0.05, 100, 20
NSAMP, NTR, NVA, EPOCHS, BATCH = 120, 90, 15, 120, 8
CTRL_MAX, OPT_ITERS = 1.0, 300
OUT = os.path.join(RESULTS, 'runs_ens'); os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f'[ens/s{SEED}]', *a, flush=True)

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
R = {'seed': SEED, 'K': K, 'kappa': KAPPA, 'heads': {}}
T0 = time.time()

rng = np.random.default_rng(SEED); R0, CU, RT = [], [], []
for i in range(NSAMP):
    r0 = random_rho(D, rng)
    sg = CTRL_MAX * np.tanh(rng.standard_normal((NSEG, NQ)))
    sg[1:-1] = (sg[:-2] + sg[1:-1] + sg[2:]) / 3
    c = expand_segments(sg, NST)
    R0.append(r0); CU.append(c); RT.append(sysq.evolve(r0, c))
R0, CU, RT = np.stack(R0), np.stack(CU), np.stack(RT)
TR = (R0[:NTR], CU[:NTR], RT[:NTR]); TE = (R0[NTR + NVA:], CU[NTR + NVA:], RT[NTR + NVA:])
U_IND = CU[110].copy()
log(f'data {(time.time()-T0):.0f}s')

def train(head, member):
    torch.manual_seed(SEED * 100 + member * 7 + 1)
    m = FNO(head=head, dim=D, n_ch=NQ, modes=16, width=64, depth=4)
    op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    for ep in range(EPOCHS):
        m.train(); idx = np.random.default_rng(SEED * 1000 + member * 31 + ep).permutation(NTR)
        for s in range(0, NTR, BATCH):
            ii = idx[s:s + BATCH]; op.zero_grad()
            pr = m(tc(TR[0][ii]), tf_(TR[1][ii])); tg = tc(TR[2][ii])
            mse = (pr.real.float() - tg.real.float()).pow(2).mean() + (pr.imag.float() - tg.imag.float()).pow(2).mean()
            trp = (pr.diagonal(dim1=-2, dim2=-1).sum(-1).real - 1).pow(2).mean()
            (mse + .1 * trp).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()
        sc.step()
    m.eval(); return m

def ens_stats(models, u):
    """fidelity mean/std across members for control u (no grad)."""
    b0 = tc(G0[None]); v = []
    with torch.no_grad():
        for m in models: v.append(pure(m(b0, tf_(u[None]))[0, -1].numpy()))
    return float(np.mean(v)), float(np.std(v)), v

def optimise(models, mode):
    """mode: 'single' | 'mean' | 'lcb'."""
    use = models[:1] if mode == 'single' else models
    b0 = tc(G0[None]); rep = NST // NSEG; rg = np.random.default_rng(7)
    p = .05 * rg.standard_normal(NSEG * NQ); m_, v_ = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, OPT_ITERS + 1):
        pt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        c = pt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        fs = torch.stack([torch.real(torch.conj(tpsi) @ m(b0, c)[0, -1].to(torch.complex128) @ tpsi) for m in use])
        obj = fs.mean() - (KAPPA * fs.std() if (mode == 'lcb' and len(use) > 1) else 0.0)
        gq = torch.autograd.grad(-obj, pt)[0].numpy()
        if not np.isfinite(gq).all(): break
        m_ = .9 * m_ + .1 * gq; v_ = .999 * v_ + .001 * gq * gq
        p = np.clip(p - 3e-2 * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    u = expand_segments(p.reshape(NSEG, NQ), NST)
    cm, cs, _ = ens_stats(models, u)
    tfd = pure(sysq.evolve(G0, u)[-1])
    return dict(mode=mode, claims=cm, disagree=cs, true_fid=tfd, gap=cm - tfd, u=u.tolist())

for head in ['herm_trace', 'cholesky']:
    t0 = time.time(); models = [train(head, k) for k in range(K)]
    tm = (time.time() - t0) / 60
    # ensemble accuracy on test set
    ftj = []
    with torch.no_grad():
        for i in range(len(TE[0])):
            prs = np.mean([m(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy() for m in models], 0)
            ftj.append(float(np.mean([fid_phys(prs[t], TE[2][i][t]) for t in range(0, NST + 1, 5)])))
    # disagreement in-distribution (held-out controls) vs at each optimum
    ind_dis = float(np.mean([ens_stats(models, CU[105 + j])[1] for j in range(5)]))
    res = {mo: optimise(models, mo) for mo in ['single', 'mean', 'lcb']}
    for mo in res:
        res[mo]['disagree_ratio'] = res[mo]['disagree'] / max(ind_dis, 1e-9)
    R['heads'][head] = dict(fid_traj_ens=float(np.mean(ftj)), train_min=tm, ind_disagree=ind_dis, **{f'opt_{k}': v for k, v in res.items()})
    log(f'{head:12s} F_ens={np.mean(ftj):.3f} ind_dis={ind_dis:.4f} | ' +
        ' | '.join(f'{mo}: claims={res[mo]["claims"]:.3f} true={res[mo]["true_fid"]:.3f} '
                   f'gap={res[mo]["gap"]:+.3f} dis_ratio={res[mo]["disagree_ratio"]:.1f}' for mo in ['single', 'mean', 'lcb']))

R['total_min'] = (time.time() - T0) / 60
json.dump(R, open(f'{OUT}/ens_s{SEED}.json', 'w'))
log(f'DONE [{R["total_min"]:.1f}m]')
