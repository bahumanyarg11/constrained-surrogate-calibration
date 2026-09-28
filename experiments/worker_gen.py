"""Generalisation study: does 'confidently wrong' survive a change of system?
usage: worker_gen.py <seed> <n_qubits> <t1> <tag>
Decisive subset only: herm_trace vs cholesky, accuracy + physicality + claims/delivers + calibration.
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

SEED, NQ, T1, TAG = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
D = 2 ** NQ
CTRL_MAX, T_TOT, DT, NST, NSEG = 1.0, 5.0, 0.05, 100, 20
NSAMP = int(os.environ.get('GEN_NSAMP', 60 if NQ >= 6 else 120))
NTR, NVA = int(NSAMP * .75), int(NSAMP * .125)
EPOCHS = int(os.environ.get('GEN_EPOCHS', 60 if NQ >= 6 else 120))
BATCH, OPT_ITERS = 8, 300
S_TRUE = 6 if NQ >= 6 else 8
TRUE_ITERS = 20 if NQ >= 6 else 60
HEADS = ['herm_trace', 'cholesky']
OUT = os.path.join(RESULTS, 'runs_gen'); os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f'[{TAG}/s{SEED}]', *a, flush=True)

sysq = QSys(n_qubits=NQ, t1=T1, t2=min(10.0, T1 / 1.5), total_time=T_TOT, dt=DT)
exc = np.zeros(D, dtype=complex); exc[-1] = 1.0
gnd = np.zeros(D, dtype=complex); gnd[0] = 1.0
G0 = np.outer(gnd, gnd.conj())
tc = lambda a: torch.tensor(np.asarray(a), dtype=torch.complex128)
tf_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
pure = lambda r: float(np.clip(np.real(exc.conj() @ r @ exc), 0, 1))
tpsi = torch.tensor(exc, dtype=torch.complex128)
def fid_phys(p, g):
    w, V = np.linalg.eigh((p + p.conj().T) / 2); w = np.clip(w, 0, None); s_ = w.sum()
    return 0.0 if s_ <= 0 else fidelity((V * (w / s_)) @ V.conj().T, g)
R = {'seed': SEED, 'n_qubits': NQ, 't1': T1, 'tag': TAG, 'dim': D, 'heads': {}}
T0 = time.time()

rng = np.random.default_rng(SEED); R0, CU, RT = [], [], []
for i in range(NSAMP):
    r0 = random_rho(D, rng)
    sg = CTRL_MAX * np.tanh(rng.standard_normal((NSEG, NQ)))
    sg[1:-1] = (sg[:-2] + sg[1:-1] + sg[2:]) / 3
    c = expand_segments(sg, NST)
    R0.append(r0); CU.append(c); RT.append(sysq.evolve(r0, c))
R0, CU, RT = np.stack(R0), np.stack(CU), np.stack(RT)
sl = lambda a, b: (R0[a:b], CU[a:b], RT[a:b])
TR, VA, TE = sl(0, NTR), sl(NTR, NTR + NVA), sl(NTR + NVA, NSAMP)
U_IND = CU[-1].copy(); R['train_ctrl_std'] = float(CU[:NTR].std())
log(f'data {(time.time()-T0):.0f}s dim={D} n={NSAMP}')

def true_fid_S(p):
    return pure(sysq.evolve(G0, expand_segments(np.clip(p, -CTRL_MAX, CTRL_MAX).reshape(S_TRUE, NQ), NST))[-1])
t0 = time.time(); rg = np.random.default_rng(7 + SEED)
p = 0.1 * rg.standard_normal(S_TRUE * NQ); m_, v_ = np.zeros_like(p), np.zeros_like(p); cur = []
for t in range(1, TRUE_ITERS + 1):
    f0 = 1 - true_fid_S(p); gq = np.zeros_like(p)
    for j in range(len(p)):
        dp = p.copy(); dp[j] += 2e-3; gq[j] = ((1 - true_fid_S(dp)) - f0) / 2e-3
    m_ = .9 * m_ + .1 * gq; v_ = .999 * v_ + .001 * gq * gq
    p = np.clip(p - .08 * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    cur.append(true_fid_S(p))
R['grape_ceiling'] = float(max(cur)); R['grape_min'] = (time.time() - t0) / 60
log(f'GRAPE {R["grape_ceiling"]:.4f} [{R["grape_min"]:.1f}m]')

for head in HEADS:
    t0 = time.time(); torch.manual_seed(SEED)
    mdl = FNO(head=head, dim=D, n_ch=NQ, modes=16, width=64, depth=4)
    op = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-5)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    for ep in range(EPOCHS):
        mdl.train(); idx = np.random.default_rng(SEED * 1000 + ep).permutation(NTR)
        for s in range(0, NTR, BATCH):
            ii = idx[s:s + BATCH]; op.zero_grad()
            pr = mdl(tc(TR[0][ii]), tf_(TR[1][ii])); tg = tc(TR[2][ii])
            mse = (pr.real.float() - tg.real.float()).pow(2).mean() + (pr.imag.float() - tg.imag.float()).pow(2).mean()
            trp = (pr.diagonal(dim1=-2, dim2=-1).sum(-1).real - 1).pow(2).mean()
            (mse + .1 * trp).backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); op.step()
        sc.step()
    tm = (time.time() - t0) / 60; mdl.eval()
    ff, ftj, nm = [], [], []
    with torch.no_grad():
        for i in range(len(TE[0])):
            pr = mdl(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy(); gt = TE[2][i]
            ev = np.linalg.eigvalsh((pr + pr.conj().transpose(0, 2, 1)) / 2)
            nm.append(float(np.abs(np.clip(ev, None, 0)).sum(1).mean()))
            f = [fid_phys(pr[t], gt[t]) for t in range(0, NST + 1, 5)]
            ff.append(f[-1]); ftj.append(float(np.mean(f)))
    # surrogate-guided control
    b0 = tc(G0[None]); rep = NST // NSEG; rg = np.random.default_rng(7)
    q = .05 * rg.standard_normal(NSEG * NQ); m_, v_ = np.zeros_like(q), np.zeros_like(q)
    for t in range(1, OPT_ITERS + 1):
        qt = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        c = qt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        ov = torch.real(torch.conj(tpsi) @ mdl(b0, c)[0, -1].to(torch.complex128) @ tpsi)
        gq = torch.autograd.grad(1 - ov, qt)[0].numpy()
        if not np.isfinite(gq).all(): break
        m_ = .9 * m_ + .1 * gq; v_ = .999 * v_ + .001 * gq * gq
        q = np.clip(q - 3e-2 * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    u = expand_segments(q.reshape(NSEG, NQ), NST)
    with torch.no_grad(): sf = pure(mdl(b0, tf_(u[None]))[0, -1].numpy())
    gt_u = sysq.evolve(G0, u); tfd = pure(gt_u[-1])
    with torch.no_grad(): pr_u = mdl(b0, tf_(u[None]))[0].numpy()
    err_u = float(np.mean([trace_dist(pr_u[t], gt_u[t]) for t in range(0, NST + 1, 10)]))
    path = []
    for a in np.linspace(0, 1, 6):
        ua = (1 - a) * U_IND + a * u
        with torch.no_grad(): pa = mdl(b0, tf_(ua[None]))[0].numpy()
        ga = sysq.evolve(G0, ua)
        path.append(dict(alpha=float(a), f_sur=pure(pa[-1]), f_true=pure(ga[-1])))
    R['heads'][head] = dict(fid_final=float(np.mean(ff)), fid_traj=float(np.mean(ftj)),
                            neg_mass=float(np.mean(nm)), surrogate_fid=float(sf), true_fid=float(tfd),
                            gap=float(sf - tfd), err_at_uopt=err_u, train_min=tm, path=path,
                            calib_err=float(np.mean([abs(z['f_sur'] - z['f_true']) for z in path])))
    log(f'{head:12s} F_tj={np.mean(ftj):.3f} neg={np.mean(nm):.1e} claims={sf:.3f} delivers={tfd:.3f} err@u*={err_u:.3f}')

R['total_min'] = (time.time() - T0) / 60
json.dump(R, open(f'{OUT}/{TAG}_s{SEED}.json', 'w'))
log(f'DONE [{R["total_min"]:.1f}m]')
