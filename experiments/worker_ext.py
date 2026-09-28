"""Extension experiments requested in review-style self-audit.

usage: worker_ext.py <seed> <mode>
  sys2       : second Hamiltonian family (exchange-coupled XY ring, non-commuting with the drift);
               FNO trunk; heads herm_trace, cholesky, and the anchored remedy; GRAPE + random search.
  curetrunks : main system; the anchored remedy on the MLP-Mixer and Transformer trunks, paired with
               the plain Cholesky runs in runs_arch/ (same seed, same data, same initialisation).
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
from physics import QSys, fidelity, trace_dist, random_rho, expand_segments, kron_n, I2, X, Y, Z
from models import FNO

SEED, MODE = int(sys.argv[1]), sys.argv[2]
D, NQ, T_TOT, DT, NST, NSEG = 16, 4, 5.0, 0.05, 100, 20
NSAMP, NTR, NVA, EPOCHS, BATCH = 120, 90, 15, 120, 8
CTRL_MAX, OPT_ITERS, LAM_MASS = 1.0, 300, 0.1
S_TRUE, TRUE_ITERS, RS_DRAWS = 8, 60, 300
J_XY = 0.1
OUT = os.path.join(RESULTS, 'runs_ext'); os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f'[{MODE}/s{SEED}]', *a, flush=True)

sysq = QSys(n_qubits=NQ, total_time=T_TOT, dt=DT)
if MODE == 'sys2':
    # same detunings, but exchange (XX+YY) coupling on a ring: non-commuting with the drift, different topology
    Hd = np.zeros((D, D), dtype=np.complex128)
    for i in range(NQ):
        ops = [I2] * NQ; ops[i] = Z; Hd += 0.5 * (1.0 + 0.1 * i) * kron_n(ops)
    for i in range(NQ):
        j = (i + 1) % NQ
        for P in (X, Y):
            ops = [I2] * NQ; ops[i] = P; ops[j] = P; Hd += J_XY * kron_n(ops)
    sysq.Hd = Hd

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
R = {'seed': SEED, 'mode': MODE, 'lam_mass': LAM_MASS, 'runs': {}}
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
log(f'data {(time.time()-T0):.0f}s')

def train(head, trunk):
    torch.manual_seed(SEED)
    m = FNO(head=head, trunk=trunk, dim=D, n_ch=NQ, modes=16, width=64, depth=4, T=NST + 1)
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
                loss = loss + LAM_MASS * m.aux_logmass.pow(2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()
        sc.step()
    m.eval(); return m

def evaluate(m):
    ftj, nm = [], []
    with torch.no_grad():
        for i in range(len(TE[0])):
            pr = m(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy()
            ev = np.linalg.eigvalsh((pr + pr.conj().transpose(0, 2, 1)) / 2)
            nm.append(float(np.abs(np.clip(ev, None, 0)).sum(1).mean()))
            ftj.append(float(np.mean([fid_phys(pr[t], TE[2][i][t]) for t in range(0, NST + 1, 5)])))
    b0 = tc(G0[None]); rep = NST // NSEG; rg = np.random.default_rng(7)
    p = .05 * rg.standard_normal(NSEG * NQ); m1, v1 = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, OPT_ITERS + 1):
        pt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        c = pt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        ov = torch.real(torch.conj(tpsi) @ m(b0, c)[0, -1].to(torch.complex128) @ tpsi)
        g = torch.autograd.grad(-ov, pt)[0].numpy()
        if not np.isfinite(g).all(): break
        m1 = .9 * m1 + .1 * g; v1 = .999 * v1 + .001 * g * g
        p = np.clip(p - 3e-2 * (m1 / (1 - .9 ** t)) / (np.sqrt(v1 / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    u = expand_segments(p.reshape(NSEG, NQ), NST)
    with torch.no_grad(): pr = m(b0, tf_(u[None]))[0].numpy()
    gt = sysq.evolve(G0, u)
    return dict(fid_traj=float(np.mean(ftj)), neg_mass=float(np.mean(nm)), claims=pure(pr[-1]),
                true_fid=pure(gt[-1]), gap=pure(pr[-1]) - pure(gt[-1]),
                err_at_uopt=float(np.mean([trace_dist(pr[t], gt[t]) for t in range(0, NST + 1, 10)])))

if MODE == 'sys2':
    def tf_S(p):
        return pure(sysq.evolve(G0, expand_segments(np.clip(p, -CTRL_MAX, CTRL_MAX).reshape(S_TRUE, NQ), NST))[-1])
    rg = np.random.default_rng(7 + SEED); p = .1 * rg.standard_normal(S_TRUE * NQ)
    m1, v1 = np.zeros_like(p), np.zeros_like(p); cur = []
    for t in range(1, TRUE_ITERS + 1):
        f0 = 1 - tf_S(p); g = np.zeros_like(p)
        for j in range(len(p)):
            dp = p.copy(); dp[j] += 2e-3; g[j] = ((1 - tf_S(dp)) - f0) / 2e-3
        m1 = .9 * m1 + .1 * g; v1 = .999 * v1 + .001 * g * g
        p = np.clip(p - .08 * (m1 / (1 - .9 ** t)) / (np.sqrt(v1 / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
        cur.append(tf_S(p))
    rs = np.random.default_rng(31 + SEED); best = 0.0
    for _ in range(RS_DRAWS):
        best = max(best, tf_S((CTRL_MAX * np.tanh(rs.standard_normal((S_TRUE, NQ)))).ravel()))
    R['grape_ceiling'], R['random_search'] = float(max(cur)), float(best)
    log(f'GRAPE {R["grape_ceiling"]:.3f} random-search {best:.3f}')
    plan = [('fno', 'herm_trace'), ('fno', 'cholesky'), ('fno', 'cholesky_gated')]
else:
    plan = [('mixer', 'cholesky_gated'), ('transformer', 'cholesky_gated')]

for trunk, head in plan:
    t0 = time.time(); m = train(head, trunk); r = evaluate(m); r['train_min'] = (time.time() - t0) / 60
    R['runs'][f'{trunk}_{head}'] = r
    log(f'{trunk:11s} {head:15s} F_tj={r["fid_traj"]:.3f} neg={r["neg_mass"]:.1e} claims={r["claims"]:.3f} '
        f'delivers={r["true_fid"]:.3f} gap={r["gap"]:+.3f}')
R['total_min'] = (time.time() - T0) / 60
json.dump(R, open(f'{OUT}/{MODE}_s{SEED}.json', 'w'))
log(f'DONE [{R["total_min"]:.1f}m]')
