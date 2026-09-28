"""(1) Is the calibration failure FNO-specific?  trunk in {fno, transformer, mixer} x head in {herm_trace, cholesky}
(2) A cure: trace-gated Cholesky. Keep rho = LL^dag/Tr(LL^dag) exactly physical, but train the *unnormalised*
    mass Tr(LL^dag) to sit at 1 in-distribution and use |log mass| as a single-model OOD gate.
usage: worker_arch.py <seed>
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

SEED = int(sys.argv[1])
D, NQ, T_TOT, DT, NST, NSEG = 16, 4, 5.0, 0.05, 100, 20
NSAMP, NTR, NVA, EPOCHS, BATCH = 120, 90, 15, 120, 8
CTRL_MAX, OPT_ITERS, KAPPA, LAM_MASS = 1.0, 300, 1.0, 0.1
OUT = os.path.join(RESULTS, 'runs_arch'); os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f'[arch/s{SEED}]', *a, flush=True)

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
R = {'seed': SEED, 'kappa': KAPPA, 'lam_mass': LAM_MASS, 'arch': {}, 'cure': {}}
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
            if head == 'cholesky_gated':                 # pin the mass to 1 in-distribution
                loss = loss + LAM_MASS * m.aux_logmass.pow(2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()
        sc.step()
    m.eval(); return m

def accuracy(m):
    ftj, nm = [], []
    with torch.no_grad():
        for i in range(len(TE[0])):
            pr = m(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy()
            ev = np.linalg.eigvalsh((pr + pr.conj().transpose(0, 2, 1)) / 2)
            nm.append(float(np.abs(np.clip(ev, None, 0)).sum(1).mean()))
            ftj.append(float(np.mean([fid_phys(pr[t], TE[2][i][t]) for t in range(0, NST + 1, 5)])))
    return float(np.mean(ftj)), float(np.mean(nm))

def logmass_at(m, u):
    with torch.no_grad():
        m(tc(G0[None]), tf_(u[None]))
        return float(m.aux_logmass.abs().mean()) if m.aux_logmass is not None else float('nan')

def optimise(m, gated=False):
    b0 = tc(G0[None]); rep = NST // NSEG; rg = np.random.default_rng(7)
    p = .05 * rg.standard_normal(NSEG * NQ); m1, v1 = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, OPT_ITERS + 1):
        pt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        c = pt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        rho = m(b0, c)[0, -1]
        ov = torch.real(torch.conj(tpsi) @ rho.to(torch.complex128) @ tpsi)
        obj = ov - (KAPPA * m.aux_logmass.abs().mean() if gated else 0.0)
        g = torch.autograd.grad(-obj, pt)[0].numpy()
        if not np.isfinite(g).all(): break
        m1 = .9 * m1 + .1 * g; v1 = .999 * v1 + .001 * g * g
        p = np.clip(p - 3e-2 * (m1 / (1 - .9 ** t)) / (np.sqrt(v1 / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
    u = expand_segments(p.reshape(NSEG, NQ), NST)
    with torch.no_grad(): sf = pure(m(b0, tf_(u[None]))[0, -1].numpy())
    gt = sysq.evolve(G0, u); tfd = pure(gt[-1])
    with torch.no_grad(): pr = m(b0, tf_(u[None]))[0].numpy()
    return dict(claims=float(sf), true_fid=float(tfd), gap=float(sf - tfd),
                err_at_uopt=float(np.mean([trace_dist(pr[t], gt[t]) for t in range(0, NST + 1, 10)])),
                logmass_uopt=logmass_at(m, u), u=u.tolist())

# ---------------- (1) architecture sweep ----------------
for trunk in ['fno', 'transformer', 'mixer']:
    for head in ['herm_trace', 'cholesky']:
        t0 = time.time(); m = train(head, trunk)
        ft, nm = accuracy(m); o = optimise(m)
        R['arch'][f'{trunk}_{head}'] = dict(trunk=trunk, head=head, fid_traj=ft, neg_mass=nm,
                                            train_min=(time.time() - t0) / 60, **{k: v for k, v in o.items() if k != 'u'})
        log(f'{trunk:11s} {head:11s} F_tj={ft:.3f} neg={nm:.1e} claims={o["claims"]:.3f} '
            f'delivers={o["true_fid"]:.3f} gap={o["gap"]:+.3f}')

# ---------------- (2) the cure: trace-gated Cholesky ----------------
t0 = time.time(); mg = train('cholesky_gated', 'fno')
ft, nm = accuracy(mg)
lm_ind = float(np.mean([logmass_at(mg, CU[105 + j]) for j in range(5)]))
o_plain = optimise(mg, gated=False)
o_gated = optimise(mg, gated=True)
R['cure'] = dict(fid_traj=ft, neg_mass=nm, logmass_in_dist=lm_ind, train_min=(time.time() - t0) / 60,
                 plain={k: v for k, v in o_plain.items() if k != 'u'},
                 gated={k: v for k, v in o_gated.items() if k != 'u'},
                 ratio_plain=o_plain['logmass_uopt'] / max(lm_ind, 1e-9),
                 ratio_gated=o_gated['logmass_uopt'] / max(lm_ind, 1e-9))
log(f'CURE F_tj={ft:.3f} |logmass| in-dist={lm_ind:.4f} | plain: claims={o_plain["claims"]:.3f} '
    f'true={o_plain["true_fid"]:.3f} gap={o_plain["gap"]:+.3f} ratio={R["cure"]["ratio_plain"]:.1f} | '
    f'gated: claims={o_gated["claims"]:.3f} true={o_gated["true_fid"]:.3f} gap={o_gated["gap"]:+.3f} '
    f'ratio={R["cure"]["ratio_gated"]:.1f}')

R['total_min'] = (time.time() - T0) / 60
json.dump(R, open(f'{OUT}/arch_s{SEED}.json', 'w'))
log(f'DONE [{R["total_min"]:.1f}m]')
