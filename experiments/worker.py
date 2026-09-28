"""Final experiment: one seed -> all heads, baselines, ablations, mechanism probes.

usage: worker.py <seed> <ctrl_max> <core|full>
  core : heads + baselines + control + mechanism + mitigation
  full : core + modes sweep + training-set-size sweep
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
from models import FNO, HEADS

SEED = int(sys.argv[1]); CTRL_MAX = float(sys.argv[2]); MODE = sys.argv[3] if len(sys.argv) > 3 else 'core'
D, NQ, T_TOT, DT, NST, NSEG = 16, 4, 5.0, 0.05, 100, 20
NSAMP, EPOCHS, BATCH, NTR = 120, 120, 8, 90
OPT_ITERS, TRUE_ITERS, S_TRUE, WARM_ITERS, RS_DRAWS = 300, 60, 8, 30, 300
OUT = os.path.join(RESULTS, 'runs'); CK = CHECKPOINTS
os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)
log = lambda *a: print(f'[s{SEED}]', *a, flush=True)

sysq = QSys(n_qubits=NQ, total_time=T_TOT, dt=DT)
exc = np.zeros(D, dtype=complex); exc[-1] = 1.0
gnd = np.zeros(D, dtype=complex); gnd[0] = 1.0
G0 = np.outer(gnd, gnd.conj())
tc = lambda a: torch.tensor(np.asarray(a), dtype=torch.complex128)
tf_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
pure = lambda r: float(np.clip(np.real(exc.conj() @ r @ exc), 0, 1))

def fid_phys(p, g):
    """Uhlmann fidelity after projecting p onto the physical state space.
    Required because F is undefined for non-PSD predictions; negative mass is
    reported separately so the projection never hides a violation."""
    w, V = np.linalg.eigh((p + p.conj().T) / 2)
    w = np.clip(w, 0, None); s_ = w.sum()
    if s_ <= 0: return 0.0
    return fidelity((V * (w / s_)) @ V.conj().T, g)
tpsi = torch.tensor(exc, dtype=torch.complex128)
R = {'seed': SEED, 'ctrl_max': CTRL_MAX, 'mode': MODE, 'heads': {}, 'ablation': {}}
T0 = time.time()

# ------------------------------- data ---------------------------------------
t0 = time.time(); rng = np.random.default_rng(SEED)
R0, CU, RT = [], [], []
for i in range(NSAMP):
    r0 = random_rho(D, rng)
    sg = CTRL_MAX * np.tanh(rng.standard_normal((NSEG, NQ)))
    sg[1:-1] = (sg[:-2] + sg[1:-1] + sg[2:]) / 3
    c = expand_segments(sg, NST)
    R0.append(r0); CU.append(c); RT.append(sysq.evolve(r0, c))
R0, CU, RT = np.stack(R0), np.stack(CU), np.stack(RT)
sl = lambda a, b: (R0[a:b], CU[a:b], RT[a:b])
TR, VA, TE = sl(0, NTR), sl(NTR, 105), sl(105, NSAMP)
R['train_ctrl_std'] = float(CU[:NTR].std()); R['data_min'] = (time.time() - t0) / 60
U_IND = CU[110].copy()                     # a held-out in-distribution control
RHO_EQ = QSys(n_qubits=NQ, total_time=200.0, dt=DT).evolve(np.eye(D) / D, np.zeros((4000, NQ)))[-1]
log(f'data {R["data_min"]*60:.0f}s')

# ------------------- reference optimisers on TRUE dynamics ------------------
def true_fid_S(p, S=S_TRUE):
    return pure(sysq.evolve(G0, expand_segments(np.clip(p, -CTRL_MAX, CTRL_MAX).reshape(S, NQ), NST))[-1])

def grape(p0, iters, S=S_TRUE):
    p = p0.copy(); m_, v_ = np.zeros_like(p), np.zeros_like(p); cur = []
    for t in range(1, iters + 1):
        f0 = 1 - true_fid_S(p, S); g = np.zeros_like(p)
        for j in range(len(p)):
            dp = p.copy(); dp[j] += 2e-3; g[j] = ((1 - true_fid_S(dp, S)) - f0) / 2e-3
        m_ = .9 * m_ + .1 * g; v_ = .999 * v_ + .001 * g * g
        p = np.clip(p - 0.08 * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
        cur.append(true_fid_S(p, S))
    return p, cur

t0 = time.time(); rg = np.random.default_rng(7 + SEED)
p_cold, curve = grape(0.1 * rg.standard_normal(S_TRUE * NQ), TRUE_ITERS)
R['grape_ceiling'] = float(max(curve)); R['grape_curve'] = [float(x) for x in curve]
R['grape_at_30'] = float(max(curve[:WARM_ITERS])); R['grape_calls'] = TRUE_ITERS * (S_TRUE * NQ + 2)
R['grape_min'] = (time.time() - t0) / 60
# random-search baseline on true dynamics, matched in true-solver calls
rs = np.random.default_rng(31 + SEED); best = 0.0; rc = []
for _ in range(RS_DRAWS):
    sg = CTRL_MAX * np.tanh(rs.standard_normal((S_TRUE, NQ)))
    best = max(best, true_fid_S(sg.ravel())); rc.append(best)
R['random_search'] = dict(best=float(best), calls=RS_DRAWS, curve=[float(x) for x in rc])
log(f'GRAPE {R["grape_ceiling"]:.4f} ({R["grape_calls"]} calls) | random-search {best:.4f} ({RS_DRAWS} calls) [{R["grape_min"]:.1f}m]')

# ------------------------------ models --------------------------------------
class LinearSurrogate(torch.nn.Module):
    """Ridge-regression linear map [Re p0, Im p0, u_t, cumsum u_t, t] -> rho_t, same output head."""
    def __init__(self, dim=D):
        super().__init__(); self.dim = dim; self.W = None
    def feats(self, rho0, controls):
        B, T, _ = controls.shape
        x0 = torch.cat([rho0.real.reshape(B, -1), rho0.imag.reshape(B, -1)], -1).float()
        x0 = x0.unsqueeze(1).expand(B, T + 1, -1)
        cp = torch.cat([torch.zeros(B, 1, NQ), controls.float()], 1)
        cs = torch.cumsum(cp, 1) * DT
        tt = torch.linspace(0, 1, T + 1).view(1, T + 1, 1).expand(B, T + 1, 1)
        return torch.cat([x0, cp, cs, tt, torch.ones(B, T + 1, 1)], -1)
    def fit(self, data, lam=1e-3):
        Fx = self.feats(tc(data[0]), tf_(data[1])); X = Fx.reshape(-1, Fx.shape[-1])
        Y = torch.tensor(np.concatenate([data[2].real.reshape(len(data[2]), NST + 1, -1),
                                         data[2].imag.reshape(len(data[2]), NST + 1, -1)], -1),
                         dtype=torch.float32).reshape(X.shape[0], -1)
        A = X.T @ X + lam * torch.eye(X.shape[1]); self.W = torch.linalg.solve(A, X.T @ Y)
        return self
    def forward(self, rho0, controls):
        B, T, _ = controls.shape; d = self.dim
        f = self.feats(rho0, controls) @ self.W
        rho = torch.complex(f[..., :d * d].double(), f[..., d * d:].double()).reshape(B, T + 1, d, d)
        rho = (rho + rho.conj().transpose(-1, -2)) / 2
        tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).real.unsqueeze(-1).unsqueeze(-1)
        return rho / tr

def train(head, data, epochs=EPOCHS, modes=16, init=None):
    torch.manual_seed(SEED)
    mdl = FNO(head=head, dim=D, n_ch=NQ, modes=modes, width=64, depth=4)
    if init is not None: mdl.load_state_dict(init)
    op = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-5)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, epochs)
    X0, XC, XR = data; n = len(X0); htr, hva = [], []
    for ep in range(epochs):
        mdl.train(); idx = np.random.default_rng(SEED * 1000 + ep).permutation(n); tot = nb = 0
        for s in range(0, n, BATCH):
            ii = idx[s:s + BATCH]; op.zero_grad()
            pr = mdl(tc(X0[ii]), tf_(XC[ii])); tg = tc(XR[ii])
            mse = (pr.real.float() - tg.real.float()).pow(2).mean() + (pr.imag.float() - tg.imag.float()).pow(2).mean()
            trp = (pr.diagonal(dim1=-2, dim2=-1).sum(-1).real - 1).pow(2).mean()
            (mse + 0.1 * trp).backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); op.step()
            tot += float(mse + 0.1 * trp); nb += 1
        sc.step(); htr.append(tot / nb); mdl.eval()
        with torch.no_grad():
            pr = mdl(tc(VA[0]), tf_(VA[1])); tg = tc(VA[2])
            hva.append(float((pr.real.float() - tg.real.float()).pow(2).mean() + (pr.imag.float() - tg.imag.float()).pow(2).mean()))
    return mdl, htr, hva

# ------------------------------ evaluation ----------------------------------
def evaluate(mdl):
    ff, ftj, td, nm, mn, tee, bad, tot = [], [], [], [], [], [], 0, 0
    curve = np.zeros(NST + 1)
    with torch.no_grad():
        for i in range(len(TE[0])):
            pr = mdl(tc(TE[0][i:i + 1]), tf_(TE[1][i:i + 1]))[0].numpy(); gt = TE[2][i]
            ev = np.linalg.eigvalsh((pr + pr.conj().transpose(0, 2, 1)) / 2)
            nm.append(float(np.abs(np.clip(ev, None, 0)).sum(1).mean())); mn.append(float(ev.min()))
            bad += int((ev.min(1) < -1e-8).sum()); tot += ev.shape[0]
            tee.append(float(np.abs(np.trace(pr, axis1=1, axis2=2) - 1).mean()))
            f = np.array([fid_phys(pr[t], gt[t]) for t in range(NST + 1)]); curve += f
            ff.append(f[-1]); ftj.append(float(f.mean()))
            td.append(float(np.mean([trace_dist(pr[t], gt[t]) for t in range(0, NST + 1, 5)])))
    n = len(TE[0])
    return dict(fid_final=float(np.mean(ff)), fid_traj=float(np.mean(ftj)), trace_dist=float(np.mean(td)),
                neg_mass=float(np.mean(nm)), min_eig=float(np.min(mn)), frac_nonpsd=float(bad / tot),
                trace_err=float(np.mean(tee)), fid_curve=(curve / n).tolist())

def extrapolate(mdl):
    sl2 = QSys(n_qubits=NQ, total_time=10.0, dt=DT); rg = np.random.default_rng(1234 + SEED)
    o = {k: [] for k in ['fno_in', 'fno_ex', 'freeze', 'mixed', 'eq']}; cur = np.zeros(2 * NST + 1)
    for _ in range(5):
        r0 = random_rho(D, rg)
        cl = expand_segments(0.2 * CTRL_MAX * np.tanh(rg.standard_normal((NSEG, NQ))), 2 * NST)
        gt = sl2.evolve(r0, cl)
        with torch.no_grad(): pr = mdl(tc(r0[None]), tf_(cl[None]))[0].numpy()
        f = np.array([fid_phys(pr[i], gt[i]) for i in range(2 * NST + 1)]); cur += f
        o['fno_in'].append(float(f[:NST + 1].mean())); o['fno_ex'].append(float(f[NST + 1:].mean()))
        o['freeze'].append(float(np.mean([fid_phys(pr[NST], gt[i]) for i in range(NST + 1, 2 * NST + 1)])))
        o['mixed'].append(float(np.mean([fid_phys(np.eye(D) / D, gt[i]) for i in range(NST + 1, 2 * NST + 1)])))
        o['eq'].append(float(np.mean([fid_phys(RHO_EQ, gt[i]) for i in range(NST + 1, 2 * NST + 1)])))
    r = {k: float(np.mean(v)) for k, v in o.items()}; r['curve'] = (cur / 5).tolist(); return r

def pred_err(mdl, u):
    """Trajectory-mean trace distance between surrogate and truth for a GIVEN control."""
    gt = sysq.evolve(G0, u)
    with torch.no_grad(): pr = mdl(tc(G0[None]), tf_(u[None]))[0].numpy()
    return float(np.mean([trace_dist(pr[t], gt[t]) for t in range(0, NST + 1, 5)]))

def control_gap(mdl, iters=OPT_ITERS):
    b0 = tc(G0[None]); rep = NST // NSEG; rg = np.random.default_rng(7)
    p = 0.05 * rg.standard_normal(NSEG * NQ); m_, v_ = np.zeros_like(p), np.zeros_like(p); sc = []
    for t in range(1, iters + 1):
        pt = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        c = pt.reshape(NSEG, NQ).repeat_interleave(rep, 0)[:NST].unsqueeze(0)
        ov = torch.real(torch.conj(tpsi) @ mdl(b0, c)[0, -1].to(torch.complex128) @ tpsi)
        g = torch.autograd.grad(1 - ov, pt)[0].numpy()
        if not np.isfinite(g).all(): break
        m_ = .9 * m_ + .1 * g; v_ = .999 * v_ + .001 * g * g
        p = np.clip(p - 3e-2 * (m_ / (1 - .9 ** t)) / (np.sqrt(v_ / (1 - .999 ** t)) + 1e-8), -CTRL_MAX, CTRL_MAX)
        sc.append(float(np.clip(ov.item(), 0, 1)))
    u = expand_segments(p.reshape(NSEG, NQ), NST)
    with torch.no_grad(): sf = pure(mdl(b0, tf_(u[None]))[0, -1].numpy())
    tf2 = pure(sysq.evolve(G0, u)[-1])
    # --- mechanism: calibration path from in-distribution control to u* ---
    path = []
    for a in np.linspace(0, 1, 11):
        ua = (1 - a) * U_IND + a * u
        with torch.no_grad(): pa = mdl(b0, tf_(ua[None]))[0].numpy()
        gta = sysq.evolve(G0, ua)
        path.append(dict(alpha=float(a), f_sur=pure(pa[-1]), f_true=pure(gta[-1]),
                         err=float(np.mean([trace_dist(pa[t], gta[t]) for t in range(0, NST + 1, 10)])),
                         neg=float(np.abs(np.clip(np.linalg.eigvalsh((pa + pa.conj().transpose(0, 2, 1)) / 2), None, 0)).sum(1).mean())))
    return dict(surrogate_fid=float(sf), true_fid=float(tf2), gap=float(sf - tf2),
                curve=[float(x) for x in sc], u=u.tolist(), path=path,
                err_at_uopt=pred_err(mdl, u), err_in_dist=pred_err(mdl, U_IND),
                ood_std_ratio=float(u.std() / R['train_ctrl_std']),
                sat_frac=float(np.mean(np.abs(u) > 0.9 * CTRL_MAX)))

def full_probe(mdl, tag, tmin):
    ev = evaluate(mdl); ex = extrapolate(mdl); cg = control_gap(mdl)
    nrep = 5; t0 = time.time()
    for _ in range(nrep): sysq.evolve(TE[0][0], TE[1][0])
    sms = (time.time() - t0) / nrep * 1000
    with torch.no_grad():
        b0, bc = tc(TE[0][:1]), tf_(TE[1][:1]); mdl(b0, bc); t0 = time.time()
        for _ in range(nrep): mdl(b0, bc)
        fms = (time.time() - t0) / nrep * 1000
    d = dict(**ev, extrap=ex, control=cg, train_min=tmin, solver_ms=sms, fno_ms=fms, speedup=sms / fms)
    log(f'{tag:16s} F_tj={ev["fid_traj"]:.3f} neg={ev["neg_mass"]:.1e} claims={cg["surrogate_fid"]:.3f} '
        f'delivers={cg["true_fid"]:.3f} err@u*={cg["err_at_uopt"]:.3f} (ind {cg["err_in_dist"]:.3f})')
    return d

# ------------------------------ main heads ----------------------------------
for head in HEADS:
    t0 = time.time(); mdl, htr, hva = train(head, TR); tm = (time.time() - t0) / 60
    mdl.eval(); torch.save(mdl.state_dict(), f'{CK}/s{SEED}_{head}.pt')
    R['heads'][head] = full_probe(mdl, head, tm); R['heads'][head].update(hist_tr=htr, hist_va=hva)

t0 = time.time(); lin = LinearSurrogate().fit(TR)
R['heads']['linear'] = full_probe(lin, 'linear(ridge)', (time.time() - t0) / 60)
R['heads']['linear'].update(hist_tr=[], hist_va=[])

# ---------------- mitigation: matched budget, warm start --------------------
R['mitigation'] = {}
for head in ['herm_trace', 'cholesky']:
    u = np.array(R['heads'][head]['control']['u']); rg = np.random.default_rng(99 + SEED)
    aR0, aCU, aRT = [], [], []
    for i in range(20):
        r0 = G0 if i < 4 else random_rho(D, rg)
        c = u if i == 0 else np.clip(u + 0.15 * CTRL_MAX * rg.standard_normal(u.shape), -CTRL_MAX, CTRL_MAX)
        aR0.append(r0); aCU.append(c); aRT.append(sysq.evolve(r0, c))
    aug = (np.concatenate([TR[0], np.stack(aR0)]), np.concatenate([TR[1], np.stack(aCU)]),
           np.concatenate([TR[2], np.stack(aRT)]))
    ck = torch.load(f'{CK}/s{SEED}_{head}.pt')
    m_aug, _, _ = train(head, aug, epochs=EPOCHS, init=ck); m_aug.eval()       # data +20, matched epochs, warm start
    m_ctl, _, _ = train(head, TR, epochs=EPOCHS, init=ck); m_ctl.eval()        # CONTROL: same extra epochs, no new data
    ca, cc = control_gap(m_aug), control_gap(m_ctl)
    R['mitigation'][head] = dict(before=R['heads'][head]['control']['gap'], after=ca['gap'], ctrl_after=cc['gap'],
                                 true_before=R['heads'][head]['control']['true_fid'],
                                 true_after=ca['true_fid'], ctrl_true_after=cc['true_fid'],
                                 sur_after=ca['surrogate_fid'], err_after=ca['err_at_uopt'])
    log(f'MITIG {head:12s} gap {R["mitigation"][head]["before"]:+.3f} -> {ca["gap"]:+.3f} '
        f'(epoch-matched control {cc["gap"]:+.3f}) | true {R["mitigation"][head]["true_before"]:.3f} -> {ca["true_fid"]:.3f}')

# ------------- surrogate as GRAPE initialiser (practical utility) ------------
R['warmstart'] = {}
for head in ['herm_trace', 'cholesky']:
    u = np.array(R['heads'][head]['control']['u'])
    _ix = np.linspace(0, NST, S_TRUE + 1).astype(int)
    p0 = np.stack([u[_ix[i]:_ix[i + 1]].mean(0) for i in range(S_TRUE)]).ravel()   # u* -> GRAPE segments
    _, cw = grape(p0, WARM_ITERS)
    R['warmstart'][head] = dict(warm_best=float(max(cw)), warm_curve=[float(x) for x in cw],
                                cold_at_same_budget=R['grape_at_30'], init_fid=float(true_fid_S(p0)))
    log(f'WARMSTART {head:12s} init={R["warmstart"][head]["init_fid"]:.3f} '
        f'warm@{WARM_ITERS}={max(cw):.3f} vs cold@{WARM_ITERS}={R["grape_at_30"]:.3f}')

# ------------------------------ ablations -----------------------------------
if MODE == 'full':
    for m in [0, 4, 8, 32]:
        t0 = time.time(); mdl, _, _ = train('herm_trace', TR, modes=m); mdl.eval()
        ev = evaluate(mdl); cg = control_gap(mdl)
        R['ablation'][f'modes_{m}'] = dict(modes=m, fid_traj=ev['fid_traj'], fid_final=ev['fid_final'],
                                           neg_mass=ev['neg_mass'], gap=cg['gap'], true_fid=cg['true_fid'],
                                           surrogate_fid=cg['surrogate_fid'], train_min=(time.time() - t0) / 60)
        log(f'ABL modes={m:2d} F_tj={ev["fid_traj"]:.3f} gap={cg["gap"]:+.3f}')
    R['ablation']['modes_16'] = dict(modes=16, fid_traj=R['heads']['herm_trace']['fid_traj'],
                                     fid_final=R['heads']['herm_trace']['fid_final'],
                                     neg_mass=R['heads']['herm_trace']['neg_mass'],
                                     gap=R['heads']['herm_trace']['control']['gap'],
                                     true_fid=R['heads']['herm_trace']['control']['true_fid'],
                                     surrogate_fid=R['heads']['herm_trace']['control']['surrogate_fid'],
                                     train_min=R['heads']['herm_trace']['train_min'])
    for head in ['herm_trace', 'cholesky']:
        for n in [30, 60]:
            sub = (TR[0][:n], TR[1][:n], TR[2][:n])
            mdl, _, _ = train(head, sub); mdl.eval()
            ev = evaluate(mdl); cg = control_gap(mdl)
            R['ablation'][f'ntrain_{head}_{n}'] = dict(head=head, n=n, fid_traj=ev['fid_traj'],
                                                       gap=cg['gap'], true_fid=cg['true_fid'],
                                                       surrogate_fid=cg['surrogate_fid'])
            log(f'ABL {head} n={n} F_tj={ev["fid_traj"]:.3f} gap={cg["gap"]:+.3f}')
        R['ablation'][f'ntrain_{head}_90'] = dict(head=head, n=90, fid_traj=R['heads'][head]['fid_traj'],
                                                  gap=R['heads'][head]['control']['gap'],
                                                  true_fid=R['heads'][head]['control']['true_fid'],
                                                  surrogate_fid=R['heads'][head]['control']['surrogate_fid'])

R['total_min'] = (time.time() - T0) / 60
R['env'] = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0])
json.dump(R, open(f'{OUT}/seed_{SEED}.json', 'w'))
log(f'DONE [{R["total_min"]:.1f} min]')
