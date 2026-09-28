"""4-qubit open system Lindblad solver (numpy, RK4)."""
import numpy as np

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SM = np.array([[0, 1], [0, 0]], dtype=np.complex128)
SP = np.array([[0, 0], [1, 0]], dtype=np.complex128)


def kron_n(ops):
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r


class QSys:
    def __init__(self, n_qubits=4, drift=0.5, coupling=0.02, t1=15.0, t2=10.0,
                 thermal=0.01, total_time=5.0, dt=0.05):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.total_time = total_time
        self.dt = dt
        self.n_steps = int(round(total_time / dt))
        # drift
        Hd = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.n):
            ops = [I2] * self.n
            ops[i] = Z
            Hd += drift * (1.0 + 0.1 * i) * kron_n(ops)
        for i in range(self.n - 1):
            ops = [I2] * self.n
            ops[i] = Z
            ops[i + 1] = Z
            Hd += coupling * kron_n(ops)
        self.Hd = Hd
        # controls: local X
        Hc = []
        for k in range(self.n):
            ops = [I2] * self.n
            ops[k] = X
            Hc.append(kron_n(ops))
        self.Hc = np.stack(Hc)  # (4,16,16)
        self.n_ch = self.n
        # collapse ops
        g1 = 1.0 / t1
        gp = max(1.0 / t2 - 0.5 / t1, 0.0)
        Ls = []
        for i in range(self.n):
            ops = [I2] * self.n
            ops[i] = SM
            Ls.append(np.sqrt(g1 * (1 + thermal)) * kron_n(ops))
            ops = [I2] * self.n
            ops[i] = SP
            Ls.append(np.sqrt(g1 * thermal) * kron_n(ops))
            if gp > 0:
                ops = [I2] * self.n
                ops[i] = Z
                Ls.append(np.sqrt(gp) * kron_n(ops))
        self.Ls = Ls
        # local observables
        obs = []
        for i in range(self.n):
            for P in (X, Y, Z):
                ops = [I2] * self.n
                ops[i] = P
                obs.append(kron_n(ops))
        self.obs = np.stack(obs)

    def H(self, u):
        return self.Hd + np.tensordot(u, self.Hc, axes=(0, 0))

    def rhs(self, rho, H):
        dr = -1j * (H @ rho - rho @ H)
        for L in self.Ls:
            Ld = L.conj().T
            dr += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
        return dr

    def evolve(self, rho0, controls):
        """controls: (T, n_ch) -> rhos (T+1, dim, dim). RK4."""
        T = controls.shape[0]
        dt = self.dt
        rhos = np.empty((T + 1, self.dim, self.dim), dtype=np.complex128)
        rhos[0] = rho0
        rho = rho0.copy()
        for t in range(T):
            H = self.H(controls[t])
            k1 = self.rhs(rho, H)
            k2 = self.rhs(rho + 0.5 * dt * k1, H)
            k3 = self.rhs(rho + 0.5 * dt * k2, H)
            k4 = self.rhs(rho + dt * k3, H)
            rho = rho + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            rho = (rho + rho.conj().T) / 2
            rho = rho / np.trace(rho)
            rhos[t + 1] = rho
        return rhos


def fidelity(rho, sigma):
    # Uhlmann fidelity via eigenvalues (stable for 16x16)
    try:
        w, V = np.linalg.eigh(rho)
        w = np.clip(w, 0, None)
        sq = (V * np.sqrt(w)) @ V.conj().T
        m = sq @ sigma @ sq
        ev = np.clip(np.linalg.eigvalsh(m), 0, None)
        return float(np.real(np.sum(np.sqrt(ev))) ** 2)
    except Exception:
        return float(np.real(np.trace(rho @ sigma)))


def trace_dist(rho, sigma):
    ev = np.linalg.eigvalsh(rho - sigma)
    return float(0.5 * np.sum(np.abs(ev)))


def random_rho(dim, rng):
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_controls(T, n_ch, rng, amp=0.1):
    raw = rng.standard_normal((T, n_ch))
    # smooth with moving average length 3
    sm = raw.copy()
    sm[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:]) / 3
    return amp * np.tanh(sm)


def expand_segments(seg, T):
    """seg: (S,C) piecewise constant -> (T,C)."""
    S = seg.shape[0]
    rep = T // S
    out = np.repeat(seg, rep, axis=0)
    if out.shape[0] < T:
        pad = np.zeros((T - out.shape[0], seg.shape[1]))
        out = np.vstack([out, pad])
    return out[:T]
