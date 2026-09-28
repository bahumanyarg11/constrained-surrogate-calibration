"""Four density-matrix output heads sharing one FNO trunk.

unconstrained : raw regression (no physical structure)
herm_trace    : Hermitian symmetrisation + trace division   <- "standard FNO" in the literature
eigh_clip     : herm_trace + eigenvalue clipping            <- post-hoc PSD correction
cholesky      : rho = L L^dag / Tr(L L^dag)                 <- exact PSD + trace-1 by construction
"""
import torch, torch.nn as nn, torch.nn.functional as Fn

HEADS = ['unconstrained', 'herm_trace', 'eigh_clip', 'cholesky', 'cholesky_gated']
TRUNKS = ['fno', 'transformer', 'mixer']


class SpectralConv1d(nn.Module):
    def __init__(self, ch, modes):
        super().__init__()
        self.modes = modes; self.ch = ch
        sc = 1.0 / (ch * ch)
        self.wr = nn.Parameter(sc * torch.randn(ch, ch, modes))
        self.wi = nn.Parameter(sc * torch.randn(ch, ch, modes))

    def forward(self, x):
        B, C, T = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        w = torch.complex(self.wr[:, :, :m], self.wi[:, :, :m])
        of = torch.zeros(B, self.ch, xf.shape[-1], dtype=torch.cfloat, device=x.device)
        of[:, :, :m] = torch.einsum('bim,iom->bom', xf[:, :, :m].to(torch.cfloat), w)
        return torch.fft.irfft(of, n=T, dim=-1)


class FourierLayer(nn.Module):
    def __init__(self, ch, modes):
        super().__init__()
        self.spec = SpectralConv1d(ch, modes)
        self.local = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spec(x) + self.local(x))) + x


class TransformerTrunk(nn.Module):
    """Pre-norm Transformer encoder over the time axis."""
    def __init__(self, in_ch, width, depth, T=101, nhead=4):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.pos = nn.Parameter(0.02 * torch.randn(1, T, width))
        layer = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=2 * width,
                                           batch_first=True, dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        h = self.inp(x).permute(0, 2, 1)
        h = h + self.pos[:, :h.shape[1]]
        return self.enc(h).permute(0, 2, 1)


class MixerBlock(nn.Module):
    def __init__(self, width, T, exp=2):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.tok = nn.Sequential(nn.Linear(T, exp * T), nn.GELU(), nn.Linear(exp * T, T))
        self.n2 = nn.LayerNorm(width)
        self.ch = nn.Sequential(nn.Linear(width, exp * width), nn.GELU(), nn.Linear(exp * width, width))

    def forward(self, h):
        h = h + self.tok(self.n1(h).transpose(1, 2)).transpose(1, 2)
        return h + self.ch(self.n2(h))


class MixerTrunk(nn.Module):
    """MLP-Mixer: no attention, no spectral convolution."""
    def __init__(self, in_ch, width, depth, T=101):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList([MixerBlock(width, T) for _ in range(depth)])

    def forward(self, x):
        h = self.inp(x).permute(0, 2, 1)
        for b in self.blocks:
            h = b(h)
        return h.permute(0, 2, 1)


def _cholesky_rho(flat, d, normalise=True):
    """flat: (...,d*d) real -> rho, exactly PSD. Returns (rho, log mass) where mass=Tr(LL^dag).

    The mass is the quantity a trace-normalising head throws away. In-distribution it is
    pinned to 1 by a training penalty; out of distribution it drifts, which is exactly the
    distress signal that normalisation destroys."""
    n_off = d * (d - 1) // 2
    diag = Fn.softplus(flat[..., :d]) + 1e-6
    re = flat[..., d:d + n_off]
    im = flat[..., d + n_off:d + 2 * n_off]
    idx = torch.tril_indices(d, d, offset=-1, device=flat.device)
    L = torch.zeros(*flat.shape[:-1], d, d, dtype=torch.complex128, device=flat.device)
    ar = torch.arange(d, device=flat.device)
    L[..., ar, ar] = diag.to(torch.complex128)
    L[..., idx[0], idx[1]] = torch.complex(re.double(), im.double())
    rho = L @ L.conj().transpose(-1, -2)
    tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).real
    return rho / tr.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1), torch.log(tr.clamp_min(1e-12))


class FNO(nn.Module):
    def __init__(self, head='herm_trace', dim=16, n_ch=4, modes=16, width=64, depth=4,
                 trunk='fno', T=101):
        super().__init__()
        assert head in HEADS and trunk in TRUNKS
        self.head_kind = head; self.trunk_kind = trunk; self.dim = dim; self.n_ch = n_ch
        self.aux_logmass = None
        out = dim * dim if head.startswith('cholesky') else 2 * dim * dim
        in_ch = 2 * dim * dim + n_ch
        if trunk == 'fno':
            self.lift = nn.Conv1d(in_ch, width, 1)
            self.layers = nn.ModuleList([FourierLayer(width, modes) for _ in range(depth)])
        elif trunk == 'transformer':
            self.trunk = TransformerTrunk(in_ch, width, depth, T=T)
        else:
            self.trunk = MixerTrunk(in_ch, width, depth, T=T)
        self.proj = nn.Conv1d(width, width, 1)
        self.head = nn.Linear(width, out)
        self.act = nn.GELU()

    def forward(self, rho0, controls):
        B, T, _ = controls.shape
        d = self.dim
        x0 = torch.cat([rho0.real.reshape(B, -1), rho0.imag.reshape(B, -1)], -1).float()
        x0 = x0.unsqueeze(1).expand(B, T + 1, -1)
        cp = torch.cat([torch.zeros(B, 1, self.n_ch), controls.float()], 1)
        x = torch.cat([x0, cp], -1).permute(0, 2, 1)
        if self.trunk_kind == 'fno':
            x = self.act(self.lift(x))
            for l in self.layers:
                x = l(x)
        else:
            x = self.trunk(x)
        x = self.act(self.proj(x)).permute(0, 2, 1)
        f = self.head(x)
        if self.head_kind.startswith('cholesky'):
            rho, logmass = _cholesky_rho(f, d)
            self.aux_logmass = logmass
            return rho
        rho = torch.complex(f[..., :d * d].double(), f[..., d * d:].double()).reshape(B, T + 1, d, d)
        if self.head_kind == 'unconstrained':
            return rho
        rho = (rho + rho.conj().transpose(-1, -2)) / 2
        if self.head_kind == 'eigh_clip':
            ev, v = torch.linalg.eigh(rho)
            rho = v @ (torch.clamp(ev, min=1e-8).to(rho.dtype).unsqueeze(-1) * v.conj().transpose(-1, -2))
        tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).real.unsqueeze(-1).unsqueeze(-1)
        return rho / tr.clamp_min(1e-12) if self.head_kind == 'eigh_clip' else rho / tr
