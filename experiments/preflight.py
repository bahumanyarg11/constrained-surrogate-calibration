import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import sys,numpy as np,time,warnings
warnings.filterwarnings('ignore'); np.seterr(all='ignore')
from physics import QSys, expand_segments
s=QSys(n_qubits=4,total_time=5.0,dt=0.05)
exc=np.zeros(16,dtype=complex); exc[-1]=1.0
gnd=np.zeros(16,dtype=complex); gnd[0]=1.0; g0=np.outer(gnd,gnd.conj())
S=8
def fid(p,A):
    u=expand_segments(np.clip(p,-A,A).reshape(S,4),100)
    r=s.evolve(g0,u)[-1]
    return float(np.clip(np.real(exc.conj()@r@exc),0,1))
for A in [0.5,1.0,2.0,3.0]:
    rng=np.random.default_rng(7); p=0.1*rng.standard_normal(S*4)
    m=np.zeros_like(p); v=np.zeros_like(p); t0=time.time()
    for t in range(1,41):
        f0=1-fid(p,A); g=np.zeros_like(p)
        for j in range(len(p)):
            dp=p.copy(); dp[j]+=2e-3; g[j]=((1-fid(dp,A))-f0)/2e-3
        m=.9*m+.1*g; v=.999*v+.001*g*g
        p=np.clip(p-0.08*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8),-A,A)
    print(f'CTRL_MAX={A}: GRAPE ceiling(40it) fid={fid(p,A):.4f}  [{time.time()-t0:.0f}s]',flush=True)
