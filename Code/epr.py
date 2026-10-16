"""Entropic EPR comparator (Schneeloch & Howland, PRA 97, 042338, 2018), discretized form
E_F >= log2(1/c) - H(X_A|X_B) - H(K_A|K_B), with c = max_{a,b} ||P_a Q_b||^2 <= s_x s_k / n on the
n-point grid. Outside-window events are handled conservatively: the conditional entropy of the
fine-binned variable is bounded by that of the merged variable plus Pr(outside) log2(#outside bins)."""
import numpy as np
from core import *

def cond_entropy(J):
    """H(A|B) in bits for joint table J[a,b]."""
    pb = J.sum(0); H = 0.0
    for b in range(J.shape[1]):
        if pb[b] <= 0: continue
        col = J[:, b]/pb[b]; col = col[col > 0]; H += pb[b]*(-(col*np.log2(col)).sum())
    return H

def epr_bound(g, comps, a, b, p):
    labx, labk = bins(g.x, a, p), bins(g.k, b, p)
    sx = np.bincount(labx, minlength=p+1)[:p].max(); sk = np.bincount(labk, minlength=p+1)[:p].max()
    nout_x = int(np.ceil((labx == p).sum()/sx)); nout_k = int(np.ceil((labk == p).sum()/sk))
    Jx = np.zeros((p+1, p+1)); Jk = np.zeros((p+1, p+1))
    for w, Psi in comps:
        np.add.at(Jx, (labx[:, None].repeat(g.nx, 1), labx[None, :].repeat(g.nx, 0)), w*np.abs(Psi)**2)
        np.add.at(Jk, (labk[:, None].repeat(g.nx, 1), labk[None, :].repeat(g.nx, 0)), w*np.abs(ff(Psi))**2)
    Hx = cond_entropy(Jx) + Jx[p, :].sum()*np.log2(nout_x)
    Hk = cond_entropy(Jk) + Jk[p, :].sum()*np.log2(nout_k)
    logc = np.log2(g.nx/(sx*sk))
    return dict(EF_lb=float(logc - Hx - Hk), log2_1_over_c=float(logc), Hx=float(Hx), Hk=float(Hk), sx=int(sx), sk=int(sk))
