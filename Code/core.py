"""Discretized optical model: grid, aperture projectors consistent with camera labels,
concentration-deficit operator, biphoton states."""
import numpy as np

class Grid:
    def __init__(self, nx=256, L=24.0):
        self.nx, self.L = nx, L
        self.dx = 2*L/nx
        self.x = -L + self.dx*np.arange(nx)
        self.k = 2*np.pi*np.fft.fftfreq(nx, self.dx)
        self.F = np.fft.fft(np.eye(nx), axis=0, norm='ortho')   # unitary DFT

def bins(coord, a, p):
    """camera labels: p equal pixels on the half-open window [-a, a); label p = outside."""
    edges = np.linspace(-a, a, p+1)
    lab = np.digitize(coord, edges) - 1
    lab[(coord < -a) | (coord >= a)] = p
    return lab

def projectors(g, a, b, p):
    """P_X and P_K defined from exactly the same labels as the camera outcomes."""
    labx, labk = bins(g.x, a, p), bins(g.k, b, p)
    PX = np.diag((labx < p).astype(float))
    PK = np.real(g.F.conj().T @ np.diag((labk < p).astype(float)) @ g.F)
    PK = (PK + PK.T)/2
    return PX, PK, labx, labk

def deficit_basis(g, a, b, p, N, alpha=0.5):
    """H_alpha = 2[alpha(I-PX)+(1-alpha)(I-PK)]; returns spectrum, retained basis and
    validated constants (Weyl/backward-error margin) plus a Davis-Kahan subspace bound."""
    PX, PK, labx, labk = projectors(g, a, b, p)
    I = np.eye(g.nx)
    H = 2*(alpha*(I-PX) + (1-alpha)*(I-PK)); H = (H+H.T)/2
    h, U = np.linalg.eigh(H)
    V = U[:, :N]
    # rigorous-in-floating-point margins: eigh backward error ~ n*eps*||H||
    margin = 10*g.nx*np.finfo(float).eps*np.linalg.norm(H, 2)
    res = np.linalg.norm(H @ V - V @ np.diag(h[:N]), 2)
    sep = h[N] - h[N-1]
    sin_theta = (res + margin)/max(sep - 2*margin, 1e-300)   # Davis-Kahan sin-theta bound
    return dict(h=h, V=V, ell=max(h[0]-margin, 0.0), gap=h[N]-margin, H=H,
                labx=labx, labk=labk, margin=margin, sin_theta=sin_theta, sep=sep)

def double_gaussian(g, sp, sm):
    X1, X2 = np.meshgrid(g.x, g.x, indexing='ij')
    psi = np.exp(-(X1+X2)**2/(4*sp**2) - (X1-X2)**2/(4*sm**2))
    return psi/np.linalg.norm(psi)

def k_eff(sp, sm):
    """inverse-purity effective Schmidt mode number of the 1D double-Gaussian state."""
    return 0.5*(sp/sm + sm/sp)

def hg_mode(g, n, w, x0=0.0, k0=0.0):
    from numpy.polynomial.hermite import hermval
    c = np.zeros(n+1); c[n] = 1
    u = (g.x-x0)/w
    f = hermval(u, c)*np.exp(-u**2/2)*np.exp(1j*k0*g.x)
    return f/np.linalg.norm(f)

def ff(Psi):
    return np.fft.fft2(Psi, norm='ortho')

def prolate_eigs(c, n_quad=(80, 120), nmax=12):
    """continuum eigenvalues of P_X P_K P_X on [-a,a] with band b (c = ab) by Gauss-Legendre
    Nystrom discretization of sin(c(s-t))/(pi(s-t)) on [-1,1]; two orders for convergence."""
    out = []
    for nq in n_quad:
        t, w = np.polynomial.legendre.leggauss(nq)
        S, T = np.meshgrid(t, t, indexing='ij')
        D = S - T
        with np.errstate(invalid='ignore', divide='ignore'):
            K = np.where(np.abs(D) < 1e-15, c/np.pi, np.sin(c*D)/(np.pi*D))
        sw = np.sqrt(w)
        lam = np.linalg.eigvalsh(sw[:, None]*K*sw[None, :])[::-1]
        out.append(lam[:nmax])
    return out[-1], float(np.max(np.abs(out[-1]-out[0])))
