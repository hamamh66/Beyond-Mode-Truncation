"""Continuum certificate for one geometry (Section 'Continuum certificate').
Continuous operators on L^2(R): apertures X=[-a,a), K=[-b,b); prolate eigenpairs of T=P_X P_K P_X by
Gauss-Legendre Nystrom; explicit Bernstein-ellipse remainder bounds for every quadrature; eigenvalue
enclosures by residuals plus a trace count; low-branch modes chi_n=(psi_n+P_K psi_n/sqrt(lam_n))/nu_n;
compressed pixel effects with operator-norm error bounds; exact source probabilities."""
import numpy as np
from math import erf, sqrt, pi
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermval

FP_MARGIN = 1e-12      # explicit margin for floating-point evaluation of the finite sums below

def gl(lo, hi, n):
    t, w = leggauss(n); return 0.5*(hi-lo)*t + 0.5*(hi+lo), 0.5*(hi-lo)*w

def gl_remainder(lo, hi, n, M, rho):
    """|exact - GL_n| for f analytic in the Bernstein ellipse E_rho mapped to [lo,hi], |f|<=M there
    (Trefethen, SIAM Rev. 50 (2008) 67-87): (hi-lo)/2 * 64 M / (15 (1 - rho^-2) rho^(2n))."""
    return 0.5*(hi-lo)*64*M*np.exp(-2*n*np.log(rho))/(15*(1 - rho**-2))

def ellipse_h(lo, hi, rho):
    return 0.5*(hi-lo)*0.5*(rho - 1/rho)   # max |Im z| on the ellipse

class Continuum:
    def __init__(self, a, b, N, p, nnys=120, K=14):
        self.a, self.b, self.N, self.p = a, b, N, p
        x, w = gl(-a, a, nnys); self.xn, self.wn = x, w
        Kmat = self.kern(x[:, None], x[None, :])
        sw = np.sqrt(w); lam, U = np.linalg.eigh(sw[:, None]*Kmat*sw[None, :])
        idx = np.argsort(lam)[::-1][:K]; self.lam = lam[idx]
        self.vals = (U[:, idx]/sw[:, None])                     # psi_hat at nodes, sum w psi^2 = 1
        self.nnys = nnys; self.K = K
        # bound |psi_hat(z)| <= A_n exp(b|Im z|) for the Nystrom interpolant
        self.A = (b/pi)*np.sum(w[:, None]*np.abs(self.vals), 0)/self.lam
        self.L1 = np.sum(w[:, None]*np.abs(self.vals), 0)     # ~||psi||_L1(X)
        self._enclose()

    def kern(self, x, y):
        d = x - y
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(np.abs(d) < 1e-14, self.b/pi, np.sin(self.b*d)/(pi*d))

    def psi(self, x, n):
        """Nystrom interpolant psi_hat_n(x) = (1/lam) sum_j w_j K(x,t_j) psi_j (entire)."""
        return (self.kern(np.asarray(x)[:, None], self.xn[None, :]) @ (self.wn*self.vals[:, n]))/self.lam[n]

    def _enclose(self):
        a, b, n = self.a, self.b, self.nnys
        best = np.inf
        for rho in np.linspace(1.5, 12, 200):
            h = ellipse_h(-a, a, rho)
            M = (b/pi)*np.exp(b*h)*self.A.max()*np.exp(b*h)        # |K(x,y) psi(y)| on ellipse (x real)
            best = min(best, gl_remainder(-a, a, n, M, rho))
        # residual: (T psi_hat - lam psi_hat)(x) is exactly the GL_n remainder of K(x,.)psi_hat(.)
        self.resid = np.sqrt(2*a)*best + FP_MARGIN                 # L2(X) norm bound (unnormalized)
        # normalization: ||psi_hat||^2 = sum w psi^2 (=1) + remainder of psi_hat^2
        bestn = np.inf
        for rho in np.linspace(1.5, 12, 200):
            h = ellipse_h(-a, a, rho); M = (self.A.max()*np.exp(b*h))**2
            bestn = min(bestn, gl_remainder(-a, a, n, M, rho))
        self.norm_err = bestn + FP_MARGIN
        r = (self.resid)/np.sqrt(1 - self.norm_err)              # residual for unit vector
        r = np.full(self.K, float(r)); self.r = r
        trace = 2*a*b/pi
        rem = trace - np.sum(self.lam - r)                        # mass of eigenvalues not enclosed
        self.rem = rem
        lo, hi = self.lam - r, self.lam + r
        assert np.all(lo[:-1] > hi[1:]), 'enclosures overlap'
        N = self.N
        assert rem < lo[N], 'cannot exclude a missed eigenvalue above lambda_N'
        self.lamN_up = hi[N]                                      # lambda_N <= lamN_up
        self.g_lo = 1 - np.sqrt(self.lamN_up)
        # Davis-Kahan per retained eigenvector: separation from all other spectrum
        self.sin = np.zeros(N)
        for k in range(N):
            others = [abs(self.lam[k] - self.lam[m]) - r[m] for m in range(self.K) if m != k] + [self.lam[k] - rem]
            sep = min(others) - r[k]
            self.sin[k] = r[k]/sep
        # error of the reconstructed low-branch vector chi_hat_k vs exact chi_k (L^2(R)):
        # ||L_lam|| <= (1+1/sqrt(lam))/nu(lam), nu = sqrt(2(1+sqrt(lam))); plus lam-uncertainty term
        self.e = np.zeros(N)
        for k in range(N):
            l = self.lam[k]; nu = np.sqrt(2*(1+np.sqrt(l)))
            Lnorm = (1 + 1/np.sqrt(l))/nu
            dpsi = np.sqrt(2)*self.sin[k] + np.sqrt(self.norm_err)
            ls = np.array([l - r[k], l + r[k]])
            c1 = np.abs(1/np.sqrt(2*(1+np.sqrt(ls))) - 1/nu).max()
            c2 = np.abs(1/(np.sqrt(ls)*np.sqrt(2*(1+np.sqrt(ls)))) - 1/(np.sqrt(l)*nu)).max()
            self.e[k] = Lnorm*dpsi + c1 + c2*np.sqrt(l) + FP_MARGIN

    # ---------------- compressed single-arm effects -------------------------------------------
    def Tpsi(self, x, n, nq=200):
        """(P_X P_K psi_hat_n)(x) for x in X by GL quadrature; returns values and a uniform error bound."""
        y, wy = gl(-self.a, self.a, nq)
        val = self.kern(np.asarray(x)[:, None], y[None, :]) @ (wy*self.psi(y, n))
        best = np.inf
        for rho in np.linspace(1.5, 12, 100):
            h = ellipse_h(-self.a, self.a, rho)
            M = (self.b/pi)*np.exp(self.b*h)*self.A[n]*np.exp(self.b*h)
            best = min(best, gl_remainder(-self.a, self.a, nq, M, rho))
        return val, best + FP_MARGIN

    def Fpsi(self, k, n, nq=200):
        """(F psi_hat_n)(k) = (2 pi)^(-1/2) int_X e^{-ikx} psi_hat_n(x) dx and its error bound."""
        x, wx = gl(-self.a, self.a, nq)
        k = np.asarray(k)
        val = (np.exp(-1j*np.outer(k, x)) @ (wx*self.psi(x, n)))/np.sqrt(2*pi)
        best = np.inf; km = np.abs(k).max()
        for rho in np.linspace(1.5, 12, 100):
            h = ellipse_h(-self.a, self.a, rho)
            M = np.exp(km*h)*self.A[n]*np.exp(self.b*h)/np.sqrt(2*pi)
            best = min(best, gl_remainder(-self.a, self.a, nq, M, rho))
        return val, best + FP_MARGIN

    def effects(self, nq_pix=40):
        """single-arm compressed effects C_a = <chi_m|M_a|chi_n> for p near-field pixels + outside,
        and p far-field pixels + outside, with an operator-norm error bound d1 covering quadrature and
        reconstruction errors (relative to the exact low-branch basis)."""
        N, p, a, b = self.N, self.p, self.a, self.b
        lam = self.lam[:N]; nu = np.sqrt(2*(1+np.sqrt(lam)))
        cX = (1 + np.sqrt(lam))/nu          # chi on X = cX * psi (exactly for exact pairs)
        cK = (1 + 1/np.sqrt(lam))/nu        # F chi on K = cK * F psi
        # chi_hat on X uses (psi_hat + T psi_hat / sqrt(lam))/nu, T psi_hat = lam psi_hat + residual
        ex = np.linspace(-a, a, p+1); ek = np.linspace(-b, b, p+1)
        Cx = np.zeros((p+1, N, N)); Ck = np.zeros((p+1, N, N), complex)
        quad_err = 0.0
        for i in range(p):
            x, wx = gl(ex[i], ex[i+1], nq_pix)
            Ps = np.array([self.psi(x, n) for n in range(N)])
            Cx[i] = (cX[:, None]*Ps*wx) @ (cX[:, None]*Ps).T
            k, wk = gl(ek[i], ek[i+1], nq_pix)
            Fs = []; ferr = 0.0
            for n in range(N):
                v, e_ = self.Fpsi(k, n); Fs.append(v); ferr = max(ferr, e_)
            Fs = np.array(Fs)
            Ck[i] = (cK[:, None]*Fs.conj()*wk) @ (cK[:, None]*Fs).T
            # remainder bounds of the pixel integrals (products of entire functions)
            for lo_, hi_, kind in [(ex[i], ex[i+1], 'x'), (ek[i], ek[i+1], 'k')]:
                best = np.inf
                for rho in np.linspace(1.5, 40, 200):
                    h = ellipse_h(lo_, hi_, rho)
                    if kind == 'x':
                        M = (cX.max()*self.A.max()*np.exp(b*(h)))**2
                    else:
                        M = (cK.max()*self.A.max()*np.exp(b*0)*np.exp(a*h)*2*a/np.sqrt(2*pi))**2
                    best = min(best, gl_remainder(lo_, hi_, nq_pix, M, rho))
                quad_err = max(quad_err, best)
            quad_err = max(quad_err, 2*cK.max()**2*ferr*(np.abs(Fs).max() + ferr)*(ek[i+1]-ek[i]))
        # Gram of exact chi is the identity; outside-window effects = I - sum of inside pixels
        Cx[p] = np.eye(N) - Cx[:p].sum(0)
        Ck[p] = np.eye(N) - Ck[:p].sum(0)
        # error budget: entrywise quadrature error -> operator norm <= N * entry error;
        # reconstruction error of chi_hat (e_k) enters each inside effect as e_m + e_n + e_m e_n;
        # outside effects accumulate p inside errors.
        # subspace-level reconstruction error. For chi_hat_n = (psi_hat + P_K psi_hat/sqrt(lam))/nu one has
        # exactly H chi_hat = (1 - sqrt(lam)) chi_hat - rho_n/(sqrt(lam) nu), rho_n = T psi_hat - lam psi_hat,
        # so the residual of the retained cluster is <= sqrt(N) max_n r/(sqrt(lam_n) nu_n).
        resH = np.sqrt(N)*np.max(self.r[:N]/(np.sqrt(lam)*nu)) + FP_MARGIN
        hN1 = 1 - np.sqrt(self.lam[N-1] - self.r[N-1])            # upper bound of retained cluster
        gapc = self.g_lo - hN1
        sinT = resH/(gapc - resH)
        # Gram deviation of chi_hat from orthonormality (quadrature inner products + residual terms)
        Gdev = N*(4*self.norm_err + 4*self.r.max()/np.sqrt(lam.min())) + FP_MARGIN
        resid_term = 2*cX.max()*np.max(self.r)/np.sqrt(lam.min())
        d_in = N*quad_err + resid_term + 2*Gdev + 2*np.sqrt(2)*sinT + FP_MARGIN
        d_out = p*d_in
        self.sinT, self.gapc, self.Gdev = float(sinT), float(gapc), float(Gdev)
        self.d_parts = dict(quad=float(N*quad_err), subspace=float(2*np.sqrt(2)*sinT), gram=float(2*Gdev), resid=float(resid_term), d_in=float(d_in), d_out=float(d_out))
        self.d1 = float(d_out)
        return Cx, Ck

# ---------------- continuum source probabilities ------------------------------------------------
def _ncdf(z): return 0.5*(1 + erf(z/sqrt(2)))

def gauss_pair_probs(edges, s_plus, s_minus, nq=60):
    """joint pixel probabilities for |Psi|^2 = Gaussian with Var(x1+x2)=s_plus^2, Var(x1-x2)=s_minus^2.
    Inside x inside by 2D GL on each pixel rectangle; outside entries from exact marginals."""
    p = len(edges) - 1
    s1 = 0.5*np.sqrt(s_plus**2 + s_minus**2)       # marginal std
    cov = 0.25*np.array([[s_plus**2 + s_minus**2, s_plus**2 - s_minus**2],
                         [s_plus**2 - s_minus**2, s_plus**2 + s_minus**2]])
    Ci = np.linalg.inv(cov); norm = 1/(2*pi*np.sqrt(np.linalg.det(cov)))
    J = np.zeros((p+1, p+1))
    for i in range(p):
        x, wx = gl(edges[i], edges[i+1], nq)
        for j in range(p):
            y, wy = gl(edges[j], edges[j+1], nq)
            X, Y = np.meshgrid(x, y, indexing='ij')
            Q = Ci[0, 0]*X**2 + 2*Ci[0, 1]*X*Y + Ci[1, 1]*Y**2
            J[i, j] = norm*np.sum(np.outer(wx, wy)*np.exp(-0.5*Q))
    marg = np.array([_ncdf(edges[i+1]/s1) - _ncdf(edges[i]/s1) for i in range(p)])
    J[:p, p] = marg - J[:p, :p].sum(1); J[p, :p] = marg - J[:p, :p].sum(0)
    J[p, p] = 1 - J[:p, :p].sum() - J[:p, p].sum() - J[p, :p].sum()
    out_marg = 1 - marg.sum()
    return J, out_marg

def hg_pixel_probs(edges, n, w, nq=60):
    """single-arm probabilities of |HG_n(x/w)|^2 normalized, on pixels + outside."""
    from math import factorial
    c = np.zeros(n+1); c[n] = 1
    nrm = 1/np.sqrt(w*np.sqrt(pi)*2**n*factorial(n))
    p = len(edges) - 1; P = np.zeros(p+1)
    for i in range(p):
        x, wx = gl(edges[i], edges[i+1], nq); u = x/w
        P[i] = np.sum(wx*(nrm*hermval(u, c)*np.exp(-u**2/2))**2)
    P[p] = 1 - P[:p].sum()
    return P

def source_probs(a, b, p, sp, sm, w_tail=0.0, tail_n=8, tail_w=1.0):
    ex = np.linspace(-a, a, p+1); ek = np.linspace(-b, b, p+1)
    Jx, ox = gauss_pair_probs(ex, sp, sm)
    Jk, ok = gauss_pair_probs(ek, 1/sp, 1/sm)          # |Psi~|^2 has Var(k1+k2)=1/sp^2, Var(k1-k2)=1/sm^2
    if w_tail > 0:
        tx = hg_pixel_probs(ex, tail_n, tail_w); tk = hg_pixel_probs(ek, tail_n, 1/tail_w)
        Jx = (1-w_tail)*Jx + w_tail*np.outer(tx, tx); Jk = (1-w_tail)*Jk + w_tail*np.outer(tk, tk)
    P = np.concatenate([Jx.flatten(), Jk.flatten()])
    deficit = Jx[p, :].sum() + Jk[p, :].sum()          # arm s: Pr(x out) + Pr(k out); symmetric source
    return P, [deficit, deficit]


# ---------------- reference projected state (not needed for the certificate) ------------------
def reference_tau(C, sp, sm, w_tail=0.0, tail_n=8, nq=80):
    a, b, N = C.a, C.b, C.N
    lam = C.lam[:N]; nu = np.sqrt(2*(1+np.sqrt(lam)))
    x, wx = gl(-a, a, nq); k, wk = gl(-b, b, nq)
    Ps = np.array([C.psi(x, n) for n in range(N)])                       # psi on X
    Fp = np.array([C.Fpsi(k, n)[0] for n in range(N)])/np.sqrt(lam)[:, None]   # F phi on K
    N0 = 1/np.sqrt(pi*sp*sm); Nt = np.sqrt(sp*sm/pi)
    al = 1/(4*sp**2) + 1/(4*sm**2); be = 1/(2*sp**2) - 1/(2*sm**2)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Psi = N0*np.exp(-al*X**2 - al*Y**2 - be*X*Y)
    Kk, Qq = np.meshgrid(k, k, indexing='ij')
    Pst = Nt*np.exp(-sp**2*(Kk+Qq)**2/4 - sm**2*(Kk-Qq)**2/4)
    Xm, Qm = np.meshgrid(x, k, indexing='ij')
    FyP = N0*np.exp(-al*Xm**2)*np.sqrt(pi/al)/np.sqrt(2*pi)*np.exp((be*Xm + 1j*Qm)**2/(4*al))
    I_pp = (Ps*wx) @ Psi @ (Ps*wx).T
    I_ff = (Fp.conj()*wk) @ Pst @ (Fp.conj()*wk).T
    I_pf = (Ps*wx) @ FyP @ (Fp.conj()*wk).T
    Cc = (I_pp + I_pf + I_pf.T + I_ff)/np.outer(nu, nu)
    c = Cc.flatten(); T = (1 - w_tail)*np.outer(c, c.conj())
    if w_tail > 0:
        from math import factorial
        cc_ = np.zeros(tail_n+1); cc_[tail_n] = 1; nrm = 1/np.sqrt(np.sqrt(pi)*2**tail_n*factorial(tail_n))
        tx = nrm*hermval(x, cc_)*np.exp(-x**2/2); tk = ((-1j)**tail_n)*nrm*hermval(k, cc_)*np.exp(-k**2/2)
        ct = ((Ps*wx) @ tx + (Fp.conj()*wk) @ tk)/nu
        v = np.kron(ct, ct); T = T + w_tail*np.outer(v, v.conj())
    return T

class CSetup:
    """certification setup built from continuum compressed effects (interface of cert.Setup)."""
    def __init__(self, C):
        self.N, self.p = C.N, C.p; self.d = C.N**2; self.m_set = (C.p+1)**2
        Cx, Ck = C.effects()
        self.Ax, self.Ak = list(Cx), list(Ck)
        rows = []
        for A in (self.Ax, self.Ak):
            for i in range(C.p+1):
                for j in range(C.p+1):
                    rows.append(np.kron(A[i], A[j]).T.flatten(order='F'))
        self.Amat = np.array(rows); self.Ar, self.Ai = self.Amat.real.copy(), self.Amat.imag.copy()
        d1 = C.d1; self.d_a = 2*d1 + d1**2
        self.gap, self.ell = C.g_lo, 0.0
    def eps_from_delta(self, dplus):
        e = [min(1.0, dp/self.gap) for dp in dplus]
        return min(1.0, sum(e)), e
