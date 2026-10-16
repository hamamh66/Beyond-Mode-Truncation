"""Measurement model, certification programs (HC, GP, AG, CS, Schmidt) and validated
Lagrangian lower bounds obtained from solver dual variables with residual correction."""
import numpy as np, cvxpy as cp, clarabel, scipy.sparse as sp
from scipy.stats import beta as _beta
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones
from core import *

class Setup:
    def __init__(self, g, a, b, N, p, alpha=0.5):
        self.g, self.a, self.b, self.N, self.p = g, a, b, N, p
        B = deficit_basis(g, a, b, p, N, alpha); self.B = B
        self.V, self.ell, self.gap, self.H = B['V'], B['ell'], B['gap'], B['H']
        self.labx, self.labk = B['labx'], B['labk']
        FV = g.F @ self.V
        self.Ax = [self.V.T @ np.diag((self.labx == i).astype(float)) @ self.V for i in range(p+1)]
        self.Ak = [FV.conj().T @ np.diag((self.labk == i).astype(float)) @ FV for i in range(p+1)]
        self.d = N*N; self.m_set = (p+1)**2
        rows = []
        for A in (self.Ax, self.Ak):
            for i in range(p+1):
                for j in range(p+1):
                    rows.append(np.kron(A[i], A[j]).T.flatten(order='F'))
        self.Amat = np.array(rows); self.Ar, self.Ai = self.Amat.real.copy(), self.Amat.imag.copy()
        # numerical compression error of effects from the subspace bound: ||QMQ - Q^M Q^|| <= 2 sin(theta)
        self.d_a = 2*B['sin_theta']

    def probs(self, comps, crosstalk=0.0):
        """exact outcome probabilities of the untruncated grid state; optional nearest-neighbour
        pixel crosstalk kappa (fraction moved to each adjacent inside pixel, per arm)."""
        g, p = self.g, self.p
        P = np.zeros(2*self.m_set)
        for w, Psi in comps:
            nf = np.abs(Psi)**2; fk = np.abs(ff(Psi))**2
            Pn = np.zeros((p+1, p+1)); Pk = np.zeros((p+1, p+1))
            np.add.at(Pn, (self.labx[:, None].repeat(g.nx, 1), self.labx[None, :].repeat(g.nx, 0)), nf)
            np.add.at(Pk, (self.labk[:, None].repeat(g.nx, 1), self.labk[None, :].repeat(g.nx, 0)), fk)
            P += w*np.concatenate([Pn.flatten(), Pk.flatten()])
        if crosstalk > 0:
            T = np.eye(p+1)
            for i in range(p):
                for j in (i-1, i+1):
                    if 0 <= j < p:
                        T[j, i] += crosstalk; T[i, i] -= crosstalk
            out = []
            for blk in (P[:self.m_set], P[self.m_set:]):
                M = blk.reshape(p+1, p+1); out.append((T @ M @ T.T).flatten())
            P = np.concatenate(out)
        return P

    def tau(self, comps):
        T = np.zeros((self.d, self.d), complex)
        for w, Psi in comps:
            c = (self.V.T @ Psi @ self.V).flatten(); T += w*np.outer(c, c.conj())
        return T

    def deficits(self, comps):
        out = []
        for arm in (0, 1):
            dl = 0.0
            for w, Psi in comps:
                M = Psi if arm == 0 else Psi.T
                dl += w*np.real(np.trace((M @ M.conj().T) @ self.H))
            out.append(dl)
        return out

    def deficits_from_probs(self, P):
        p1 = self.p + 1
        nf = P[:self.m_set].reshape(p1, p1); kf = P[self.m_set:].reshape(p1, p1)
        return [nf[self.p, :].sum() + kf[self.p, :].sum(), nf[:, self.p].sum() + kf[:, self.p].sum()]

    def eps_from_delta(self, dplus):
        e = []
        for dp in dplus:
            if dp - self.ell < -1e-12:
                raise ValueError("inconsistent deficit bound")
            e.append(min(1.0, (dp - self.ell)/(self.gap - self.ell) + 2*self.B['sin_theta']))
        return min(1.0, sum(e)), e

def nu(T, N):
    Tp = T.reshape(N, N, N, N).transpose(0, 3, 2, 1).reshape(N*N, N*N)
    return (np.abs(np.linalg.eigvalsh(Tp)).sum() - np.real(np.trace(T)))/2

def negativity_pure(Psi):
    s = np.linalg.svd(Psi, compute_uv=False); s = s/np.linalg.norm(s)
    return ((s.sum())**2 - 1)/2

def keff_numeric(Psi):
    s = np.linalg.svd(Psi, compute_uv=False)**2; s = s/s.sum()
    return 1/np.sum(s**2)

# ---------------------------------------------------------------- confidence regions
def hoeffding(m, n, alpha=0.05):
    return np.sqrt(np.log(2*m/alpha)/(2*n))

def clopper_pearson(k, n, a):
    lo = np.where(k > 0, _beta.ppf(a/2, k, n-k+1), 0.0)
    hi = np.where(k < n, _beta.ppf(1-a/2, k+1, n-k), 1.0)
    return lo, hi

# ---------------------------------------------------------------- programs
def _pt_perm(N):
    d = N*N
    idx = np.arange(d*d).reshape(d, d, order='F')
    return idx.reshape(N, N, N, N).transpose(0, 3, 2, 1).reshape(d, d).flatten(order='F')

def _herm(d):
    X = cp.Variable((d, d), symmetric=True); Y = cp.Variable((d, d))
    return X, Y, [Y == -Y.T]

def _E(X, Y):
    return cp.bmat([[X, -Y], [Y, X]])

def _pt(X, perm, d):
    return cp.reshape(cp.vec(X, order='F')[perm], (d, d), order='F')

def refined_radius(eps):
    """max_{0<=t<=eps} sqrt(t(4-3t)): trace-norm bound ||rho - Q rho Q||_1 <= sqrt(t(1+3r)), r = 1-t."""
    return float(np.sqrt(eps*(4-3*eps))) if eps <= 2/3 else 2/np.sqrt(3)

def _geometry_rows(S):
    """rows of Tr(sigma G~_a), G~_a = C~_a - C~_a^2 (projective pixel effects), and the certified
    operator-norm margin eta_G with G_a <= G~_a + eta_G I."""
    if getattr(S, 'Grows', None) is None:
        p = S.p; rows = []
        for A in (S.Ax, S.Ak):
            for i in range(p+1):
                for j in range(p+1):
                    C = np.kron(A[i], A[j]); G = C - C @ C; G = (G + G.conj().T)/2
                    rows.append(G.T.flatten(order='F'))
        S.Grows = np.array(rows); S.Gr, S.Gi = S.Grows.real.copy(), S.Grows.imag.copy()
        S.eta_G = float(S.d_a*(2 + S.d_a)) + 1e-14
    return S.Gr, S.Gi, S.eta_G

def build_feasible_set(S, X, Y, lo, hi, eps, method, c_extra=0.0):
    """Feasible sets. lo/hi: statistical interval of each outcome probability; c_extra: calibration
    radius c_a (array or scalar); S.d_a: operator-norm error of the compressed effects.
    method tokens: 'HC', 'GP', 'AG' (2 sqrt eps aggregate), 'AGr' (refined aggregate),
    'CS' (outcome-resolved cones with latent exact compressed probabilities), '+AG' (intersect
    refined aggregate), '+G' (geometry cones x_a^2 <= 4 t Tr(sigma G_a))."""
    lo = np.asarray(lo - c_extra, float); hi = np.asarray(hi + c_extra, float)
    s_t = S.Ar @ cp.vec(X, order='F') - S.Ai @ cp.vec(Y, order='F')     # Tr(sigma C~_a)
    tr = cp.trace(X)
    C = [_E(X, Y) >> 0]; bounds = {X.id: 1.0, Y.id: 1.0}
    ms = S.m_set; m = len(lo); da = S.d_a
    toks = method.split('+'); base = toks[0]
    def aggregate(radius, sexp):
        t = cp.Variable(m, nonneg=True); bounds[t.id] = 2.0
        cen = (lo+hi)/2; rad = (hi-lo)/2 + da
        return [sexp - cen <= t, cen - sexp <= t,
                cp.sum(t[:ms]) <= radius + rad[:ms].sum(), cp.sum(t[ms:]) <= radius + rad[ms:].sum()]
    if base == 'HC':
        C += [tr == 1, s_t <= hi + da, s_t >= lo - da]
    elif base in ('GP', 'AG', 'AGr'):
        r = 2*np.sqrt(eps) if base != 'AGr' else refined_radius(eps)
        C += [tr <= 1, tr >= 1-eps, s_t <= hi + r + da, s_t >= lo - r - da]
        if base in ('AG', 'AGr'):
            C += aggregate(r, s_t)
    elif base == 'CS':
        # latent exact compressed probability z_a with |z_a - s_a| <= d_a Tr(sigma) <= d_a, eliminated
        # conservatively: x^2 <= 4 z w <= 4 (s + d_a) w and |p - s - w - x| <= u + c + d_a
        w = cp.Variable(m, nonneg=True); x = cp.Variable(m)
        bounds[w.id] = 1.0; bounds[x.id] = 2.0
        zb = s_t + da
        C += [tr <= 1, tr >= 1-eps,
              cp.sum(w[:ms]) == 1 - tr, cp.sum(w[ms:]) == 1 - tr,
              cp.sum(x[:ms]) == 0, cp.sum(x[ms:]) == 0,
              s_t + w + x <= hi + da, s_t + w + x >= lo - da,
              cp.SOC(zb + w, cp.vstack([x, zb - w]), axis=0)]
        if 'AG' in toks[1:]:
            C += aggregate(refined_radius(eps), s_t)
        if 'G' in toks[1:]:
            Gr, Gi, eta = _geometry_rows(S)
            g = Gr @ cp.vec(X, order='F') - Gi @ cp.vec(Y, order='F') + eta*tr
            t = 1 - tr
            C += [cp.SOC(t + g, cp.vstack([x, t - g]), axis=0)]
    return C, tr, bounds

_EPS = np.finfo(float).eps

def _project_dual(z, dims):
    """projection onto K* followed by a certified shift: PSD blocks are shifted by
    max(0, -lambda_min) + 10 n eps ||M|| (backward-stable eigensolver), SOC heads are inflated."""
    z = z.copy(); i = dims.zero
    z[i:i+dims.nonneg] = np.maximum(z[i:i+dims.nonneg], 0); i += dims.nonneg
    for q in dims.soc:
        t, v = z[i], z[i+1:i+q]; nv = np.linalg.norm(v)
        if nv > t:
            if nv <= -t: z[i:i+q] = 0; t = 0.0; nv = 0.0
            else:
                a_ = (t + nv)/2; z[i] = a_; z[i+1:i+q] = a_*v/nv
        nv = np.linalg.norm(z[i+1:i+q]); z[i] = max(z[i], nv*(1 + 4*q*_EPS) + 1e-300)
        i += q
    for n in dims.psd:
        L = n*(n+1)//2; vec = z[i:i+L]
        cols, rows = np.meshgrid(np.arange(n), np.arange(n))
        mask = rows <= cols
        r_, c_ = rows.T[mask.T], cols.T[mask.T]
        off = r_ != c_
        vals = vec.copy(); vals[off] /= np.sqrt(2)
        M = np.zeros((n, n)); M[r_, c_] = vals; M[c_, r_] = vals
        lam, U = np.linalg.eigh(M); M = (U*np.maximum(lam, 0)) @ U.T; M = (M + M.T)/2
        lmin = np.linalg.eigvalsh(M)[0]
        shift = max(0.0, -lmin) + 10*n*_EPS*max(np.linalg.norm(M, 2), 1e-300)
        M = M + shift*np.eye(n)
        vals = M[r_, c_].copy(); vals[off] *= np.sqrt(2); z[i:i+L] = vals
        i += L
    return z

def certified_solve(prob, bounds, default_bound=None):
    """Clarabel on the cvxpy conic data; validated bound by weak duality (Proposition) with the dual
    repaired into K* and a-priori floating-point error bounds on every evaluated sum."""
    data, chain, inv = prob.get_problem_data('CLARABEL')
    A = sp.csc_matrix(data['A']); b = data['b']; q = data['c']; dims = data['dims']
    n = len(q)
    Pm = sp.csc_matrix((n, n))
    def _run(reg):
        st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = 800
        if reg:
            st.static_regularization_constant = 1e-7; st.presolve_enable = False
        return clarabel.DefaultSolver(Pm, q, A, b, dims_to_solver_cones(dims), st).solve()
    sols = [_run(False)]
    st0 = str(sols[0].status)
    if 'NumericalError' in st0 or 'MaxIterations' in st0 or 'InsufficientProgress' in st0:
        sols.append(_run(True))
    if all(('Infeasible' in str(x.status) and 'Almost' not in str(x.status)) for x in sols):
        return None, None, str(sols[-1].status)
    col = data['param_prob'].var_id_to_col; starts = sorted(col.values()) + [n]
    Bv = np.full(n, np.nan)
    for vid, c0 in col.items():
        c1 = starts[starts.index(c0)+1]
        if vid not in bounds:
            if default_bound is None: raise ValueError('unbounded canonical variable')
            Bv[c0:c1] = default_bound
        else:
            Bv[c0:c1] = bounds[vid]
    assert not np.isnan(Bv).any(), 'canonical variables without bounds (auxiliaries?)'
    absA = abs(A)
    kcol = int(np.diff(A.indptr).max()) + 2; gk = kcol*_EPS/(1 - kcol*_EPS)
    gm = (len(b) + 2)*_EPS/(1 - (len(b) + 2)*_EPS); gn = (n + 2)*_EPS/(1 - (n + 2)*_EPS)
    best = None
    for k_, sol in enumerate(sols):
        z = _project_dual(np.array(sol.z), dims)
        r = q + A.T @ z
        er = gk*(np.abs(q) + absA.T @ np.abs(z))
        btz = float(b @ z); ebtz = gm*float(np.abs(b) @ np.abs(z))
        corr = float((np.abs(r) + er) @ Bv); ecorr = gn*corr
        lb_k = -btz - ebtz - corr - ecorr
        if best is None or lb_k > best[0]:
            best = (lb_k, sol, str(sol.status) + (' (regularized retry)' if k_ else ''))
    lb, sol, status = best
    primal = float(q @ np.array(sol.x))
    return primal, lb, status

def certify_negativity(S, lo, hi, eps, method, c_extra=0.0):
    d, N = S.d, S.N; perm = _pt_perm(N)
    X, Y, C0 = _herm(d); RX, RY, C1 = _herm(d)
    C, tr, bounds = build_feasible_set(S, X, Y, lo, hi, eps, method, c_extra)
    C += C0 + C1 + [_E(RX, RY) >> 0, cp.trace(RX) <= N,
                    _E(_pt(X, perm, d) + RX, _pt(Y, perm, d) + RY) >> 0]
    bounds[RX.id] = float(N); bounds[RY.id] = float(N)
    pr = cp.Problem(cp.Minimize(cp.trace(RX)), C)
    val, lb, st = certified_solve(pr, bounds, float(N))
    return dict(primal=val, lb=None if lb is None else max(0.0, lb), lb_raw=lb, status=st)

def certify_schmidt(S, lo, hi, eps, method, phi, kmax=None):
    d, N = S.d, S.N
    sv = np.linalg.svd(phi.reshape(N, N), compute_uv=False)**2; sv = sv/sv.sum()
    W = np.outer(phi, phi.conj()); out = []; best = 1
    for k in range(1, (kmax or N)):
        Bk = sv[:k].sum()
        X, Y, C0 = _herm(d)
        C, tr, bounds = build_feasible_set(S, X, Y, lo, hi, eps, method)
        obj = cp.sum(cp.multiply(W.real, X)) + cp.sum(cp.multiply(W.imag, Y)) - Bk*tr
        val, lb, st = certified_solve(cp.Problem(cp.Minimize(obj), C + C0), bounds, 2.0)
        out.append(dict(k=k, Bk=float(Bk), primal=val, lb=lb, status=st))
        if lb is not None and lb > 0: best = k+1
        else: break
    return best, out
