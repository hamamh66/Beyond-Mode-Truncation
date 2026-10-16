"""Numerical experiments. Usage: python experiments.py <name>; results -> ../Supplementary/results_<name>.json"""
import sys, json, time, numpy as np
from cert import *
G = Grid(256, 24.0); SP, SM = 1.5, 0.6; A, B, N, P = 3.0, 3.25, 6, 8; U0 = 1e-4
OUT = '../Supplementary/results_%s.json'

def base(): return double_gaussian(G, SP, SM)
def tail(): return hg_mode(G, 8, 1.0)
def mix(w):
    Psi = base(); t = tail()
    return [(1-w, Psi)] + ([(w, np.outer(t, t))] if w > 0 else [])

def run(S, P_, u, eps, methods, c_extra=0.0):
    row = {}
    for m in methods:
        t0 = time.time(); r = certify_negativity(S, P_-u, P_+u, eps, m, c_extra)
        r['time'] = time.time()-t0; row[m] = r
    return row

def E1():
    """leakage bound vs truth (rank, aperture), mesh, weighted deficit, continuum gaps"""
    Psi = base(); res = {'rank': [], 'mesh': [], 'alpha': []}
    for (a, b) in [(2.5, 2.75), (3.0, 3.25), (3.5, 3.75)]:
        lam, lerr = prolate_eigs(a*b, nmax=14)
        for n in range(3, 13):
            S = deficit_basis(G, a, b, P, n); V = S['V']
            r = Psi @ Psi.conj().T; d = float(np.real(np.trace(r @ S['H'])))
            q = 1 - float(np.real(np.trace(V.T @ r @ V)))
            e = (d - S['ell'])/(S['gap'] - S['ell']) + 2*S['sin_theta']
            res['rank'].append(dict(a=a, b=b, c=a*b, N=n, delta=d, ell=S['ell'], gap=S['gap'], q=q,
                                    eps=min(1, e), gap_cont=float(1-np.sqrt(lam[n])), sin_theta=S['sin_theta']))
    for nx, L in [(128, 24.0), (256, 24.0), (512, 24.0), (1024, 24.0), (512, 48.0)]:
        g = Grid(nx, L); S = deficit_basis(g, A, B, P, N); Ps = double_gaussian(g, SP, SM); rr = Ps @ Ps.T
        res['mesh'].append(dict(nx=nx, L=L, ell=S['ell'], gap=S['gap'], delta=float(np.trace(rr @ S['H'])),
                                q=1-float(np.trace(S['V'].T @ rr @ S['V'])), sin_theta=S['sin_theta']))
    lam, lerr = prolate_eigs(A*B, nmax=14)
    res['continuum'] = dict(c=A*B, lam=[float(x) for x in lam], quad_err=lerr, gap_N6=float(1-np.sqrt(lam[6])))
    for al in np.linspace(0.2, 0.8, 13):
        S = deficit_basis(G, A, B, P, N, al); r = Psi @ Psi.T
        d = float(np.trace(r @ S['H'])); q = 1 - float(np.trace(S['V'].T @ r @ S['V']))
        res['alpha'].append(dict(alpha=float(al), eps=min(1, (d-S['ell'])/(S['gap']-S['ell'])), q=q, gap=S['gap']))
    return res

def E2():
    """exact probabilities: all programs + model-tolerance sensitivity"""
    S = Setup(G, A, B, N, P); c = mix(0); Pr = S.probs(c); T = S.tau(c)
    eps, _ = S.eps_from_delta(S.deficits(c))
    info = dict(eps=eps, q=1-float(np.trace(T).real), nu=float(nu(T, N)), Nfull=float(negativity_pure(base())),
                keff=float(keff_numeric(base())), gap=S.gap, ell=S.ell, sin_theta=S.B['sin_theta'],
                maxdev=float(np.max(np.abs(Pr - (S.Amat @ T.flatten(order='F')).real))))
    rows = []
    for u in [1e-6, 1e-5, 1e-4, 1e-3]:
        meth = ['HC', 'GP', 'AGr', 'CS'] if u == U0 else ['AGr', 'CS']
        r = run(S, Pr, u, eps, meth); r['u'] = u; rows.append(r); print(r, flush=True)
    return dict(info=info, rows=rows)

def E3():
    """separable high-order tail contamination"""
    S = Setup(G, A, B, N, P); rows = []
    for w in [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]:
        c = mix(w); Pr = S.probs(c); T = S.tau(c); eps, _ = S.eps_from_delta(S.deficits(c))
        meth = ['AGr', 'CS'] + (['GP', 'HC'] if w in (0.0025, 0.01) else [])
        r = run(S, Pr, U0, eps, meth); r.update(w=w, eps=eps, nu=float(nu(T, N)), q=1-float(np.trace(T).real))
        rows.append(r); print(r, flush=True)
    return rows

def sample(S, ph, n, rng):
    ms = S.m_set
    return [rng.multinomial(n, ph[:ms]/ph[:ms].sum()), rng.multinomial(n, ph[ms:]/ph[ms:].sum())]

def E4():
    """finite counts: Hoeffding vs Clopper-Pearson (Bonferroni over all events)"""
    S = Setup(G, A, B, N, P); ph = S.probs(mix(0)); rng = np.random.default_rng(2026); rows = []
    m = len(ph) + 4; alpha = 0.05
    for n in [1e5, 1e6, 1e7, 1e8]:
        n = int(n)
        for rep in range(3):
            cnt = sample(S, ph, n, rng); k = np.concatenate(cnt); phat = k/n
            row = dict(n=n, rep=rep)
            # Hoeffding
            u = hoeffding(m, n, alpha)
            ds = [d + 2*u for d in S.deficits_from_probs(phat)]
            epsH = S.eps_from_delta(ds)[0]
            # Clopper-Pearson: per-outcome and the four outside-window marginal events
            lo, hi = clopper_pearson(k, n, alpha/m)
            p1 = P + 1
            def outcnt(c_, arm):
                M = c_.reshape(p1, p1); return M[P, :].sum() if arm == 0 else M[:, P].sum()
            dcp = []
            for arm in (0, 1):
                hx = clopper_pearson(np.array([outcnt(cnt[0], arm)]), n, alpha/m)[1][0]
                hk = clopper_pearson(np.array([outcnt(cnt[1], arm)]), n, alpha/m)[1][0]
                dcp.append(hx + hk)
            epsC = S.eps_from_delta(dcp)[0]
            row.update(u=u, epsH=epsH, epsCP=epsC)
            r = certify_negativity(S, phat-u, phat+u, epsH, 'CS'); row['CS_H'] = r
            r = certify_negativity(S, lo, hi, epsC, 'CS'); row['CS_CP'] = r
            r = certify_negativity(S, lo, hi, epsC, 'AGr'); row['AGr_CP'] = r
            if rep == 0:
                row['HC_H'] = certify_negativity(S, phat-u, phat+u, epsH, 'HC')
            print(row, flush=True); rows.append(row)
    return rows

def E5():
    """separable consistency checks"""
    S = Setup(G, A, B, N, P); cases = []
    v0 = S.V[:, 0]; t = tail()
    for beta in [0.05, 0.1, 0.2]:
        f = np.sqrt(1-beta)*v0 + np.sqrt(beta)*t; f = f/np.linalg.norm(f)
        cases.append(('coherent product, beta=%.2f' % beta, [(1.0, np.outer(f, f))]))
    comps = []
    for x0 in np.linspace(-2.5, 2.5, 11):
        f = hg_mode(G, 0, SM/np.sqrt(2), x0=x0); comps.append((0.5/11, np.outer(f, f)))
    for k0 in np.linspace(-2.5, 2.5, 11):
        f1 = hg_mode(G, 0, 1.5, k0=k0); f2 = hg_mode(G, 0, 1.5, k0=-k0); comps.append((0.5/11, np.outer(f1, f2)))
    cases.append(('position/momentum-correlated mixture', comps))
    rows = []
    for name, c in cases:
        Pr = S.probs(c); T = S.tau(c); eps, _ = S.eps_from_delta(S.deficits(c))
        r = run(S, Pr, U0, eps, ['HC', 'GP', 'AGr', 'CS']); r.update(state=name, eps=eps, nu=float(nu(T, N)),
                                                                   q=1-float(np.trace(T).real))
        rows.append(r); print(r, flush=True)
    return rows

def E6():
    """Schmidt-number certificate"""
    S = Setup(G, A, B, N, P); phi = (S.V.T @ base() @ S.V).flatten(); phi = phi/np.linalg.norm(phi); rows = []
    for w in [0.0, 0.0025, 0.01, 0.05]:
        c = mix(w); Pr = S.probs(c); eps, _ = S.eps_from_delta(S.deficits(c)); row = dict(w=w, eps=eps)
        for m in ['GP', 'AGr', 'CS']:
            k, out = certify_schmidt(S, Pr-U0, Pr+U0, eps, m, phi, kmax=4); row[m] = dict(k=k, steps=out)
        rows.append(row); print(row, flush=True)
    return rows

def E7():
    """calibration errors: pixel crosstalk and pixel-efficiency non-uniformity"""
    S = Setup(G, A, B, N, P); c = mix(0); rows = []
    eps0, _ = S.eps_from_delta(S.deficits(c)); rng = np.random.default_rng(7)
    for kap in [1e-4, 1e-3, 3e-3]:
        Pr = S.probs(c, crosstalk=kap)
        r = run(S, Pr, U0, eps0, ['AGr', 'CS'], c_extra=4*kap); r.update(model='crosstalk', level=kap, c_a=4*kap)
        rows.append(r); print(r, flush=True)
    for dl in [1e-3, 1e-2, 3e-2]:
        Pr0 = S.probs(c); Pr = Pr0*(1 + dl*rng.uniform(-1, 1, Pr0.size))
        ds = [x*(1+dl) for x in S.deficits_from_probs(Pr)]; eps = S.eps_from_delta(ds)[0]
        r = run(S, Pr, U0, eps, ['AGr', 'CS'], c_extra=dl*(Pr+U0)/(1-dl)); r.update(model='efficiency', level=dl, eps=eps)
        rows.append(r); print(r, flush=True)
    return rows

def choose_apertures(Psi, n, p):
    best = None
    for a in np.arange(2.0, 4.51, 0.25):
        for b in np.arange(2.0, 4.51, 0.25):
            S = deficit_basis(G, a, b, p, n); r = Psi @ Psi.T
            d = float(np.trace(r @ S['H'])); e = 2*(d - S['ell'])/(S['gap'] - S['ell'])
            if best is None or e < best[0]: best = (e, float(a), float(b))
    return best

def E8(cfgs=None):
    """operating region: source entanglement, retained rank, pixel number (one factor at a time)"""
    cfgs = cfgs or [(1.2, 0.6, 6, 8), (1.5, 0.6, 6, 8), (2.0, 0.6, 6, 8), (2.4, 0.6, 6, 8),
            (1.5, 0.6, 5, 8), (1.5, 0.6, 7, 8), (1.5, 0.6, 6, 6), (1.5, 0.6, 6, 10)]
    rows = []
    for sp_, sm_, n, p in cfgs:
        Psi = double_gaussian(G, sp_, sm_); e0, a, b = choose_apertures(Psi, n, p)
        S = Setup(G, a, b, n, p); c = [(1.0, Psi)]; Pr = S.probs(c); T = S.tau(c)
        eps, _ = S.eps_from_delta(S.deficits(c))
        r = run(S, Pr, U0, eps, ['AGr', 'CS'])
        r.update(sp=sp_, sm=sm_, N=n, p=p, a=a, b=b, keff=float(keff_numeric(Psi)), eps=eps,
                 nu=float(nu(T, n)), Nfull=float(negativity_pure(Psi)), d=n*n)
        rows.append(r); print({k: v for k, v in r.items()}, flush=True)
    return rows

def E8b():
    return E8([(1.5, 0.6, 7, 8), (1.5, 0.6, 6, 6), (1.5, 0.6, 6, 10)])

def E10():
    """relaxation ablation: AG, refined AG, CS, CS+AG, CS+G, CS+AG+G on fixed data"""
    S = Setup(G, A, B, N, P); Psi = base(); t = tail(); T2 = np.outer(t, t); rows = []
    cases = [('Gaussian, exact', [(1.0, Psi)]), ('tail mixture w=0.005', mix(0.005))]
    for ph in [0.0, np.pi/2, np.pi]:
        v = np.sqrt(1-0.005)*Psi + np.sqrt(0.005)*np.exp(1j*ph)*T2; v = v/np.linalg.norm(v)
        cases.append(('coherent tail w=0.005, phase=%.2f' % ph, [(1.0, v)]))
    for name, c in cases:
        Pr = S.probs(c); T = S.tau(c); eps, _ = S.eps_from_delta(S.deficits(c))
        r = run(S, Pr, U0, eps, ['AG', 'AGr', 'CS', 'CS+AG', 'CS+G', 'CS+AG+G'])
        r.update(case=name, eps=eps, nu=float(nu(T, N)), q=1-float(np.trace(T).real)); rows.append(r); print(r, flush=True)
    return rows

def E11():
    """continuum certificate staircase and error budget"""
    from continuum import Continuum, CSetup, source_probs, reference_tau
    out = {}
    Sd = Setup(G, A, B, N, P); cd = mix(0); Pd = Sd.probs(cd); epsd, _ = Sd.eps_from_delta(Sd.deficits(cd))
    out['discrete'] = dict(eps=epsd, gap=Sd.gap, CS=certify_negativity(Sd, Pd-U0, Pd+U0, epsd, 'CS'))
    C = Continuum(A, B, N, P); Sc = CSetup(C)
    epsg = min(1.0, epsd*(Sd.gap/C.g_lo))
    out['gap_only'] = dict(eps=epsg, gap=C.g_lo, CS=certify_negativity(Sd, Pd-U0, Pd+U0, epsg, 'CS'))
    rows = []
    for w in [0.0, 0.0025]:
        Pc, dc = source_probs(A, B, P, SP, SM, w_tail=w); eps = Sc.eps_from_delta(dc)[0]
        T = reference_tau(C, SP, SM, w_tail=w)
        row = dict(w=w, eps=eps, deficit=dc[0], nu=float(nu(T, N)), q=1-float(np.trace(T).real),
                   maxdev=float(np.max(np.abs(Pc - (Sc.Amat @ T.flatten(order='F')).real))))
        for m in ['AGr', 'CS', 'CS+G']:
            row[m] = certify_negativity(Sc, Pc-U0, Pc+U0, eps, m)
        rows.append(row); print(row, flush=True)
    out['continuum'] = rows
    out['budget'] = dict(lam=[float(x) for x in C.lam], resid=float(C.r.max()), trace_rem=float(C.rem),
                         lamN_up=float(C.lamN_up), g_lo=float(C.g_lo), cluster_gap=C.gapc, sinT=C.sinT, Gdev=C.Gdev,
                         d_parts=C.d_parts, d1=C.d1, d_joint=Sc.d_a, fp_margin=1e-12)
    return out

def E12():
    """source generality: non-Gaussian and mixed sources, CS vs refined AG vs entropic EPR"""
    from epr import epr_bound
    X1, X2 = np.meshgrid(G.x, G.x, indexing='ij'); rows = []
    ng = (X1 - X2)*double_gaussian(G, 1.5, 0.8); ng = ng/np.linalg.norm(ng)
    p0 = double_gaussian(G, 1.5, 0.6); ph = np.exp(1j*1.5*(X1 - X2)); p1 = p0*ph
    srcs = [('non-Gaussian HG1 relative mode', [(1.0, ng)]),
            ('mixed: two momentum-offset copies', [(0.5, p0), (0.5, p1)])]
    for name, c in srcs:
        rr = sum(w*(M @ M.conj().T) for w, M in c)
        best = None
        for a in np.arange(2.0, 4.51, 0.25):
            for b in np.arange(2.0, 4.51, 0.25):
                Sb = deficit_basis(G, a, b, P, N); d = float(np.real(np.trace(rr @ Sb['H'])))
                e = 2*(d - Sb['ell'])/(Sb['gap'] - Sb['ell'])
                if best is None or e < best[0]: best = (e, float(a), float(b))
        _, a, b = best; S = Setup(G, a, b, N, P); Pr = S.probs(c); T = S.tau(c); eps, _ = S.eps_from_delta(S.deficits(c))
        r = run(S, Pr, U0, eps, ['AGr', 'CS', 'CS+G'])
        r.update(source=name, a=a, b=b, eps=eps, nu=float(nu(T, N)), q=1-float(np.trace(T).real),
                 epr=epr_bound(G, c, a, b, P)); rows.append(r); print(r, flush=True)
    return rows

def E13():
    """Schmidt number above two for the most entangled operating-region source"""
    Psi = double_gaussian(G, 2.4, 0.6); a, b = 3.75, 2.5
    S = Setup(G, a, b, N, P); c = [(1.0, Psi)]; Pr = S.probs(c); eps, _ = S.eps_from_delta(S.deficits(c))
    phi = (S.V.T @ Psi @ S.V).flatten(); phi = phi/np.linalg.norm(phi); row = dict(eps=eps)
    for m in ['CS', 'CS+G']:
        k, out = certify_schmidt(S, Pr-U0, Pr+U0, eps, m, phi, kmax=4); row[m] = dict(k=k, steps=out)
    print(row, flush=True); return [row]

if __name__ == '__main__':
    e = sys.argv[1]; t0 = time.time(); r = globals()[e]()
    json.dump(dict(results=r, runtime=time.time()-t0,
                   config=dict(nx=256, L=24.0, sp=SP, sm=SM, a=A, b=B, N=N, pixels=P, u0=U0)),
              open(OUT % e, 'w'), indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print('done', e, time.time()-t0)
