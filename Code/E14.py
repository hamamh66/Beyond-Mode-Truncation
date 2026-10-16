"""re-evaluate the hard-cutoff program for the contamination and separable cases with the corrected solver logic"""
import json, numpy as np
from experiments import *
S = Setup(G, A, B, N, P); out = []
cases = [('w=0.0025', mix(0.0025)), ('w=0.01', mix(0.01))]
v0 = S.V[:, 0]; t = tail()
for beta in [0.05, 0.1, 0.2]:
    f = np.sqrt(1-beta)*v0 + np.sqrt(beta)*t; f = f/np.linalg.norm(f); cases.append(('coherent product %.2f' % beta, [(1.0, np.outer(f, f))]))
comps = []
for x0 in np.linspace(-2.5, 2.5, 11):
    f = hg_mode(G, 0, SM/np.sqrt(2), x0=x0); comps.append((0.5/11, np.outer(f, f)))
for k0 in np.linspace(-2.5, 2.5, 11):
    f1 = hg_mode(G, 0, 1.5, k0=k0); f2 = hg_mode(G, 0, 1.5, k0=-k0); comps.append((0.5/11, np.outer(f1, f2)))
cases.append(('mixture', comps))
for name, c in cases:
    Pr = S.probs(c); eps, _ = S.eps_from_delta(S.deficits(c))
    r = certify_negativity(S, Pr-U0, Pr+U0, eps, 'HC'); r['case'] = name; out.append(r); print(r, flush=True)
json.dump(dict(results=out), open('../Supplementary/results_E14.json', 'w'), indent=1, default=float)
