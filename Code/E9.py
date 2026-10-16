import json, numpy as np
from epr import *
from experiments import G, SP, SM, A, B, N, P, mix, choose_apertures
out = {'contamination': [], 'operating': [], 'pixels': []}
for w in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]:
    r = epr_bound(G, mix(w), A, B, P); r['w'] = w; out['contamination'].append(r)
for sp_, sm_ in [(1.2, .6), (1.5, .6), (2.0, .6), (2.4, .6)]:
    Psi = double_gaussian(G, sp_, sm_); e, a, b = choose_apertures(Psi, 6, 8)
    r = epr_bound(G, [(1.0, Psi)], a, b, 8); r.update(sp=sp_, a=a, b=b, keff=k_eff(sp_, sm_)); out['operating'].append(r)
for p in [6, 8, 10, 16, 24, 32]:
    r = epr_bound(G, mix(0), A, B, p); r['p'] = p; out['pixels'].append(r)
json.dump(dict(results=out), open('../Supplementary/results_E9.json', 'w'), indent=1)
for k, v in out.items():
    for r in v: print(k, r)
