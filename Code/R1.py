"""Recompute the headline values with the corrected constants (d_a = 4 sin theta; variable bounds
|x_a| <= 2 sqrt(1+2 d_a), aggregate slacks <= 1 + d_a + max|p|) and compare with the reported values."""
import json, numpy as np
from experiments import *
out = []
def rec(name, new, old):
    out.append(dict(case=name, new=new, old=old, diff=None if (new is None or old is None) else new - old)); print(out[-1], flush=True)
J = lambda e: json.load(open('../Supplementary/results_%s.json' % e))['results']
S = Setup(G, A, B, N, P); c = mix(0); Pr = S.probs(c); eps, _ = S.eps_from_delta(S.deficits(c))
e2 = [r for r in J('E2')['rows'] if r['u'] == U0][0]
for m in ['GP', 'AGr', 'CS']:
    rec('E2 ' + m, certify_negativity(S, Pr-U0, Pr+U0, eps, m)['lb'], e2[m]['lb'])
ph = S.probs(mix(0)); rng = np.random.default_rng(2026); e4 = J('E4'); m_ = len(ph) + 4
for n in [1e5, 1e6, 1e7, 1e8]:
    n = int(n)
    for rep in range(3):
        cnt = sample(S, ph, n, rng)
        if rep != 0 or n not in (100000, 10000000): continue
        k = np.concatenate(cnt); phat = k/n; lo, hi = clopper_pearson(k, n, 0.05/m_)
        p1 = P + 1; dcp = []
        for arm in (0, 1):
            oc = lambda c_: (c_.reshape(p1, p1)[P, :].sum() if arm == 0 else c_.reshape(p1, p1)[:, P].sum())
            dcp.append(clopper_pearson(np.array([oc(cnt[0])]), n, 0.05/m_)[1][0] + clopper_pearson(np.array([oc(cnt[1])]), n, 0.05/m_)[1][0])
        epsC = S.eps_from_delta(dcp)[0]; u = hoeffding(m_, n, 0.05)
        epsH = S.eps_from_delta([d + 2*u for d in S.deficits_from_probs(phat)])[0]
        old = [x for x in e4 if x['n'] == n and x['rep'] == 0][0]
        rec('E4 n=%d CS_CP' % n, certify_negativity(S, lo, hi, epsC, 'CS')['lb'], old['CS_CP']['lb'])
        if n == 10000000:
            rec('E4 n=%d HC_H' % n, certify_negativity(S, phat-u, phat+u, epsH, 'HC')['lb'], old['HC_H']['lb'])
from continuum import Continuum, CSetup, source_probs
C = Continuum(A, B, N, P); Sc = CSetup(C); Pc, dc = source_probs(A, B, P, SP, SM); epsc = Sc.eps_from_delta(dc)[0]
e11 = J('E11')['continuum'][0]
for m in ['AGr', 'CS', 'CS+G']:
    rec('E11 ' + m, certify_negativity(Sc, Pc-U0, Pc+U0, epsc, m)['lb'], e11[m]['lb'])
json.dump(dict(results=out), open('../Supplementary/results_R1.json', 'w'), indent=1, default=float)
