import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
C = {'HC': '#9a9a9a', 'GP': '#d55e00', 'AG': '#009e73', 'CS': '#0072b2', 'ref': '#222222'}
R = lambda e: json.load(open('../Supplementary/results_%s.json' % e))['results']
F = '../Figures/'
lb = lambda r: (r or {}).get('lb') if (r or {}).get('lb') is not None else np.nan

r = R('E1')
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
for (a, b), mk in zip([(2.5, 2.75), (3.0, 3.25), (3.5, 3.75)], ['o', 's', '^']):
    rows = [x for x in r['rank'] if x['a'] == a]; Ns = [x['N'] for x in rows]
    l, = ax[0].semilogy(Ns, [max(x['eps'], 1e-6) for x in rows], marker=mk, ms=3.5, label='$c=%.2f$' % (a*b))
    ax[0].semilogy(Ns, [x['q'] for x in rows], marker=mk, ms=3.5, ls=':', color=l.get_color())
    ax[1].plot(Ns, [x['gap'] for x in rows], marker=mk, ms=3.5, color=l.get_color(), label='$c=%.2f$' % (a*b))
    ax[1].plot(Ns, [x['gap_cont'] for x in rows], ls='--', lw=0.8, color=l.get_color())
for k in (0, 1): ax[k].xaxis.set_major_locator(MaxNLocator(integer=True)); ax[k].set_xlabel('retained rank $N$')
ax[0].set_ylabel('one-arm leakage'); ax[0].legend(fontsize=6.5, frameon=False)
ax[0].set_title('(a) bound $\\epsilon_j$ (solid), true $q_j$ (dotted)', fontsize=8)
ax[1].set_ylabel('gap $g$'); ax[1].set_title('(b) discrete (solid), continuum (dashed)', fontsize=8)
al = r['alpha']
ax[2].plot([x['alpha'] for x in al], [x['eps'] for x in al], 'o-', ms=3, color=C['CS'], label='bound')
ax[2].plot([x['alpha'] for x in al], [x['q'] for x in al], 's:', ms=3, color=C['ref'], label='true')
ax[2].set_xlabel('weight $\\alpha$'); ax[2].set_ylabel('one-arm leakage'); ax[2].set_title('(c) weighted deficit, $N=6$', fontsize=8)
ax[2].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(F + 'fig_leakage.pdf'); plt.close(fig)

r = R('E3'); e2 = R('E2'); base = [x for x in e2['rows'] if x['u'] == 1e-4][0]
w = [0.0] + [x['w'] for x in r]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(w, [e2['info']['nu']] + [x['nu'] for x in r], 'k-o', ms=3.5, label=r'reference $\nu(Q\rho Q)$')
ax.plot(w, [lb(base['CS'])] + [lb(x['CS']) for x in r], '-s', ms=3.5, color=C['CS'], label='CS')
ax.plot(w, [lb(base['AGr'])] + [lb(x['AGr']) for x in r], '-D', ms=3.5, color=C['AG'], label='AGr (refined aggregate)')
ax.plot([0] + [x['w'] for x in r if 'GP' in x], [lb(base['GP'])] + [lb(x['GP']) for x in r if 'GP' in x], '^', ms=4, color=C['GP'], label='GP')
ax.set_xscale('symlog', linthresh=0.002); ax.set_xlabel('tail weight $w$'); ax.set_ylabel('validated negativity bound')
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(F + 'fig_contamination.pdf'); plt.close(fig)

r = R('E4'); ns = sorted(set(x['n'] for x in r))
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for key, mk, col, lab in [('CS_CP', 's', C['CS'], 'CS, Clopper--Pearson'), ('CS_H', 'o', '#56b4e9', 'CS, Hoeffding'),
                          ('AGr_CP', 'D', C['AG'], 'AGr, Clopper--Pearson')]:
    m = [np.mean([lb(x[key]) for x in r if x['n'] == n]) for n in ns]
    s = [np.std([lb(x[key]) for x in r if x['n'] == n]) for n in ns]
    ax.errorbar(ns, m, yerr=s, marker=mk, ms=3.5, color=col, capsize=2, label=lab)
hc = [(x['n'], lb(x['HC_H'])) for x in r if 'HC_H' in x]
ax.plot([h[0] for h in hc if not np.isnan(h[1])], [h[1] for h in hc if not np.isnan(h[1])], 'x--', color=C['HC'], label='HC (not valid)')
for h in hc:
    if np.isnan(h[1]): ax.plot(h[0], 0.02, 'x', color=C['HC'], ms=6)
ax.axhline(e2['info']['nu'], color='k', lw=0.8, ls=':', label=r'$\nu(Q\rho Q)$')
ax.set_xscale('log'); ax.set_xlabel('detected pairs per setting $n$'); ax.set_ylabel('validated negativity bound')
ax.legend(fontsize=6.5, frameon=False); fig.tight_layout(); fig.savefig(F + 'fig_counts.pdf'); plt.close(fig)
print('figures done')
