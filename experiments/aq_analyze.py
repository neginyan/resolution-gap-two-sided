"""Stage AQ summary: strip x multipole hybrids.  Adds the strip's own decomposition (Num = 2 sigma^2 J_ww + N, Den = sigma^2 (J_vv + J_ww))
and, for every hybrid, the marginal ratio dNum/dDen compared with sigma g_strip (Proposition AQ1: g decreases only if dNum < sigma g dDen).
-> results/summary_AQ.json, figures/stageAQ.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from aq_hybrid import strip_params, gap_of

HA = {}
for fn in sorted(glob.glob('results/aq_hA_*.json')):
    s = float(fn.split('aq_hA_')[1][:-5]); HA[s] = json.load(open(fn))
TS = json.load(open('results/aq_twoscale.json'))
S = {'H_A': {}, 'H_B': TS}
for s, d in sorted(HA.items(), reverse=True):
    P = strip_params(s); o = gap_of(s, P)
    num0 = 2 * s * s * o['J_ww'] + o['N_ibp']; den0 = s * s * (o['J_vv'] + o['J_ww'])
    row = {'strip': {'g': o['gap_over_sigma'], 'num': num0, 'den': den0, 'sig2Jww': s * s * o['J_ww'], 'N': o['N_ibp'], 'sig2Jvv': s * s * o['J_vv']}, 'hybrids': {}}
    for k, v in d.items():
        if k == 'strip': continue
        num = 2 * v['sig2Jww'] + v['N']; den = v['sig2Jvv'] + v['sig2Jww']
        dn, dd = num - num0, den - den0
        v2 = dict(v); v2['dNum'] = dn; v2['dDen'] = dd; v2['sigma_g_strip'] = s * o['gap_over_sigma']
        v2['marginal_ratio'] = dn / dd if dd != 0 else None; v2['raises_g'] = v['g'] > o['gap_over_sigma']
        row['hybrids'][k] = v2
    row['min_hybrid_g'] = min(v['g'] for v in row['hybrids'].values()) if row['hybrids'] else None
    S['H_A']['%g' % s] = row
    print('sigma %g: strip g %.5f (Num %.3e = 2 s2Jww %.3e + N %.3e; Den %.3e) | hybrids: min g %.5f, all raise g: %s' % (
        s, row['strip']['g'], num0, 2 * row['strip']['sig2Jww'], row['strip']['N'], den0, row['min_hybrid_g'] or np.nan, all(v['raises_g'] for v in row['hybrids'].values())))
    for k, v in row['hybrids'].items():
        print('    %-14s g %.5f  dNum %+.2e dDen %+.2e  ratio %s vs sigma g %.1e' % (k, v['g'], v['dNum'], v['dDen'], '%.2e' % v['marginal_ratio'] if v['marginal_ratio'] is not None else '-', v['sigma_g_strip']))
json.dump(S, open('results/summary_AQ.json', 'w'), indent=1, default=float)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2)); a = ax[0]
for s, c in zip(sorted(HA, reverse=True), ('C0', 'C1', 'C2', 'C3')):
    r = S['H_A']['%g' % s]
    for n, mk in ((1, 'o-'), (2, 's--')):
        bs = sorted(v['beta'] for v in r['hybrids'].values() if v['n'] == n)
        if bs: a.plot(bs, [[v['g'] for v in r['hybrids'].values() if v['n'] == n and v['beta'] == b][0] / r['strip']['g'] for b in bs], mk, color=c, label='σ = %g, n = %d' % (s, n))
a.axhline(1, color='k', lw=0.6); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('relative mass β of the multipole rows'); a.set_ylabel('g(hybrid) / g(strip optimum)')
a.legend(fontsize=6.5); a.set_title('(a) H-A: signed rows in the strip raise the true gap', fontsize=9)
a = ax[1]; ss = sorted((float(k) for k in TS['min']), reverse=True)
a.plot(np.log(1 / np.array(ss)), [TS['min']['%g' % s]['g'] for s in ss], 'o-', color='k', label='H-B: multipole rows, any k (two-scale), min g')
strip = {1e-4: 1.149, 1e-6: 0.740, 1e-12: 0.335, 1e-24: 0.179, 1e-48: 0.110, 1e-96: 0.073}
a.plot(np.log(1 / np.array(list(strip))), list(strip.values()), 's-', color='C2', label='strip g_min (paper E)')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('ln(1/σ)'); a.set_ylabel('g = gap/σ'); a.legend(fontsize=7)
a.set_title('(b) H-B: the multipole family stays at g ≥ 1', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAQ.png', dpi=130); print('figure written')
