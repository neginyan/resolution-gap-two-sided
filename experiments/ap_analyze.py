"""Stage AP summary: confined hidden multipoles.  -> results/summary_AP.json, figures/stageAP.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = json.load(open('results/ap_confined.json')); Dd = json.load(open('results/ap_direct.json'))
S = {'confined': C, 'direct': Dd}
json.dump(S, open('results/summary_AP.json', 'w'), indent=1, default=float)
sig = sorted((float(k) for k in C['table']['H2max']), reverse=True); L = np.log(1 / np.array(sig))
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
a = ax[0]
for tag, c, lab in (('H2max', 'C3', 'lines of length Lw = 4.40 (the (H2) maximum)'), ('gapmin', 'C0', 'lines of the gap-minimising length 2$\\sqrt{\\sigma}$')):
    a.plot(L, [C['table'][tag]['%g' % s]['Lambda_over_ln'] for s in sig], 'o-', color=c, label=lab)
a.axhline(8, color='gray', ls='--', lw=0.8); a.text(9.5, 8.2, "8: Z2-F' (B = 4) through Cramér–Rao, Λ ≤ 8 ln(1/σ) + C", fontsize=7, color='gray')
a.set_xscale('log'); a.set_xlabel('ln(1/σ)'); a.set_ylabel('$\\Lambda_{max}$ / ln(1/σ)  (orders n ≤ 8: lower bounds)'); a.legend(fontsize=7)
a.set_title('(a) under (H1), (H2) hidden multipoles still make Λ/ln(1/σ) grow', fontsize=9)
a = ax[1]
a.plot(L, [C['table']['H2max']['%g' % s]['g_chain'] for s in sig], 'v--', color='C3', label='chain lower bound (√2/π)/√Λ at the Λ-maximiser')
a.plot(L, [C['g_min_family']['%g' % s]['g_true_min'] for s in sig], 'o-', color='k', label='TRUE gap, minimised over the whole family and Lw')
ss = [s for s in sig if C['g_min_family']['%g' % s]['strip_gmin']]
a.plot(np.log(1 / np.array(ss)), [C['g_min_family']['%g' % s]['strip_gmin'] for s in ss], 's-', color='C2', label='strip family g_min (paper E)')
x = np.linspace(np.log(1e24), np.log(1e96), 50); a.plot(x, 1.02 / np.sqrt(x - 23.4), ':', color='C2', lw=0.8)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('ln(1/σ)'); a.set_ylabel('g = gap/σ'); a.legend(fontsize=7)
a.set_title('(b) but the true gap of these states stays ≥ 1, above the strips', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAP.png', dpi=130)
print('figure written')
