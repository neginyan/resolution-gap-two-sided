"""Stage AS figure: strip gap vs sigma — true g, Theorem-2, two-scale and AR2 bounds; leading-order minimum and g sqrt(ln 1/sigma) -> 1.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
S = json.load(open('results/as_slices.json')); L = json.load(open('results/as_strip_lo.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lv = np.array([float(k.replace('1e-', '')) * np.log(10) for k in L]); gl = np.array([L[k]['g'] for k in L])
ks = ('1e-06', '1e-12', '1e-24'); ls = np.array([-np.log(float(k)) for k in ks])
ax[0].plot(lv, gl, '-', color='0.3', label='leading-order minimum (strip family)')
ax[0].plot(lv, 1 / np.sqrt(lv), ':', color='0.5', label=r'$1/\sqrt{\ln(1/\sigma)}$')
ax[0].plot(ls, [S[k]['g'] for k in ks], 'o', color='C0', ms=7, label='true g (paper E evaluator)')
ax[0].plot(ls, [S[k]['bound_2s'] for k in ks], 'x', color='C2', ms=8, mew=2, label='two-scale Cramér–Rao bound (Thm AS1)')
ax[0].plot(ls, [S[k]['bound_AR2'] for k in ks], 's', color='C3', ms=6, mfc='none', label='AR2 = v-slice bound')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$\ln(1/\sigma)$'); ax[0].set_ylabel(r'$g$ = gap/$\sigma$'); ax[0].legend(fontsize=8)
ax[0].set_title('strips: gap and its lower bounds')
ax[1].plot(lv, [L[k]['g_sqrt_ln'] for k in L], 'o-', color='C0', ms=4, label=r'$g\,\sqrt{\ln(1/\sigma)}$')
ax[1].plot(lv, [L[k]['two_k0_g'] for k in L], 's-', color='C1', ms=4, label=r'$2k_0\,g$')
ax[1].plot(lv, [4 * L[k]['k0sq_over_ln'] for k in L], '^-', color='C2', ms=4, label=r'$4k_0^2/\ln(1/\sigma)$')
ax[1].axhline(1, color='0.5', lw=0.8); ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\ln(1/\sigma)$'); ax[1].legend(fontsize=8)
ax[1].set_title('leading order: all three tend to 1')
fig.tight_layout(); fig.savefig('figures/stageAS.png', dpi=140); print('figure written')
