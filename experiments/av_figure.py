"""Stage AV figure: the strip family from above (explicit sequence: true g and leading order; the optimum of stage AS) and the threshold of
the fixed-k0 minimum g*(k0).  -> figures/stageAV.png.  Python 3.11."""
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
A = json.load(open('results/av_strip.json'))['sequence']; Q = json.load(open('results/av_seq_lo.json')); O = json.load(open('results/as_strip_lo.json')); L = json.load(open('results/av_lower.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
lq = [float(k.replace('1e-', '')) * np.log(10) for k in Q]; ax[0].plot(lq, [Q[k]['g_LO_sqrt_ln'] for k in Q], '-', color='C0', label='explicit sequence, leading order')
la = [np.log(1 / float(k)) for k in A]
ax[0].plot(la, [A[k]['g_sqrt_ln'] for k in A], 'o', color='C0', ms=7, label='explicit sequence, true g (paper E evaluator)')
ax[0].plot(la, [A[k]['bound_AT2'] * np.sqrt(np.log(1 / float(k))) for k in A], 's', mfc='none', color='C3', label='Theorem AT2 on the same states')
lo = [float(k.replace('1e-', '')) * np.log(10) for k in O]; ax[0].plot(lo, [O[k]['g_sqrt_ln'] for k in O], '--', color='0.4', label='family minimum (leading order, stage AS)')
ax[0].axhline(1, color='0.6', lw=0.8); ax[0].set_xscale('log'); ax[0].set_xlabel(r'$\ln(1/\sigma)$'); ax[0].set_ylabel(r'$g\,\sqrt{\ln(1/\sigma)}$'); ax[0].legend(fontsize=7)
ax[0].set_title('strips from above: g sqrt(ln 1/sigma) -> 1')
for i, (k, rows) in enumerate(L.items()):
    t = [r['k0_over_sqrt_ln_4'] for r in rows if r['gstar'] < 1e8 and np.isfinite(r['gstar'])]; y = [r['two_k0_gstar'] for r in rows if r['gstar'] < 1e8 and np.isfinite(r['gstar'])]
    ax[1].plot(t, y, 'o-', ms=4, color='C%d' % i, label='sigma = %s' % k)
ax[1].axvline(1, color='0.6', lw=0.8); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$k_0 / \sqrt{\ln(1/\sigma)/4}$'); ax[1].set_ylabel(r'$2k_0\, g^*(k_0)$ (leading order)')
ax[1].set_title('fixed ripple k0: g* = 1/(2k0) below the threshold'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig('figures/stageAV.png', dpi=140); print('figure written')
