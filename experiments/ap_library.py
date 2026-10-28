"""Stage AP: library of exact v-problems for hidden multipoles between two w-lines (the part that does not depend on sigma or the line length).

For order n, depth D, source spacing d and hiddenness z (each source term e^{-z} below the lines at the zero), the sources of
an_multipole.realise (optimal leading-order polynomial of degree n, tilt alpha) are evaluated exactly (an_signed1d.quantities):
  Lam_v = E[(s g + 2 g')^2]/E[g^2],  lnMI = ln(M/I_g),  Jvv = E[s^2] (t = 1, per unit mass),  Rq = E[g'^2]/E[g^2].
With rows of profile (1 + f cos(k w/sigma)), k = 1, the multipole adds t J_mp = (k^2 e^{-k^2}/2) e^{-lnMI} to sigma^2 J_ww (exactly one
harmonic: the Gaussian blur of cos is exact).  usage: python ap_library.py N [small] -> results/ap_library_<N>[_s].json
('small': the extra depths D = 2, 3, 5, 7 needed at large sigma).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
from an_signed1d import quantities
from an_multipole import realise

n = sys.argv[1]; small = len(sys.argv) > 2 and sys.argv[2] == 'small'
M = json.load(open('results/an_multipole.json')); L = M['leading'][n]; p = np.array(L['p'])
rows = []
for D in ((2.0, 3.0, 5.0, 7.0) if small else np.arange(4.0, 50.0, 2.0)):
    for d in (0.5, 1.0, 2.0, 3.0):
        if d * int(n) > 0.8 * D and not (small and d == 0.5): continue
        for z in ((0.5, 1.0, 2.0, 3.0, 6.0) if small else (3.0, 6.0, 10.0)):
            xs, lw, f = realise(D, L['alpha'], p, d, z); lw = lw - lw.max()
            if not np.all(np.isfinite(lw)): continue
            o = quantities(xs, lw, f)
            if not (np.isfinite(o['Lam_v']) and np.isfinite(o['lnMI'])): continue
            rows.append({'n': int(n), 'D': float(D), 'd': d, 'z': z, 'Lam_v': o['Lam_v'], 'lnMI': o['lnMI'], 'Jvv': o['Jvv'], 'Rq': o['Rq']})
    print('n %s D %g: %d rows, best Lam_v/lnMI so far %.3f' % (n, D, len(rows), max((r['Lam_v'] / r['lnMI'] for r in rows), default=0)), flush=True)
json.dump(rows, open('results/ap_library_%s%s.json' % (n, '_s' if small else ''), 'w'), indent=1)
