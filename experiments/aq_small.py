"""Stage AQ: the tiny decrease at sigma = 1e-24, beta = 1e-5 — signed multipole, or plain mass added to the strip's v-profile?
Rows with f = +-1 (multipole) and with f = 0 (unmodulated, positive mass only) for beta = 1e-6 ... 1e-4, and the strip itself on a
twice finer grid (the numerical noise level).  -> results/aq_small.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aq_hybrid import strip_params, gap_of, rows_terms, multipole

sig = 1e-24; P = strip_params(sig); kw = 2 * np.pi * 8 / (P['lam'] / np.sqrt(sig))
g0 = gap_of(sig, P)['gap_over_sigma']; g0f = gap_of(sig, P, nper=50, coarse=400, n0=29, nv=64)['gap_over_sigma']
out = {'strip': g0, 'strip_fine': g0f, 'noise': abs(g0f - g0), 'rows': {}}
print('strip g %.9f, finer grid %.9f (diff %.1e)' % (g0, g0f, g0f - g0), flush=True)
xs, a, f = multipole(1, np.pi / P['k0'], 0.6); a = a / a.sum()
for lab, ff in (('signed', f), ('unsigned_f0', 0 * f)):
    for beta in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        g = gap_of(sig, P, rows_terms(sig, P, P['k0'], xs, a, ff, beta, kw), kw=kw)['gap_over_sigma']
        out['rows']['%s_%g' % (lab, beta)] = {'beta': beta, 'g': g, 'dg': g - g0}
        print('  %-12s beta %.0e: dg = %+.3e' % (lab, beta, g - g0), flush=True)
json.dump(out, open('results/aq_small.json', 'w'), indent=1)
