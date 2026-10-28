"""Stage BF §2 (leading order): where the comb and the strip reach the mixtures' 0.884 — corrected.

Stage BE's value 2.7e-4 for the comb came from bd_comb_lo.gmin, whose start set lies outside the feasible region for sigma >~ 1e-4 (it returns the
sentinel there), so the root was taken from the wrong side.  bf_mix.lo_min uses a wider start set (M from e^{0.5}, k0 from 1).  Proposition BE1 predicts the
comb's crossing at about 256 x the strip's: 256 x 5.6e-6 = 1.4e-3 (the (1 + r) factor shifts it slightly).  -> results/bf_lo_cross.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import brentq
from bf_mix import lo_min

if __name__ == '__main__':
    tab = {}
    for e in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        s = 10 ** -e; tab['%.1f' % e] = {'sigma': s, 'comb_LO': lo_min(s, 2.0)[0], 'strip_LO': lo_min(s, 1.0)[0]}
        print('sigma 10^-%.1f: comb LO %.4f  strip LO %.4f' % (e, tab['%.1f' % e]['comb_LO'], tab['%.1f' % e]['strip_LO']), flush=True)
    f = lambda e, c: lo_min(10 ** -e, c)[0] - 0.884
    ec = brentq(lambda e: f(e, 2.0), 2.5, 3.5, xtol=1e-4); es = brentq(lambda e: f(e, 1.0), 5.0, 6.0, xtol=1e-4)
    out = {'table': tab, 'comb_LO_cross': 10 ** -ec, 'strip_LO_cross': 10 ** -es, 'ratio': 10 ** (es - ec), 'stageBE_value_superseded': 2.664e-4, 'mixtures': 0.884}
    print('LO crossing of 0.884: comb %.3e, strip %.3e, ratio %.0f (Proposition BE1: ~256)' % (out['comb_LO_cross'], out['strip_LO_cross'], out['ratio']))
    json.dump(out, open('results/bf_lo_cross.json', 'w'), indent=1)
