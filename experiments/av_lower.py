"""Stage AV: the strip family from below — the leading-order minimum of g at FIXED ripple wavenumber k0, g*(k0; sigma).

Used for the step 'a strip with k0^2 > (1/4 + delta) ln(1/sigma) has a large gap' (leading-order derivation): the background term
b Lw^2 of N and the window-dominance condition r = b/W << 1 cannot both be small once e^{4 k0^2} exceeds ~1/sigma.
The other regime (k0^2 <= (1/4 + delta) ln(1/sigma)) is covered by Theorem AT2 with k_eff <= k0 (stage AV, Lemma AV2): g >= (1 - o(1))/(2 k0).
-> results/av_lower.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import minimize
from as_strip_lo import g_lo_log


def gstar(lsig, k0):
    def f(x):
        llam, lM, leta = x[:3]; c = 1 / (1 + np.exp(-x[3]))
        if not (lM > 0 and lM + llam + lsig / 2 < np.log(2.0) and leta + llam + lsig / 2 > np.log(30.0 / k0) + lsig): return 1e9
        return g_lo_log(lsig, k0, llam, lM, leta, c)
    best = None
    for lM0 in (k0 * k0 + np.log(10), k0 * k0, max(1.0, k0 * k0 - 5), k0 * k0 + 5):
        for leta0 in (-3.0, -1.0, -6.0):
            r = minimize(f, np.array([0.3, lM0, leta0, 4.0]), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            for _ in range(2): r = minimize(f, r.x, method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-11, 'fatol': 1e-14, 'adaptive': True})
            if best is None or r.fun < best.fun: best = r
    return float(best.fun)


if __name__ == '__main__':
    out = {}
    for e in (24, 96, 384, 1536):
        L = e * np.log(10); lsig = -L; k4 = np.sqrt(L / 4); rows = []
        for t in (0.4, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0):
            k0 = t * k4; g = gstar(lsig, k0)
            rows.append({'k0_over_sqrt_ln_4': t, 'k0': k0, 'gstar': g, 'two_k0_gstar': 2 * k0 * g, 'gstar_sqrt_ln': g * np.sqrt(L)})
            print('sigma 1e-%-5d k0 = %.2f sqrt(ln/4) = %7.3f: g*(k0) %.5g  2 k0 g* %.4f  g* sqrt(ln) %.4f' % (e, t, k0, g, 2 * k0 * g, g * np.sqrt(L)), flush=True)
        out['1e-%d' % e] = rows
        json.dump(out, open('results/av_lower.json', 'w'), indent=1)
