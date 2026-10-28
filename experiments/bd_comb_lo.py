"""Stage BD: why the comb window beats the strip at finite sigma, and why not asymptotically (leading-order derivation + numerics).

A window profile periodic in v with Fourier amplitudes a_m (density 1 + e^{-w^2/2 ell^2} sum_m a_m cos(m k v/sigma) >= 0) has, after the blur, its
m-th harmonic suppressed by e^{-m^2 k^2/2}; for k >~ 1.2 the fundamental dominates every leading-order quantity of stage AS (W, CR, N), so the window
acts as a strip with c -> a_1.  Positivity limits a single cosine to c <= 1 but a comb of thin teeth to a_1 -> 2 (a_1 = 2c e^{-(k st)^2/2}).
So the comb family is, at leading order, the strip family with c in (0, 2].  as_strip_lo.g_lo_log with c <= 2 against c <= 1:
-> results/bd_comb_lo.json.  Python 3.11."""
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


def g_x(x, lsig, cmax):
    lk0, llam, lM, leta = x[:4]; k0 = np.exp(lk0); c = cmax / (1 + np.exp(-x[4]))
    if not (0.3 < k0 < 200 and lM > 0 and lM + llam + lsig / 2 < np.log(2.0) and leta + llam + lsig / 2 > np.log(30.0 / k0) + lsig): return 1e9
    return g_lo_log(lsig, k0, llam, lM, leta, c)


def gmin(lsig, cmax):
    best = None; kg = np.sqrt(-lsig / 4)
    for k0 in (1.0, 2.0, 3.0, 0.7 * kg, kg):
        for leta in (-3.0, -1.5):
            x0 = np.array([np.log(k0), np.log(1.5), np.log(10) + k0 * k0, leta, 4.0])
            r = minimize(g_x, x0, args=(lsig, cmax), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            for _ in range(3): r = minimize(g_x, r.x, args=(lsig, cmax), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            if best is None or r.fun < best.fun: best = r
    return float(best.fun), float(np.exp(best.x[0])), float(cmax / (1 + np.exp(-best.x[4])))


if __name__ == '__main__':
    out = {}
    for e in (6, 12, 24, 48, 96, 256, 1024, 4096, 16384):
        ls = -e * np.log(10); g1, k1, c1 = gmin(ls, 1.0); g2, k2, c2 = gmin(ls, 2.0)
        out['1e-%d' % e] = {'strip_LO': g1, 'comb_LO': g2, 'ratio': g2 / g1, 'strip_g_sqrt_ln': g1 * np.sqrt(-ls), 'comb_g_sqrt_ln': g2 * np.sqrt(-ls), 'k_strip': k1, 'k_comb': k2, 'c_strip': c1, 'c_comb': c2}
        print('sigma 1e-%-5d strip LO %.5f (g sqrt ln %.4f, k0 %.3f)  comb LO (a1 <= 2) %.5f (g sqrt ln %.4f, k0 %.3f, a1 %.3f)  ratio %.4f' % (
            e, g1, g1 * np.sqrt(-ls), k1, g2, g2 * np.sqrt(-ls), k2, c2, g2 / g1), flush=True)
    json.dump(out, open('results/bd_comb_lo.json', 'w'), indent=1)
