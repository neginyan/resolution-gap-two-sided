"""Stage AS: leading-order (two-scale) formula for the strip's gap and its minimisation over the strip parameters (derivation + numerics).

Strip: rho = N(v;0,Lv^2) N(w;0,Lw^2)(1 + c e^{-w^2/2 ell^2} cos(k0 v/sigma)), ell = lam sqrt(sigma), Lw = M ell.  Leading order in 1/M, ell/Lw,
e^{-k0^2/2} (blurred ripple contrast), Lv/ell:
  sigma^2 J_vv      = W + b,               W = c^2 k0^2 e^{-k0^2}/(2 sqrt2 M)  (window),   b = (sigma/Lv)^2  (background)
  sigma^2 J_ww      = sigma^2 CR/Lw^2,     CR = 1 + (sqrt2/8) c^2 e^{-k0^2} M   (window w-information = the two-scale Cramer-Rao value)
  sigma^2 E[s_v^2 phi] = [W (ell^2/2 + Lv^2) + b Lw^2]/4      (phi = 1 - cos m_z ~ m_z^2/2: small in the window, ~Lw^2/4 elsewhere)
  g = (2 sigma^2 J_ww + N)/(sigma sigma^2 J_vv),  N = sigma^2 E[s_v^2 phi]/2.
Checks against the true evaluator are in as_slices.py / run_all_AS.py.  -> results/as_strip_lo.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from scipy.optimize import minimize


def g_lo(sig, k0, lam, M, Lv, c):
    """direct form (for checks at moderate sigma)."""
    ell = lam * np.sqrt(sig); Lw = M * ell; e = np.exp(-k0 * k0)
    W = c * c * k0 * k0 * e / (2 * np.sqrt(2) * M); b = (sig / Lv) ** 2; CR = 1 + np.sqrt(2) / 8 * c * c * e * M
    A = 2 * sig * sig * CR / Lw ** 2; N = (W * (ell ** 2 / 2 + Lv ** 2) + b * Lw ** 2) / 8
    return (A + N) / (sig * (W + b))


def g_lo_log(lsig, k0, llam, lM, leta, c):
    """the same formula divided through by sigma W (no under/overflow for any sigma): with r = b/W,
    g = [2 CR/(M^2 lam^2 W) + lam^2 (1/2 + eta^2 + r M^2)/8] / (1 + r),  Lv = eta ell."""
    lW = 2 * np.log(c) + 2 * np.log(k0) - k0 * k0 - np.log(2 * np.sqrt(2)) - lM
    lr = lsig - 2 * leta - 2 * llam - lW
    lCR = np.logaddexp(0.0, np.log(np.sqrt(2) / 8) + 2 * np.log(c) - k0 * k0 + lM)
    t1 = 2 * np.exp(lCR - 2 * lM - 2 * llam - lW)
    t2 = np.exp(2 * llam) * (0.5 + np.exp(2 * leta) + np.exp(lr + 2 * lM)) / 8
    return (t1 + t2) / (1 + np.exp(lr))


def g_lo_x(x, lsig):
    lk0, llam, lM, leta = x[:4]; k0 = np.exp(lk0); c = 1 / (1 + np.exp(-x[4]))
    if not (0.3 < k0 < 200 and lM > 0 and lM + llam + lsig / 2 < np.log(2.0) and leta + llam + lsig / 2 > np.log(30.0 / k0) + lsig): return 1e9
    return g_lo_log(lsig, k0, llam, lM, leta, c)


def gmin_lo(lsig):
    best = None; kg = np.sqrt(-lsig / 4)
    for k0 in (1.0, 2.0, 3.0, 0.7 * kg, kg):
        for leta in (-3.0, -1.5):
            x0 = np.array([np.log(k0), np.log(1.5), np.log(10) + k0 * k0, leta, 4.0])
            r = minimize(g_lo_x, x0, args=(lsig,), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            for _ in range(3): r = minimize(g_lo_x, r.x, args=(lsig,), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            if best is None or r.fun < best.fun: best = r
    k0, lam, eta = np.exp(best.x[[0, 1, 3]]); c = 1 / (1 + np.exp(-best.x[4]))
    return {'g': float(best.fun), 'k0': float(k0), 'lam': float(lam), 'lnM': float(best.x[2]), 'Lv_over_ell': float(eta), 'c': float(c),
            'g_sqrt_ln': float(best.fun * np.sqrt(-lsig)), 'two_k0_g': float(2 * k0 * best.fun), 'k0sq_over_ln': float(k0 * k0 / -lsig)}


if __name__ == '__main__':
    out = {}
    for e in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 4096, 16384):
        r = gmin_lo(-e * np.log(10)); out['1e-%d' % e] = r
        print('sigma 1e-%-5d g_LO %.5f  g sqrt(ln 1/sigma) %.4f  2 k0 g %.4f  k0 %.3f  k0^2/ln %.4f  ln M/k0^2 %.3f  Lv/ell %.3g c %.4f' % (
            e, r['g'], r['g_sqrt_ln'], r['two_k0_g'], r['k0'], r['k0sq_over_ln'], r['lnM'] / r['k0'] ** 2, r['Lv_over_ell'], r['c']), flush=True)
    json.dump(out, open('results/as_strip_lo.json', 'w'), indent=1)
