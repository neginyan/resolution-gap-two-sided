"""Stage AI: independent semi-analytic evaluation of the w-line + point family (t = 1), sharing no code with ai_mix.
Closed form: pi = expit(l), l = -L + D y_v - D^2/2 - y_w^2/2 - 1/2 ln 2pi;  s_w = -pi y_w;  s_v = pi D - y_v;
H_vv = 1 - pi(1-pi) D^2,  H_vw = pi(1-pi) D y_w,  H_ww = pi (1 - (1-pi) y_w^2);  p = a_L (1 + e^l), a_L = e^{-y_v^2/2}/sqrt(2 pi).
Integration: y_w by Gauss-Hermite-free uniform grid on [-W, W]; y_v by a nonuniform grid resolving the valley (width 1/D) and the point tail
(width 1/max(z,1)), streamed in y_w chunks; all sums in the log domain.  Returns tR, Lambda, ln(1/tJ) (normalised with M_lb = 1 + e^{-L}).
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp, log_expit

LN2PI = np.log(2 * np.pi)

def yv_grid(D, z, Wmax):
    yc = D + z                                              # valley at y_w = 0; it bends to D + z + y_w^2/(2D)
    hi = yc + Wmax ** 2 / (2 * D) + 14.0
    lo = min(-9.0, D - 9.0)                                 # the line tail y_v ~ 2D carries J when the valley is far
    fine = np.arange(yc - 60.0 / D, hi, min(0.004, 0.08 / D, 0.08 / max(abs(z), 1.0)))   # covers the bent valley and the tail
    coarse = np.arange(lo, hi, 0.004)
    return np.unique(np.concatenate([coarse, fine]))

def evaluate(D, z, Wmax=12.0, nw=1201):
    L = D * (z + D / 2); yv = yv_grid(D, z, Wmax); dv = np.gradient(yv)
    yw_all = np.linspace(-Wmax, Wmax, nw); dw = yw_all[1] - yw_all[0]
    acc = {k: [] for k in ('J', 'G', 'Q')}
    for ch in np.array_split(yw_all, max(1, int(len(yv) * nw / 4e6) + 1)):
        V, Wg = np.meshgrid(yv, ch, indexing='ij')
        l = -L + D * V - D * D / 2 - Wg ** 2 / 2 - 0.5 * LN2PI
        lnaL = -V ** 2 / 2 - 0.5 * LN2PI; lnp = lnaL + np.logaddexp(0.0, l)
        lpi = log_expit(l); l1m = log_expit(-l); pi = np.exp(lpi)                                  # logs only: no underflow of pi
        lw = lnp + np.log(dv)[:, None] + np.log(dw)
        with np.errstate(divide='ignore'):
            law = np.log(np.abs(Wg))
            lsw2 = 2 * lpi + 2 * law                                                           # s_w^2 = pi^2 y_w^2
            lhvw2 = 2 * (lpi + l1m) + 2 * np.log(D) + 2 * law                                  # H_vw^2
            lhww2 = 2 * lpi + 2 * np.log(np.abs(1.0 - np.exp(l1m) * Wg ** 2))                 # H_ww^2 = pi^2 (1-(1-pi) y_w^2)^2
            lq2 = 2 * lpi + 2 * law + 2 * np.log(np.abs((2.0 - pi) * D - V))                 # (s_v s_w - 2H_vw)^2 = pi^2 y_w^2 ((2-pi)D - y_v)^2
            acc['J'].append(logsumexp(lw + lsw2)); acc['G'].append(logsumexp(lw + np.logaddexp(lhvw2, lhww2))); acc['Q'].append(logsumexp(lw + lq2))
    lJ, lG, lQ = (logsumexp(acc[k]) for k in ('J', 'G', 'Q'))
    lnM = np.logaddexp(0.0, -L)
    return {'D': D, 'z': z, 'tR': float(np.exp(lG - lJ)), 'Lambda': float(np.exp(lQ - lJ)), 'ln1_tJ': float(lnM - lJ), 'n_v': len(yv)}

if __name__ == '__main__':
    for D, zr in ((8, 0), (10, 0.5), (10, 1), (16, 1), (24, 1), (24, 2), (16, 4), (10, 2), (6, 1.5)):
        r = evaluate(float(D), zr * D); print(D, zr, {k: round(v, 4) for k, v in r.items()}, 'tR/ln %.3f' % (r['tR'] / r['ln1_tJ']))
