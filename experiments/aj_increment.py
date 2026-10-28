"""Stage AJ: the heat-flow increment Delta_tau = ln(J_ww(tau t)/J_ww(t)) and eps = e^Delta - 1 for the w-line + point family and the
two-line + midpoint family, by rescaling (blur tau t at fixed geometry = blur t with lengths divided by sqrt(tau); line density per unit
rescaled length sqrt(tau)), with the separate closed-form evaluators.  t = 1.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from ai_lp_exact import evaluate as lpx
from ai_twoline_exact import evaluate as tlx


def lnJraw_lp(D, z, **kw):
    r = lpx(D, z, **kw); L = D * (z + D / 2); return np.logaddexp(0.0, -L) - r['ln1_tJ'], r


def delta_lp(D, z, tau=0.5, **kw):
    """line density 1, point weight e^{-L}; at blur tau: D' = D/sqrt(tau), all densities divided by sqrt(tau) -> point weight e^{-L}/sqrt(tau)"""
    l1, r = lnJraw_lp(D, z, **kw); L = D * (z + D / 2); Dp = D / np.sqrt(tau); Lp = L - 0.5 * np.log(1 / tau); zp = Lp / Dp - Dp / 2
    l2, _ = lnJraw_lp(Dp, zp, **kw)
    # raw J of the rescaled evaluation is per rescaled length with line density 1; physical: J_phys(tau) = sqrt(tau) * J' / tau
    lJtau = l2 + 0.5 * np.log(tau) - np.log(tau)
    return float(lJtau - l1), r


def delta_tl(D, z, tau=0.5, lr=0.0):
    r = tlx(D, z, lr); L = D * (z + D / 2); l1 = float(logsum([0.0, lr, -L])) - r['ln1_tJ']
    Dp = D / np.sqrt(tau); Lp = L - 0.5 * np.log(1 / tau); zp = Lp / Dp - Dp / 2
    r2 = tlx(Dp, zp, lr); l2 = float(logsum([0.0, lr, -Lp])) - r2['ln1_tJ']
    return float(l2 + 0.5 * np.log(tau) - np.log(tau) - l1), r


def logsum(a):
    from scipy.special import logsumexp; return logsumexp(a)


if __name__ == '__main__':
    for D, z in ((8.0, 0.0), (8.0, 4.0), (8.0, 8.0)):
        d, r = delta_lp(D, z); print('line+point', D, z, 'Delta %.4f eps %.4f tR %.3f' % (d, np.expm1(d), r['tR']))
