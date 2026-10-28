"""Stage AH: the line + point family in log-domain arithmetic (no underflow for very small point weights).
rho = line (density 1 per unit length along w at v = 0; or a Gaussian envelope N(0, V) of total mass 1 - m) + point of weight eps at (D, 0).
t = blur variance.  Window y_v in [-9 sd, max(D, y_v*) + 12 sd], |y_w| <= Wd; outside the window only the line contributes
(s_w = -y_w/(V+t), H_vw = 0, H_ww = 1/(V+t): added analytically for the envelope).
Returns: Y (A = 3, a0 = 1/2 and other splits), kappa_H, X_eta, tR, t^2 R' (central difference in t), with J_ww and E[(H^2)_ww]
including the envelope's own Fisher information (V < inf).  Also an independent Hessian by finite differences of ln p (check).
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import expit, logsumexp
from af_tools import decompose

LOG2PI = np.log(2 * np.pi)


def logphi(x, t): return -x * x / (2 * t) - 0.5 * (LOG2PI + np.log(t))


def window(D, L, t, Wd=None, h=None):
    sd = np.sqrt(t); ystar = D / 2 + L / D * t if D > 0 else 0.0
    h = 0.03 * sd if h is None else h; Wd = 14 * sd if Wd is None else Wd
    yv = np.arange(-9 * sd, max(D, ystar) + 12 * sd, h); yw = np.arange(-Wd, Wd + h / 2, h); return np.meshgrid(yv, yw, indexing='ij'), h


def fields(D, L, t, V=np.inf, Wd=None, h=None):
    """point weight eps = exp(-L) relative to the line density (infinite line: density 1; envelope: (1-m) N(0,V) with m = eps/(1+eps) and
    the comparison made per unit length at w = 0 is NOT used: eps is the point mass, (1 - eps) the line mass)."""
    (Vv, Ww), h = window(D, L, t, Wd, h)
    if np.isinf(V):
        lpL = logphi(Vv, t); muw = Ww; vw = t * np.ones_like(Ww)
    else:
        lpL = np.log1p(-np.exp(-L)) + logphi(Vv, t) + logphi(Ww, V + t); muw = Ww * V / (V + t); vw = V * t / (V + t) * np.ones_like(Ww)
    lpP = -L + logphi(Vv - D, t) + logphi(Ww, t)
    lp = np.logaddexp(lpL, lpP); piP = expit(lpP - lpL); piL = 1 - piP
    Dv, Dw = D, -muw
    Cvv = piL * piP * Dv * Dv; Cvw = piL * piP * Dv * Dw; Cww = piL * vw + piL * piP * Dw * Dw
    sv = (piP * D - Vv) / t
    shrink = 0.0 if np.isinf(V) else t / (V + t)                     # muw - y_w = -y_w t/(V+t): written out to avoid cancellation
    sw = (-piL * Ww * shrink - piP * Ww) / t
    Hvv = (1 - Cvv / t) / t; Hvw = -Cvw / t / t
    Hww = (piP + piL * shrink - piL * piP * muw * muw / t) / t      # = (1 - C_ww/t)/t with C_ww = pi_L V t/(V+t) + pi_L pi_P muw^2
    return lp, h, sv, sw, Hvv, Hvw, Hww, (Vv, Ww)


def evaluate(D, L, t=1.0, V=np.inf, hh=2e-3, Wd=None, h=None):
    res = {}; out = None
    for key, tt in (('m', t * (1 - hh)), ('0', t), ('p', t * (1 + hh))):
        lp, hx, sv, sw, Hvv, Hvw, Hww, _ = fields(D, L, tt, V, Wd, h)
        c = lp.max(); W = np.exp(lp - c) * hx * hx                      # weights scaled by e^{-c}
        Jwin = np.sum(W * sw * sw); G2win = np.sum(W * (Hvw ** 2 + Hww ** 2)); Mwin = np.sum(W)
        if np.isinf(V): Jenv = G2env = 0.0
        else:   # envelope outside the window: mass (1 - eps) * P(|w| > Wd) ~ (1 - eps); Fisher 1/(V+t), H_ww^2 = 1/(V+t)^2
            Wd_ = (14 * np.sqrt(tt)) if Wd is None else Wd; frac = 1 - np.exp(logphi(0.0, V + tt)) * 2 * Wd_          # mass of the envelope outside the window (V >> Wd^2)
            le = np.log1p(-np.exp(-L)) + np.log(frac) - c
            Jenv = np.exp(le - np.log(V + tt)); G2env = np.exp(le - 2 * np.log(V + tt))
        J = Jwin + Jenv; G2 = G2win + G2env
        if key == '0':
            out = decompose(W.ravel(), Hvv.ravel(), Hvw.ravel(), Hww.ravel(), tt, J / Mwin)    # decompose normalises by sum W: pass J per window mass
            for k in list(out):
                if isinstance(out[k], dict): continue
            out['tJ_window_share'] = Jwin / J; out['lnJ_scaled'] = float(np.log(J) + c)
        res[key] = G2 / J
    out['tR'] = t * res['0']; out['t2R1'] = (t * res['p'] - t * res['m']) / (2 * hh); return out


def fd_hessian_check(D, L, t=1.0):
    """independent: Hessian of ln p by central finite differences of the closed-form ln p (no posterior formulas)."""
    lp, h, sv, sw, Hvv, Hvw, Hww, (Vv, Ww) = fields(D, L, t)
    f = lp; Hvv_fd = -(f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / h ** 2; Hww_fd = -(f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / h ** 2
    Hvw_fd = -(f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4 * h * h)
    w = np.exp(lp[1:-1, 1:-1] - lp.max()); sl = (slice(1, -1), slice(1, -1))
    rel = lambda A, B: float(np.sum(w * np.abs(A - B[sl])) / np.sum(w * np.abs(B[sl])))
    return {'Hvv': rel(Hvv_fd, Hvv), 'Hvw': rel(Hvw_fd, Hvw), 'Hww': rel(Hww_fd, Hww)}
