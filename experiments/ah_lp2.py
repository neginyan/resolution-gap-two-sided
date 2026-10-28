"""Stage AH: line with a Gaussian envelope + one point, closed form at any blur t (exact posterior; 2-d grid only for the integrals).
rho = (1 - m) * [delta(x_v) N(x_w; 0, V)] + m * delta_(D, 0)        (V = inf: the infinite line, unnormalised, line density 1, point weight eps)
blurred:  p_L = (1-m) phi_t(y_v) N(y_w; 0, V + t),   p_P = m phi_t(y_v - D) phi_t(y_w)
posterior: line part X = (0, X_w), X_w ~ N(y_w V/(V+t), V t/(V+t));  point part X = (D, 0)
K = C/t, s = (E[X|y] - y)/t, H = (I - K)/t.  Returns the eigen split (af_tools.decompose), kappa_H, tR and t^2 R' (central difference in t).
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from af_tools import decompose


def phi(x, t): return np.exp(-x * x / (2 * t)) / np.sqrt(2 * np.pi * t)


def fields(D, m, V, t, h=None, Wd=None):
    sd = np.sqrt(t); h = 0.04 * sd if h is None else h
    yv = np.arange(-9 * sd, max(D, 0) + 40 * sd, h); Wd = 12 * sd if Wd is None else Wd; yw = np.arange(-Wd, Wd, h)
    Vv, Ww = np.meshgrid(yv, yw, indexing='ij')
    if np.isinf(V): pL = (1 - m) * phi(Vv, t); muw = Ww; vw = t
    else: pL = (1 - m) * phi(Vv, t) * phi(Ww, V + t); muw = Ww * V / (V + t); vw = V * t / (V + t)
    pP = m * phi(Vv - D, t) * phi(Ww, t); p = pL + pP; ok = p > 0; p = np.where(ok, p, 1e-300); piP = pP / p; piL = 1 - piP
    Ev = piP * D; Ew = piL * muw
    Cvv = piL * piP * D * D; Cvw = piL * piP * D * (0 - muw) * -1 * -1; Cww = piL * vw + piL * piP * muw * muw
    # covariance of the two-component mixture: within (line: var vw in w) + between (Delta = (D, -muw))
    Dv, Dw = D - 0.0, 0.0 - muw
    Cvv = piL * piP * Dv * Dv; Cvw = piL * piP * Dv * Dw; Cww = piL * vw + piL * piP * Dw * Dw
    sv = (Ev - Vv) / t; sw = (Ew - Ww) / t
    Hvv = (1 - Cvv / t) / t; Hvw = -Cvw / t / t; Hww = (1 - Cww / t) / t
    return p * h * h, sv, sw, Hvv, Hvw, Hww


def evaluate(D, m, V, t=1.0, hh=2e-3):
    res = {}
    for key, tt in (('m', t * (1 - hh)), ('0', t), ('p', t * (1 + hh))):
        W, sv, sw, Hvv, Hvw, Hww = fields(D, m, V, tt)
        J = np.sum(W * sw * sw) / np.sum(W); G2 = np.sum(W * (Hvw ** 2 + Hww ** 2)) / np.sum(W)
        if key == '0': d = decompose(W.ravel(), Hvv.ravel(), Hvw.ravel(), Hww.ravel(), tt, J); d['tJ_unnorm'] = tt * np.sum(W * sw * sw)
        res[key] = G2 / J
    d['t2R1'] = (t * res['p'] - t * res['m']) / (2 * hh); return d
