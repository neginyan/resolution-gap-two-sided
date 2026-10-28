"""Stage AI: fully log-domain version of ai_mix.targets (no underflow of posterior weights).
Every field is carried as (log|x|, sign x).  For component c with log weight a_c, log posterior weight lP_c = a_c - logsumexp(a);
the centred means d_c = m_c - E[m] are formed as sums over pairs, d_c = sum_c' P_c' (m_c - m_c'), so no cancellation occurs; then
  s_w = sum_c P_c (mw_c - W)/t,  s_v = sum_c P_c (mv_c - V)/t,
  C_vw = sum_c P_c dv_c dw_c,  C_ww - t = sum_c P_c (dw_c^2 + var_c - t),  H_vw = -C_vw/t^2,  H_ww = -(C_ww - t)/t^2.
Normalisation as in ai_mix (M_lb = sum_i r_i + sum_j e^{-L_j}, conservative lower bound on ln(1/tJ)).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp

LOG2PI = np.log(2 * np.pi)
NEG = -np.inf


def lsig(x):
    with np.errstate(divide='ignore'): return np.log(np.abs(x)), np.sign(x)


def ssum(terms):
    """signed log-sum of a list of (log|x|, sign) arrays."""
    A = np.stack([t[0] for t in terms]); S = np.stack([t[1] for t in terms])
    with np.errstate(divide='ignore', invalid='ignore'):
        v, s = logsumexp(A, axis=0, b=S, return_sign=True)
    return np.where(np.isfinite(v), v, NEG), np.where(np.isfinite(v), s, 0.0)


def mul(*xs):
    return sum(x[0] for x in xs), np.prod([x[1] for x in xs], axis=0)


def grid(lines, points, h, pad, wpad):
    vs = [v for v, _ in lines] + [p[0] for p in points]; ws = [p[1] for p in points] + [0.0]
    for vi, lr in lines:
        for vj, wj, L in points:
            d = vj - vi
            if abs(d) > 1e-9: vs.append(vi + (L + lr) / d + d / 2)
    yv = np.arange(min(vs) - pad, max(vs) + pad, h); yw = np.arange(min(ws) - wpad, max(ws) + wpad, h)
    return np.meshgrid(yv, yw, indexing='ij')


def targets(lines, points, t=1.0, h=0.05, pad=10.0, wpad=10.0):
    V, W = grid(lines, points, h, pad, wpad)
    if V.size > 8e6: raise MemoryError('grid above 8e6 points')
    comps = []                                    # (log weight, mv (const), mw (const or None = W), var_w)
    for v0, lr in lines: comps.append((lr - (V - v0) ** 2 / (2 * t) - 0.5 * (LOG2PI + np.log(t)), v0, None, t))
    for v0, w0, L in points: comps.append((-L - ((V - v0) ** 2 + (W - w0) ** 2) / (2 * t) - (LOG2PI + np.log(t)), v0, w0, 0.0))
    A = np.stack([c[0] for c in comps]); lp = logsumexp(A, axis=0); lP = A - lp[None]
    mwf = lambda c: (W if comps[c][2] is None else comps[c][2] * np.ones_like(W))
    n = len(comps)
    dv, dw = [], []
    for c in range(n):
        tv = [mul((lP[k], np.ones_like(V)), lsig(np.full_like(V, comps[c][1] - comps[k][1]))) for k in range(n) if k != c]
        tw = [mul((lP[k], np.ones_like(V)), lsig(mwf(c) - mwf(k))) for k in range(n) if k != c]
        dv.append(ssum(tv) if tv else (np.full_like(V, NEG), np.zeros_like(V))); dw.append(ssum(tw) if tw else (np.full_like(V, NEG), np.zeros_like(V)))
    one = np.ones_like(V)
    sw = ssum([mul((lP[c], one), lsig(mwf(c) - W)) for c in range(n)]); sw = (sw[0] - np.log(t), sw[1])
    sv = ssum([mul((lP[c], one), lsig(comps[c][1] - V)) for c in range(n)]); sv = (sv[0] - np.log(t), sv[1])
    Cvw = ssum([mul((lP[c], one), dv[c], dw[c]) for c in range(n)])
    Cww = ssum([mul((lP[c], one), dw[c], dw[c]) for c in range(n)] + [mul((lP[c], one), lsig(np.full_like(V, comps[c][3] - t))) for c in range(n)])
    Hvw = (Cvw[0] - 2 * np.log(t), -Cvw[1]); Hww = (Cww[0] - 2 * np.log(t), -Cww[1])
    lw = lp + 2 * np.log(h)
    lJ = logsumexp(lw + 2 * sw[0])
    lG = logsumexp(lw + np.logaddexp(2 * Hvw[0], 2 * Hww[0]))
    q = ssum([mul(sv, sw), (Hvw[0] + np.log(2), -Hvw[1])])
    lQ = logsumexp(lw + 2 * q[0])
    lnM = float(logsumexp([lr for _, lr in lines] + [-L for _, _, L in points]))
    return {'tR': float(t * np.exp(lG - lJ)), 'Lambda': float(t * np.exp(lQ - lJ)), 'ln1_tJ': float(lnM - lJ - np.log(t)), 'lnM_lb': lnM}
