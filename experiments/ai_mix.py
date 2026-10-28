"""Stage AI: static targets (T1) Lambda <= A + B ln(1/(tJ_ww)) and (T2) tR <= A' + B' ln(1/(tJ_ww)) on generalisations of the
stage-AH family: n_l infinite w-lines (v = v_i, linear density r_i) + n_p points (v_j, w_j, weight e^{-L_j}), t = 1, closed form in the
log domain (mixture of w-uniform line components and point components).
For infinite lines J_ww and E[(H^2)_ww] come only from the neighbourhood of the points, so tR and Lambda are exact; ln(1/(tJ_ww)) is the
value ln(M_lb/(tJ)) with M_lb = sum_i r_i + sum_j e^{-L_j}, a LOWER bound for the normalised value of any probability-measure
realisation (envelope length >= 1; a longer envelope sqrt(V) adds 1/2 ln(2 pi V) > 0): the margins tR - B' ln(1/tJ) computed here are
therefore upper bounds.  (The bare unnormalised value 'ln1_tJ_unnorm' is NOT a bound when a point carries mass > 1.)
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp

LOG2PI = np.log(2 * np.pi)


def state_fields(lines, points, t=1.0, h=0.05, pad=10.0, wpad=10.0):
    """lines: list of (v_i, ln r_i); points: list of (v_j, w_j, L_j) with weight e^{-L_j}."""
    vs = [v for v, _ in lines] + [p[0] for p in points]; ws = [p[1] for p in points] + [0.0]
    # the valley between line i and point j sits at v_i + (L_j + ln r_i)/d + d/2 (d = v_j - v_i): widen the v window to contain it
    for vi, lr in lines:
        for vj, wj, L in points:
            d = vj - vi
            if abs(d) > 1e-9: vs.append(vi + (L + lr) / d + d / 2)
    yv = np.arange(min(vs) - pad, max(vs) + pad, h); yw = np.arange(min(ws) - wpad, max(ws) + wpad, h); V, W = np.meshgrid(yv, yw, indexing='ij')
    comps = []                                     # (log weight field, mean_v, mean_w, var_w)
    for v0, lr in lines: comps.append((lr - (V - v0) ** 2 / (2 * t) - 0.5 * (LOG2PI + np.log(t)), v0 * np.ones_like(V), W, t))
    for v0, w0, L in points: comps.append((-L - ((V - v0) ** 2 + (W - w0) ** 2) / (2 * t) - (LOG2PI + np.log(t)), v0 * np.ones_like(V), w0 * np.ones_like(W), 0.0))
    LW = np.stack([c[0] for c in comps]); lp = logsumexp(LW, axis=0); P = np.exp(LW - lp[None])
    mv = np.stack([c[1] for c in comps]); mw = np.stack([c[2] for c in comps]); vw = np.array([c[3] for c in comps])[:, None, None]
    Ev = np.sum(P * mv, 0); Ew = np.sum(P * mw, 0)
    Cvv = np.sum(P * mv ** 2, 0) - Ev ** 2; Cvw = np.sum(P * mv * mw, 0) - Ev * Ew; Cww = np.sum(P * (mw ** 2 + vw), 0) - Ew ** 2
    # cancellation-free s_w: sum_c P_c (mw_c - W); for line components mw_c - W = 0 exactly
    dW = np.stack([np.zeros_like(W) if c[3] > 0 else (c[2] - W) for c in comps]); sw = np.sum(P * dW, 0) / t
    dV = mv - V[None]; sv = np.sum(P * dV, 0) / t
    # Cww - t = sum P_c (var_c - t) + Var of means: for lines var = t -> 0 contribution; written without cancellation
    Cww_mt = np.sum(P * ((mw - Ew[None]) ** 2 + (vw - t)), 0)
    Hvv = (1 - Cvv / t) / t; Hvw = -Cvw / t / t; Hww = -Cww_mt / t / t
    return lp, h, sv, sw, Hvv, Hvw, Hww


def targets(lines, points, t=1.0, **kw):
    lp, h, sv, sw, Hvv, Hvw, Hww = state_fields(lines, points, t, **kw); lw = lp + 2 * np.log(h)
    def lsum(f):                                   # log of sum W f for f >= 0, entirely in the log domain (no underflow)
        with np.errstate(divide='ignore'): return float(logsumexp(lw + np.log(f)))
    lnJ = lsum(sw * sw); J, c = 1.0, 0.0
    G2 = np.exp(lsum(Hvw ** 2 + Hww ** 2) - lnJ); q2 = np.exp(lsum((sv * sw - 2 * Hvw) ** 2) - lnJ)
    # conservative normalisation: any probability realisation has total mass M >= sum_i r_i (envelope length >= 1) + sum_j e^{-L_j}
    lnM = float(logsumexp([lr for _, lr in lines] + [-L for _, _, L in points]))
    return {'tR': t * G2 / J, 'Lambda': t * q2 / J, 'ln1_tJ_unnorm': -(lnJ + np.log(t)), 'lnM_lb': lnM, 'ln1_tJ': lnM - (lnJ + np.log(t))}
