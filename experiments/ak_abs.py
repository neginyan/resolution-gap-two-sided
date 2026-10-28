"""Stage AK: single-blur chain with an uncentred second moment and a |sigma|-weighted Cauchy-Schwarz.
Two-level notation (t = t1 + t2, t1 = tau t, sigma = grad ln rho_1, X1 | y the level-t1 posterior), all exact:
  s_w(y) = E[sigma_w(X1) | y],   c2(y) := d_w grad p / p = E[(X1 - y) sigma_w(X1) | y] / t2   (uncentred),   grad s_w = c2 - s s_w.
Chain (Proposition AK1):
  |grad s_w|^2 <= 2|c2|^2 + 2|s|^2 s_w^2,
  |c2|^2 <= B(y) A(y) / t2^2      (Cauchy-Schwarz with weight |sigma_w|;  A = E[|sigma_w| | y],  B = E[|X1 - y|^2 |sigma_w| | y]),
  E_p[B A] = E_Gamma[|X1 - Y|^2 W],  W = |sigma_w(X1)| A(Y),  E W = J_abs := E_p A^2,
  DV (beta = 1/(4 t2), E e^{beta |X1-Y|^2} = 2):  E_Gamma[|X1-Y|^2 W] <= 4 t2 J_abs [KL + ln 2],  KL <= ln(E W^2 / J_abs^2) <= ln(3/(t1^2 J_abs^2)),
  E_p |s|^2 s_w^2 <= 4 t J [2 ln(1/(tJ)) + ln 6] / t^2   (DV with Q = p s_w^2 / J, |s|^2 <= E[|Z|^2|y]/t, E s_w^4 <= 3/t^2).
  => tR <= 8 [ kappa/(1 - tau) (2 ln(1/(tau t J_abs)) + ln 6) + 2 ln(1/(tJ)) + ln 6 ],   kappa := J_abs / J  (sign-cancellation factor, 1 <= kappa <= J(tau t)/J(t)).
Families: lines (v_i, ln r_i) uniform in w, points (v_j, w_j, L_j) with weight e^{-L_j}; t = 1.  Level-t1 posterior: per component a Gaussian
(line: N((v_i + (t1/t)(y_v - v_i), y_w), diag(t1 t2/t, t2)); point: N(x_j + (t1/t)(y - x_j), t1 t2/t I)); 2-d Gauss-Hermite; log domain.
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp

LN2PI = np.log(2 * np.pi)


def lse(a, axis=None, b=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        return logsumexp(a, axis=axis, b=b, return_sign=b is not None)


def slog(x):
    with np.errstate(divide='ignore'):
        return np.log(np.abs(x)), np.sign(x)


def comp_logw(lines, points, V, W, t):
    """log component weights of p at blur t (per y), means (mv, mw or None for lines)"""
    A, M = [], []
    for v0, lr in lines:
        A.append(lr - (V - v0) ** 2 / (2 * t) - 0.5 * (LN2PI + np.log(t))); M.append((v0, None))
    for v0, w0, L in points:
        A.append(-L - ((V - v0) ** 2 + (W - w0) ** 2) / (2 * t) - (LN2PI + np.log(t))); M.append((v0, w0))
    return np.stack(A), M


def closed_form(lines, points, V, W, t):
    """ln p, log|s_w|, sign, log|s|^2, log|grad s_w|^2 (blur t), all from pairwise sums of posterior weights (no cancellation)"""
    A, M = comp_logw(lines, points, V, W, t); lp = logsumexp(A, 0); lP = A - lp[None]; n = len(M)
    mw = lambda c: (W if M[c][1] is None else M[c][1] * np.ones_like(W))
    # s_w = sum_c P_c (mw_c - W)/t (lines exactly 0);  s_v = sum_c P_c (mv_c - V)/t
    sw = lse(np.stack([lP[c] + slog(mw(c) - W)[0] for c in range(n)]), 0, b=np.stack([slog(mw(c) - W)[1] for c in range(n)]))
    sv = lse(np.stack([lP[c] + slog(M[c][0] - V)[0] for c in range(n)]), 0, b=np.stack([slog(M[c][0] - V)[1] for c in range(n)]))
    # C_vw = sum_{c<c'} P P' dv dw ; C_ww - t = sum_c P_c (var_c - t) + sum_{c<c'} P P' dw^2   (var: line t, point 0)
    tv, sgv, tw, sgw = [], [], [], []
    for c in range(n):
        for d in range(c + 1, n):
            dv = M[c][0] - M[d][0]; dw = mw(c) - mw(d); l2 = lP[c] + lP[d]
            a_, s_ = slog(dv * dw); tv.append(l2 + a_); sgv.append(s_)
            a_, s_ = slog(dw * dw); tw.append(l2 + a_); sgw.append(s_)
        if M[c][1] is not None: tw.append(lP[c] + np.log(t)); sgw.append(-np.ones_like(V))
    Cvw = lse(np.stack(tv), 0, b=np.stack(sgv)) if tv else (np.full_like(V, -np.inf), np.zeros_like(V))
    Cww = lse(np.stack(tw), 0, b=np.stack(sgw))
    # grad s_w = -H e_w = (C_vw/t^2, (C_ww - t)/t^2)
    lg = np.logaddexp(2 * Cvw[0], 2 * Cww[0]) - 4 * np.log(t)
    ls2 = np.logaddexp(2 * sw[0], 2 * sv[0]) - 2 * np.log(t)
    return lp, sw[0] - np.log(t), sw[1], ls2, lg, (Cvw[0] - 2 * np.log(t), Cvw[1]), (Cww[0] - 2 * np.log(t), Cww[1]), (sv[0] - np.log(t), sv[1])


def chain(lines, points, yv, yw, tau=0.5, ngh=24, t=1.0, cell=False):
    """returns log-integrals over the y grid (weights p dy).  cell=True: normalise over the grid (periodic cell)."""
    t1 = tau * t; t2 = t - t1; hv = np.gradient(yv) if len(yv) > 1 else np.ones(1); hw = yw[1] - yw[0]
    gx, gwt = np.polynomial.hermite_e.hermegauss(ngh); lgw = np.log(gwt / np.sqrt(2 * np.pi))
    U1, U2 = np.meshgrid(gx, gx, indexing='ij'); U1 = U1.ravel(); U2 = U2.ravel(); LG = (lgw[:, None] + lgw[None, :]).ravel()
    acc = {k: [] for k in ('M', 'J', 'G', 'S2', 'Jabs', 'BA', 'W2', 'c2', 'idc2', 'twd')}
    nc = len(lines) + len(points); ch = max(1, int(2e6 / (nc * nc * ngh * ngh)))          # memory: nc x chunk x (nc ngh^2) <= 2e6 per array
    chunks = [slice(i, min(i + ch, len(yv))) for i in range(0, len(yv), ch)]
    for yw_, sl in [(a_, s_) for a_ in yw for s_ in chunks]:
        V_ = yv[sl]; W_ = np.full_like(V_, yw_); hvc = hv[sl]
        lp, lsw, ssw, ls2, lg, Cvw, Cww, sv = closed_form(lines, points, V_, W_, t)
        A_, M_ = comp_logw(lines, points, V_, W_, t); lP = A_ - lp[None]
        sd = np.sqrt(t1 * t2 / t); X1v, X1w, LW = [], [], []
        for c, (v0, w0) in enumerate(M_):
            X1v.append((v0 + (t1 / t) * (V_ - v0))[:, None] + sd * U1[None]); LW.append(lP[c][:, None] + LG[None])
            X1w.append((W_[:, None] + np.sqrt(t2) * U2[None]) if w0 is None else ((w0 + (t1 / t) * (W_ - w0))[:, None] + sd * U2[None]))
        X1v = np.concatenate(X1v, 1); X1w = np.concatenate(X1w, 1); LW = np.concatenate(LW, 1)
        # level-t1 score in w at the nodes: sigma_w = sum_points pi1_j (w_j - x_w)/t1  (log-signed)
        A1, M1 = comp_logw(lines, points, X1v, X1w, t1); l1 = A1 - logsumexp(A1, 0)[None]
        terms = [(l1[c] + slog(M1[c][1] - X1w)[0], slog(M1[c][1] - X1w)[1]) for c in range(len(M1)) if M1[c][1] is not None]
        lsig, ssig = lse(np.stack([a for a, _ in terms]), 0, b=np.stack([b for _, b in terms])); lsig = lsig - np.log(t1)
        lA = lse(LW + lsig, 1)
        dyv, dyw = X1v - V_[:, None], X1w - W_[:, None]
        with np.errstate(divide='ignore'):
            lB = lse(LW + np.log(dyv ** 2 + dyw ** 2) + lsig, 1)
            c2v = lse(LW + np.log(np.abs(dyv)) + lsig, 1, b=np.sign(dyv) * ssig); c2w = lse(LW + np.log(np.abs(dyw)) + lsig, 1, b=np.sign(dyw) * ssig)
            sq = lse(LW + lsig, 1, b=np.broadcast_to(ssig, LW.shape))
        lc2 = np.logaddexp(2 * c2v[0], 2 * c2w[0]) - 2 * np.log(t2)
        # identity checks (p-weighted relative L2 errors): tweedie s_w = E sigma_w ; c2 = grad s_w + s s_w (closed form)
        c2_closed_v = Cvw[1] * np.exp(Cvw[0]) + sv[1] * np.exp(sv[0]) * ssw * np.exp(lsw)
        c2_closed_w = Cww[1] * np.exp(Cww[0]) + ssw * np.exp(lsw) * ssw * np.exp(lsw)
        lw0 = lp + np.log(hvc * hw)
        dq_v = c2v[1] * np.exp(c2v[0] - np.log(t2)) - c2_closed_v; dq_w = c2w[1] * np.exp(c2w[0] - np.log(t2)) - c2_closed_w
        with np.errstate(divide='ignore'):
            acc['idc2'].append((lse(lw0 + np.log(dq_v ** 2 + dq_w ** 2)), lse(lw0 + np.log(c2_closed_v ** 2 + c2_closed_w ** 2))))
            acc['twd'].append((lse(lw0 + np.log((sq[1] * np.exp(sq[0]) - ssw * np.exp(lsw)) ** 2)), lse(lw0 + 2 * lsw)))
        lw = lp + np.log(hvc * hw)
        acc['M'].append(lse(lw)); acc['J'].append(lse(lw + 2 * lsw)); acc['G'].append(lse(lw + lg)); acc['S2'].append(lse(lw + ls2 + 2 * lsw))
        acc['Jabs'].append(lse(lw + 2 * lA)); acc['BA'].append(lse(lw + lB + lA - 2 * np.log(t2))); acc['c2'].append(lse(lw + lc2))
        acc['W2'].append(lse(lw + 2 * lA + lse(LW + 2 * lsig, 1) - 2 * lA + 2 * lA))   # E W^2 = E_p[A^2 E(sigma^2|y)]
    L = {k: float(lse(np.array(v))) for k, v in acc.items() if k not in ('idc2', 'twd')}
    lM = L['M']
    ir = lambda k: float(np.sqrt(np.exp(lse(np.array([a for a, _ in acc[k]])) - lse(np.array([b for _, b in acc[k]])))))
    o = {'tau': tau, 'identity_c2_rel_L2': ir('idc2'), 'tweedie_rel_L2': ir('twd')}
    o['tJ'] = float(np.exp(L['J'] - lM)) * t; o['ln1_tJ'] = float(-(L['J'] - lM) - np.log(t)); o['tR'] = float(t * np.exp(L['G'] - L['J']))
    o['kappa'] = float(np.exp(L['Jabs'] - L['J']))
    # step values (all divided by E|grad s_w|^2):
    G = L['G']
    o['step_split'] = float((2 * np.exp(L['c2'] - G) + 2 * np.exp(L['S2'] - G)))
    o['step_CS'] = float((2 * np.exp(L['BA'] - G) + 2 * np.exp(L['S2'] - G)))
    Jabs_n = L['Jabs'] - lM; KLj = float(L['W2'] - lM - 2 * Jabs_n); KLf = float(np.log(3 / t1 ** 2) - 2 * Jabs_n)
    dv_c2 = 4 * np.exp(L['Jabs'] - G) / t2 * (KLj + np.log(2)); fin_c2 = 4 * np.exp(L['Jabs'] - G) / t2 * (KLf + np.log(2))
    s2_bound = 4 * np.exp(L['J'] - G) * (2 * o['ln1_tJ'] + np.log(6))
    o['step_DV_jensen'] = float(2 * dv_c2 + 2 * np.exp(L['S2'] - G)); o['step_final'] = float(2 * fin_c2 + 2 * s2_bound)
    o['KL_jensen'] = KLj; o['KL_final'] = KLf
    o['bound_AK1'] = float(8 * (o['kappa'] / (1 - tau) * (2 * (o['ln1_tJ'] + np.log(1 / tau) - np.log(o['kappa'])) + np.log(6)) + 2 * o['ln1_tJ'] + np.log(6)))
    o['frac_c2'] = float(np.exp(L['c2'] - G)); o['frac_S2'] = float(np.exp(L['S2'] - G))
    return o
