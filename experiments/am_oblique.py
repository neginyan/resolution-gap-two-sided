"""Stage AM, plan A'': oblique two-level chain.  Keep the residual noise along ONE unit direction e = (cos th, sin th):
  level 1: rho_1 = rho * N(0, Sigma1),  Sigma1 = t I - t2 e e^T  (w-variance t - t2 sin^2 th),   final: p = rho_1 * N(0, t2 e e^T).
Exact identities (the posterior of X1 given y lives on the line y + R e, X1 = y - lambda e):
  s_w(y) = E[sigma_w(X1) | y],   d_e s_w = Cov(X1.e, sigma_w | y)/t2,   c2e := d_e s_w + s_e s_w = -E[lambda sigma_w | y]/t2,  s_e = e.s.
Since d_v s_w = (d_e s_w - sin th d_w s_w)/cos th:  H_vw^2 <= (2/cos^2 th)(d_e s_w)^2 + 2 tan^2 th H_ww^2.
Proposition AM1 (t = 1):  tR <= (1 + 2 tan^2 th)(A1 + B1 ln(1/tJ))                                     [w-slice lemma]
                               + (2/cos^2 th) [ 8 kappa_e/(1-tau) (ln(3/(t_w1^2 J_abs^2)) + ln sqrt2) + 8 (2 ln(1/tJ) + ln 3 + ln sqrt2) ],
  kappa_e = J_abs/J,  J_abs = E_p[(E|sigma_w(X1)| | y)^2],  t2 = (1 - tau) t,  t_w1 = t - t2 sin^2 th.
Components: w-lines (v_i, ln r_i) and points (v_j, w_j, L_j).  Posterior of lambda per component is Gaussian (computed exactly);
1-d Gauss-Hermite in lambda; log domain.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp
from ak_abs import closed_form, comp_logw, slog, lse
from al_aniso import one_d_constants

LN2PI = np.log(2 * np.pi)


def level1(lines, points, Xv, Xw, S1):
    """log weights of the level-1 components at the nodes, and their w-scores (point components only)."""
    Si = np.linalg.inv(S1); ld = np.log(np.linalg.det(S1)); A, SW = [], []
    for v0, lr in lines:
        A.append(lr - (Xv - v0) ** 2 / (2 * S1[0, 0]) - 0.5 * (LN2PI + np.log(S1[0, 0]))); SW.append(None)
    for v0, w0, L in points:
        dv, dw = Xv - v0, Xw - w0
        A.append(-L - 0.5 * (Si[0, 0] * dv * dv + 2 * Si[0, 1] * dv * dw + Si[1, 1] * dw * dw) - LN2PI - 0.5 * ld)
        SW.append(-(Si[1, 0] * dv + Si[1, 1] * dw))
    return np.stack(A), SW


def chain(lines, points, yv, yw, theta=0.0, tau=0.5, ngh=40, t=1.0):
    c, s = np.cos(theta), np.sin(theta); e = np.array([c, s]); t2 = (1 - tau) * t
    S1 = t * np.eye(2) - t2 * np.outer(e, e); Si = np.linalg.inv(S1); tw1 = S1[1, 1]
    hv = np.gradient(yv); hw = yw[1] - yw[0] if len(yw) > 1 else 1.0
    gx, gwt = np.polynomial.hermite_e.hermegauss(ngh); lgw = np.log(gwt / np.sqrt(2 * np.pi))
    acc = {k: [] for k in ('M', 'J', 'Gvw', 'Gww', 'Ge', 'S2', 'Jabs', 'BA', 'W2', 'id', 'twd')}
    nc = len(lines) + len(points); ch = max(1, int(2e6 / (nc * nc * ngh)))
    chunks = [slice(i, min(i + ch, len(yv))) for i in range(0, len(yv), ch)]
    for yw_, sl in [(a_, s_) for a_ in yw for s_ in chunks]:
        V_ = yv[sl]; W_ = np.full_like(V_, yw_); hvc = hv[sl]
        lp, lsw, ssw, ls2, lg, Cvw, Cww, sv = closed_form(lines, points, V_, W_, t)
        A_, M_ = comp_logw(lines, points, V_, W_, t); lP = A_ - lp[None]
        LAM, LW = [], []
        for k, (v0, w0) in enumerate(M_):
            if w0 is None:            # line: prior along the e-line depends on x_v = y_v - lambda c only
                P = c * c / S1[0, 0] + 1 / t2; m = (c * (V_ - v0) / S1[0, 0]) / P
            else:
                d0 = np.stack([V_ - v0, W_ - w0]); P = e @ Si @ e + 1 / t2; m = (e @ Si @ d0) / P
            LAM.append(m[:, None] + gx[None] / np.sqrt(P)); LW.append(lP[k][:, None] + lgw[None])
        LAM = np.concatenate(LAM, 1); LW = np.concatenate(LW, 1)
        X1v = V_[:, None] - LAM * c; X1w = W_[:, None] - LAM * s
        A1, SW = level1(lines, points, X1v, X1w, S1); l1 = A1 - logsumexp(A1, 0)[None]
        terms = [(l1[k] + slog(SW[k])[0], slog(SW[k])[1]) for k in range(len(SW)) if SW[k] is not None]
        lsig, ssig = lse(np.stack([a for a, _ in terms]), 0, b=np.stack([b for _, b in terms]))
        lA = lse(LW + lsig, 1)
        with np.errstate(divide='ignore'):
            lB = lse(LW + 2 * np.log(np.abs(LAM)) + lsig, 1)
            c2e = lse(LW + np.log(np.abs(LAM)) + lsig, 1, b=-np.sign(LAM) * ssig)
            sq = lse(LW + lsig, 1, b=np.broadcast_to(ssig, LW.shape)); lE2 = lse(LW + 2 * lsig, 1)
        dvs = Cvw[1] * np.exp(Cvw[0]); dws = Cww[1] * np.exp(Cww[0]); swv = ssw * np.exp(lsw); svv = sv[1] * np.exp(sv[0])
        des = c * dvs + s * dws; se = c * svv + s * swv
        c2e_closed = des + se * swv; c2e_val = c2e[1] * np.exp(c2e[0] - np.log(t2))
        lw = lp + np.log(hvc * hw)
        with np.errstate(divide='ignore'):
            acc['id'].append((lse(lw + np.log((c2e_val - c2e_closed) ** 2)), lse(lw + np.log(c2e_closed ** 2))))
            acc['twd'].append((lse(lw + np.log((sq[1] * np.exp(sq[0]) - swv) ** 2)), lse(lw + 2 * lsw)))
            acc['Ge'].append(lse(lw + np.log(des ** 2)))
            acc['S2'].append(lse(lw + np.log(se ** 2) + 2 * lsw))
        acc['M'].append(lse(lw)); acc['J'].append(lse(lw + 2 * lsw)); acc['Gvw'].append(lse(lw + 2 * Cvw[0])); acc['Gww'].append(lse(lw + 2 * Cww[0]))
        acc['Jabs'].append(lse(lw + 2 * lA)); acc['BA'].append(lse(lw + lB + lA - 2 * np.log(t2))); acc['W2'].append(lse(lw + 2 * lA + lE2))
    L = {k: float(lse(np.array(v))) for k, v in acc.items() if k not in ('id', 'twd')}
    ir = lambda k: float(np.sqrt(np.exp(lse(np.array([a for a, _ in acc[k]])) - lse(np.array([b for _, b in acc[k]])))))
    lM = L['M']; ln1 = float(-(L['J'] - lM) - np.log(t))
    o = {'theta': theta, 'tau': tau, 'identity_c2e_rel_L2': ir('id'), 'tweedie_rel_L2': ir('twd'), 'ln1_tJ': ln1,
         'tR_vw': float(t * np.exp(L['Gvw'] - L['J'])), 'tR_ww': float(t * np.exp(L['Gww'] - L['J'])), 'tR_e': float(t * np.exp(L['Ge'] - L['J'])),
         'kappa_e': float(np.exp(L['Jabs'] - L['J']))}
    o['tR'] = o['tR_vw'] + o['tR_ww']
    Jabs_n = L['Jabs'] - lM; KLf = float(np.log(3 / tw1 ** 2) - 2 * Jabs_n)
    e_bound = 2 * 4 * o['kappa_e'] / (1 - tau) * (KLf + 0.5 * np.log(2)) + 2 * 4 * (2 * ln1 + np.log(3) + 0.5 * np.log(2))   # bound on t E (d_e s_w)^2 / J
    o['e_step_CS'] = float((2 * np.exp(L['BA'] - L['Ge']) + 2 * np.exp(L['S2'] - L['Ge'])))
    o['e_bound'] = float(e_bound); o['e_bound_over_actual'] = float(e_bound / o['tR_e'])
    C1 = one_d_constants()
    o['bound_AM1'] = float((1 + 2 * np.tan(theta) ** 2) * (C1['A1'] + C1['B1'] * ln1) + 2 / np.cos(theta) ** 2 * e_bound)
    o['Bneeded_AM1'] = o['bound_AM1'] / ln1
    o['slope_AM1'] = float((1 + 2 * np.tan(theta) ** 2) * C1['B1'] + 2 / np.cos(theta) ** 2 * (16 * o['kappa_e'] / (1 - tau) + 16))   # coefficient of ln(1/tJ)
    return o
