"""Stage AL, plan A: anisotropic two-level chain.  The w-diagonal part is handled line by line (w-slices are 1-d heat-flowed measures),
only the mixed part H_vw needs the lever-arm chain, and for it the finer level is un-blurred in v only:
  level 1: rho_1 = rho * N(0, diag(t_v1, t)),  t_v1 = tau_v t;   final: p = rho_1 *_v N(0, t2), t2 = t - t_v1 (v only).
Exact identities (posterior of X1 given y lives on the line {x_w = y_w}):
  s_w(y) = E[sigma_w(X1v, y_w) | y],   d_v s_w = Cov(X1v, sigma_w | y)/t2,   c2v := d_v s_w + s_v s_w = E[(X1v - y_v) sigma_w | y]/t2.
Proposition AL1 (t = 1):
  (ww)  t E[H_ww^2] <= J (A1 + B1 ln(1/(tJ)))   (1-d theorem AE4 with the two-sided constant on every w-slice, Jensen over the slices;
        A1 = B1 ln(1/tau) + c0, B1 = K/(1 - tau), K = (1+eta)(1-tau)/(1-tau^{1+eta}), c0 = kappa[ln(1/tau)/(1-tau^{1+eta}) - 1/(1+eta)],
        eta = 0.1, kappa = 267.2, tau = 1/4)
  (vw)  E H_vw^2 <= 2 E c2v^2 + 2 E s_v^2 s_w^2;  E c2v^2 <= E_p[B A]/t2^2,  A = E[|sigma_w| | y], B = E[(X1v - y_v)^2 |sigma_w| | y];
        DV (beta = 1/(4 t2), E e^{beta Z^2} = sqrt 2):  E_p[B A] <= 4 t2 J_abs [ln(E W^2 / J_abs^2) + ln sqrt 2] <= 4 t2 J_abs [ln(3/(t^2 J_abs^2)) + ln sqrt 2];
        E s_v^2 s_w^2 <= 4 J [2 ln(1/(tJ)) + ln 3 + ln sqrt 2]/t^2.
  kappa_v := J_abs / J (sign cancellation of the w-score along the v-direction only).
Families: lines (v_i, ln r_i) uniform in w, points (v_j, w_j, L_j).  1-d Gauss-Hermite in v; log domain.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp
from ak_abs import closed_form, comp_logw, slog, lse

LN2PI = np.log(2 * np.pi)


def one_d_constants(eta=0.1, kappa=267.2, tau=0.25):
    K = (1 + eta) * (1 - tau) / (1 - tau ** (1 + eta)); c0 = kappa * (np.log(1 / tau) / (1 - tau ** (1 + eta)) - 1 / (1 + eta))
    B1 = K / (1 - tau); return {'K': K, 'c0': c0, 'B1': B1, 'A1': B1 * np.log(1 / tau) + c0}


def level1_logw(lines, points, Xv, Xw, tv1, t):
    A, M = [], []
    for v0, lr in lines:
        A.append(lr - (Xv - v0) ** 2 / (2 * tv1) - 0.5 * (LN2PI + np.log(tv1))); M.append(None)
    for v0, w0, L in points:
        A.append(-L - (Xv - v0) ** 2 / (2 * tv1) - (Xw - w0) ** 2 / (2 * t) - 0.5 * (LN2PI + np.log(tv1)) - 0.5 * (LN2PI + np.log(t))); M.append(w0)
    return np.stack(A), M


def chain(lines, points, yv, yw, tau_v=0.5, ngh=40, t=1.0):
    tv1 = tau_v * t; t2 = t - tv1; hv = np.gradient(yv); hw = yw[1] - yw[0] if len(yw) > 1 else 1.0
    gx, gwt = np.polynomial.hermite_e.hermegauss(ngh); lgw = np.log(gwt / np.sqrt(2 * np.pi))
    acc = {k: [] for k in ('M', 'J', 'Gvw', 'Gww', 'S2', 'Jabs', 'BA', 'W2', 'c2', 'id', 'twd')}
    nc = len(lines) + len(points); ch = max(1, int(2e6 / (nc * nc * ngh)))
    chunks = [slice(i, min(i + ch, len(yv))) for i in range(0, len(yv), ch)]
    for yw_, sl in [(a_, s_) for a_ in yw for s_ in chunks]:
        V_ = yv[sl]; W_ = np.full_like(V_, yw_); hvc = hv[sl]
        lp, lsw, ssw, ls2, lg, Cvw, Cww, sv = closed_form(lines, points, V_, W_, t)
        A_, M_ = comp_logw(lines, points, V_, W_, t); lP = A_ - lp[None]
        sd = np.sqrt(tv1 * t2 / t); X1v, LW = [], []
        for c, (v0, w0) in enumerate(M_):
            X1v.append((v0 + (tv1 / t) * (V_ - v0))[:, None] + sd * gx[None]); LW.append(lP[c][:, None] + lgw[None])
        X1v = np.concatenate(X1v, 1); LW = np.concatenate(LW, 1); X1w = np.broadcast_to(W_[:, None], X1v.shape)
        A1, M1 = level1_logw(lines, points, X1v, X1w, tv1, t); l1 = A1 - logsumexp(A1, 0)[None]
        terms = [(l1[c] + slog(M1[c] - X1w)[0], slog(M1[c] - X1w)[1]) for c in range(len(M1)) if M1[c] is not None]
        lsig, ssig = lse(np.stack([a for a, _ in terms]), 0, b=np.stack([b for _, b in terms])); lsig = lsig - np.log(t)
        lA = lse(LW + lsig, 1); dv = X1v - V_[:, None]
        with np.errstate(divide='ignore'):
            lB = lse(LW + 2 * np.log(np.abs(dv)) + lsig, 1)
            c2v = lse(LW + np.log(np.abs(dv)) + lsig, 1, b=np.sign(dv) * ssig)
            sq = lse(LW + lsig, 1, b=np.broadcast_to(ssig, LW.shape))
            lE2 = lse(LW + 2 * lsig, 1)
        c2v_val = c2v[1] * np.exp(c2v[0] - np.log(t2))
        c2v_closed = Cvw[1] * np.exp(Cvw[0]) + sv[1] * np.exp(sv[0]) * ssw * np.exp(lsw)
        lw = lp + np.log(hvc * hw)
        with np.errstate(divide='ignore'):
            acc['id'].append((lse(lw + np.log((c2v_val - c2v_closed) ** 2)), lse(lw + np.log(c2v_closed ** 2))))
            acc['twd'].append((lse(lw + np.log((sq[1] * np.exp(sq[0]) - ssw * np.exp(lsw)) ** 2)), lse(lw + 2 * lsw)))
        acc['M'].append(lse(lw)); acc['J'].append(lse(lw + 2 * lsw))
        acc['Gvw'].append(lse(lw + 2 * Cvw[0])); acc['Gww'].append(lse(lw + 2 * Cww[0]))
        acc['S2'].append(lse(lw + 2 * sv[0] + 2 * lsw)); acc['Jabs'].append(lse(lw + 2 * lA))
        acc['BA'].append(lse(lw + lB + lA - 2 * np.log(t2))); acc['c2'].append(lse(lw + 2 * c2v[0] - 2 * np.log(t2)))
        acc['W2'].append(lse(lw + 2 * lA + lE2))
    L = {k: float(lse(np.array(v))) for k, v in acc.items() if k not in ('id', 'twd')}
    ir = lambda k: float(np.sqrt(np.exp(lse(np.array([a for a, _ in acc[k]])) - lse(np.array([b for _, b in acc[k]])))))
    lM = L['M']; ln1 = -(L['J'] - lM) - np.log(t)
    o = {'tau_v': tau_v, 'identity_c2v_rel_L2': ir('id'), 'tweedie_rel_L2': ir('twd'), 'ln1_tJ': float(ln1)}
    o['tR'] = float(t * (np.exp(L['Gvw'] - L['J']) + np.exp(L['Gww'] - L['J']))); o['tR_vw'] = float(t * np.exp(L['Gvw'] - L['J'])); o['tR_ww'] = float(t * np.exp(L['Gww'] - L['J']))
    o['kappa_v'] = float(np.exp(L['Jabs'] - L['J']))
    Gvw = L['Gvw']
    o['vw_step_split'] = float(2 * np.exp(L['c2'] - Gvw) + 2 * np.exp(L['S2'] - Gvw))
    o['vw_step_CS'] = float(2 * np.exp(L['BA'] - Gvw) + 2 * np.exp(L['S2'] - Gvw))
    Jabs_n = L['Jabs'] - lM; KLj = float(L['W2'] - lM - 2 * Jabs_n); KLf = float(np.log(3 / t ** 2) - 2 * Jabs_n)
    s2b = 4 * np.exp(L['J'] - Gvw) * (2 * ln1 + np.log(3) + 0.5 * np.log(2)) / t ** 2
    o['vw_step_DV_jensen'] = float(2 * 4 * np.exp(L['Jabs'] - Gvw) / t2 * (KLj + 0.5 * np.log(2)) + 2 * np.exp(L['S2'] - Gvw))
    o['vw_step_final'] = float(2 * 4 * np.exp(L['Jabs'] - Gvw) / t2 * (KLf + 0.5 * np.log(2)) + 2 * s2b)
    C1 = one_d_constants(); o['ww_bound'] = float(C1['A1'] + C1['B1'] * ln1)
    o['vw_bound'] = float(o['vw_step_final'] * o['tR_vw'])
    o['bound_AL1'] = o['ww_bound'] + o['vw_bound']; o['Bneeded_AL1'] = o['bound_AL1'] / ln1
    o['Bneeded_vw_only'] = o['vw_bound'] / ln1
    return o
