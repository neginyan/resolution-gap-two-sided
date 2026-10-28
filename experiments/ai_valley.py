"""Stage AI, route 1/2 diagnostics on the w-line + point family (t = 1): the deep-valley weight against tJ and tJ ln(1/tJ).
Deep valley: lambda_max(-tH) >= theta (theta = 1, 3).  Also the exact sandwich of Lemma AI1 with N = (K - I)_+.  Measured (all per unit line density, normalised as in ai_mix):
  P_val  = E[1{valley}],  N_val = t^2 E[(H^2)_ww 1{valley}] (the part of tR*tJ carried by the valley),
  and the ratios P_val/(tJ), N_val/(tJ ln(1/tJ)), N_val/(tJ).  Log-log slopes in D give the exponents.  -> results/ai_valley.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from ai_mix import state_fields
from scipy.special import logsumexp

def one(D, z, thetas=(1.0, 3.0)):
    L = D * (z + D / 2); lp, h, sv, sw, Hvv, Hvw, Hww = state_fields([(0.0, 0.0)], [(D, 0.0, L)], h=0.04)
    lnM = float(logsumexp([0.0, -L])); c = lp.max(); W = np.exp(lp - c) * h * h
    J = np.sum(W * sw ** 2); lnJ = np.log(J) + c - lnM; lnJ_val = lnJ
    tr = Hvv + Hww; det = Hvv * Hww - Hvw ** 2; lam_min_H = tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0)); m = -lam_min_H   # lambda_max(-H)
    r = {'D': D, 'z': z, 'ln1_tJ': -lnJ_val, 'tR': float(np.sum(W * (Hvw ** 2 + Hww ** 2)) / J)}
    # split tH = P - N (P, N >= 0, PN = 0): t^2 (H^2)_ww = (P^2)_ww + (N^2)_ww, N = (K - I)_+ the excess posterior covariance (valleys).
    # Lower bound E[(N^2)_ww]/tJ <= tR is exact.  The upper bound 1 + E[N_ww + (N^2)_ww]/tJ (from P^2 <= P) is USELESS: N_ww is first
    # order in the hidden weight pi (K_ww - 1 ~ pi (y_w^2 - 1) on the line side) while tJ is second order, so it diverges (recorded).
    a_, b_, c_ = -Hvv, -Hvw, -Hww                        # -tH (t = 1), eigen-decomposition 2x2
    tr2 = (a_ + c_) / 2; rad = np.sqrt(((a_ - c_) / 2) ** 2 + b_ ** 2); lam = tr2 + rad
    ux = np.where(np.abs(b_) > 1e-300, b_, np.where(a_ >= c_, 1.0, 0.0)); uy = np.where(np.abs(b_) > 1e-300, lam - a_, np.where(a_ >= c_, 0.0, 1.0))
    nn = np.sqrt(ux ** 2 + uy ** 2); uw = uy / nn                # w-component of the top eigenvector of -tH
    lp_ = np.maximum(lam, 0.0); lam2 = tr2 - rad; lp2 = np.maximum(lam2, 0.0); uw2sq = 1 - uw ** 2
    Nww = lp_ * uw ** 2 + lp2 * uw2sq; N2ww = lp_ ** 2 * uw ** 2 + lp2 ** 2 * uw2sq
    EN = np.sum(W * Nww) / J; EN2 = np.sum(W * N2ww) / J
    r['N2/tJ'] = float(EN2); r['P2/tJ'] = r['tR'] - float(EN2); r['lower_ok'] = bool(EN2 <= r['tR'] * (1 + 1e-9))
    r['N/tJ (upper bound term, diverges)'] = float(EN); r['N2/(tJ ln)'] = float(EN2 / r['ln1_tJ'])
    for th in thetas:
        I = m >= th; Pv = np.sum(W * I); Nv = np.sum(W * I * (Hvw ** 2 + Hww ** 2))
        r['P_val/tJ_%g' % th] = float(Pv / J); r['N_val/tJ_%g' % th] = float(Nv / J)
        r['N_val/(tJ ln)_%g' % th] = float(Nv / J / r['ln1_tJ']); r['depth_%g' % th] = float(m.max())
    return r

if __name__ == '__main__':
    out = {}
    for zr in (0.0, 0.25, 0.5, 1.0):
        rows = [one(D, zr * D) for D in ((4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0) if zr < 1 else (4.0, 6.0, 8.0, 10.0, 12.0, 16.0))]
        sl = {}
        for k in rows[0]:
            if k in ('D', 'z') or isinstance(rows[0][k], bool): continue
            y = np.array([r[k] for r in rows[2:]]); x = np.log([r['D'] for r in rows[2:]])
            if np.all(y > 0) and np.all(np.isfinite(y)): sl[k] = float(np.polyfit(x, np.log(y), 1)[0])
        out['z/D=%g' % zr] = {'rows': rows, 'slopes_D>=8': sl}
        print('z/D=%g' % zr); [print({k: round(v, 4) for k, v in r.items()}) for r in rows]; print('slopes', {k: round(v, 2) for k, v in sl.items()})
    json.dump(out, open('results/ai_valley.json', 'w'), indent=1)
