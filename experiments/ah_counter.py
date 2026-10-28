"""Stage AH: the counterexample candidate (infinite w-line + one point, and its Gaussian-envelope probability-measure versions).
For D (point distance, units sqrt t) and z (L = ln(1/eps) = D (z + D/2): the valley sits at y_v* = D/2 + L/D = D + z):
  Y (A = 3, a0 = 1/2), kappa_H, tR, t^2 R', Lambda = t E[(s_v s_w - 2 H_vw)^2]/J_ww, ln(1/(t J_ww)), and the integrated (qLC_w) ratio
  K_needed = tR(t) (1 - tau)/ln(J(tau t)/J(t)) at tau = 1/4 (c0 = 0).
Finite versions: line with envelope N(0, V) (mass 1 - m) and point mass m = e^{-L} rho_0 (rho_0 = 1/sqrt(2 pi V), the line density at w = 0),
so that locally the state is the infinite-line one; V chosen so that the envelope's Fisher information 1/V is a fraction f_env of the point's.  -> results/ah_counter.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from ah_lp3 import fields, evaluate


def extras(D, L, t=1.0, V=np.inf):
    lp, h, sv, sw, Hvv, Hvw, Hww, _ = fields(D, L, t, V); c = lp.max(); W = np.exp(lp - c) * h * h
    Jw = np.sum(W * sw * sw); q2 = np.sum(W * (sv * sw - 2 * Hvw) ** 2)
    if not np.isinf(V):
        le = np.log1p(-np.exp(-L)) - c; Jw += np.exp(le - np.log(V + t)); q2 += 0.0      # the envelope: s_v s_w - 2 H_vw ~ s_v (-w/(V+t)): O(1/V), negligible
    return {'Lambda': t * q2 / Jw, 'lnJ_abs': float(np.log(Jw) + c), 'J_abs_point_window': float(np.exp(np.log(np.sum(W * sw * sw)) + c))}


def row(D, z, V=np.inf, tau=0.25):
    L = D * (z + D / 2)
    if not np.isinf(V): L = L + 0.5 * np.log(2 * np.pi * V)       # point mass e^{-L} rho_0, rho_0 = line density at w = 0 (same local state as the infinite line)
    d = evaluate(D, L, 1.0, V); e = extras(D, L, 1.0, V)
    # J at tau t: same state, blur tau (positions fixed): fields use t directly
    e2 = extras(D, L, tau, V); lnJ1 = e['lnJ_abs']; lnJt = e2['lnJ_abs']
    return {'D': D, 'z': z, 'L': L, 'V': V, 'Y3': d['A3_a0.5']['Y'], 'kappa_H': d['kappa_H'], 'tR': d['tR'], 't2R1': d['t2R1'],
            'Lambda': e['Lambda'], 'ln_1_over_tJ': -lnJ1, 'K_needed_tau0.25': d['tR'] * (1 - tau) / (lnJt - lnJ1)}


if __name__ == '__main__':
    out = {'infinite_line': [], 'finite': []}
    for D in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        for z in (0.0, 0.25 * D, 0.5 * D):
            r = row(D, z); out['infinite_line'].append(r)
            print('D %4.1f z %5.1f  Y3 %8.2f  kH %8.2f  tR %6.2f  t2R1 %7.2f  t2R1/tR %5.2f  Lambda %7.2f  ln(1/tJ) %7.2f  Lambda/ln %.3f  K_needed %.3f' % (
                D, z, r['Y3'], r['kappa_H'], r['tR'], r['t2R1'], r['t2R1'] / r['tR'], r['Lambda'], r['ln_1_over_tJ'], r['Lambda'] / r['ln_1_over_tJ'], r['K_needed_tau0.25']), flush=True)
    # probability measures: Gaussian envelope with the envelope information = f_env x the point's
    for D, z in ((10.0, 0.0), (10.0, 5.0), (16.0, 0.0)):
        L = D * (z + D / 2); Jpt = extras(D, L)['J_abs_point_window']
        for f_env in (1.0, 0.1, 0.01):
            V = 2 * np.pi / (f_env * Jpt) ** 2                # envelope information 1/V = f_env x the point's (rho_0 Jpt, rho_0 = 1/sqrt(2 pi V))
            r = row(D, z, V); r['f_env'] = f_env; r['lnV'] = float(np.log(V)); out['finite'].append(r)
            print('  finite: D %4.1f z %4.1f  ln V %6.1f (f_env %.2f)  Y3 %8.2f  kH %8.2f  tR %6.2f  t2R1 %7.2f  K_needed %.3f' % (D, z, r['lnV'], f_env, r['Y3'], r['kappa_H'], r['tR'], r['t2R1'], r['K_needed_tau0.25']), flush=True)
    json.dump(out, open('results/ah_counter.json', 'w'), indent=1, default=float)
