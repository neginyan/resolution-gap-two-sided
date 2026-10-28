"""Stage AV: the strip family's asymptotics from above (an explicit sequence) and the remainders of the leading-order formula.

Strip (paper E): rho = N(v;0,Lv^2) N(w;0,Lw^2)(1 + c e^{-w^2/2 ell^2} cos(k0 v/sigma)), ell = lam sqrt(sigma), Lw = M ell, Lv = eta ell.
Leading order (stage AS; as_strip_lo.g_lo):  g_LO^2-form  g = [2 CR/(M^2 lam^2 W) + lam^2 (1/2 + eta^2 + r M^2)/8] / (1 + r),
  W = c^2 k0^2 e^{-k0^2}/(2 sqrt2 M),  CR = 1 + (sqrt2/8) c^2 e^{-k0^2} M,  r = sigma/(eta^2 lam^2 W);  at the optimal lam and r -> 0:
  g = (1/(2 k0)) sqrt((1 + a X)(1 + 2 eta^2 + 2 r M^2)) / (1 + r) (approximately),  a = 4 sqrt2/c^2,  X = e^{k0^2}/M.
Explicit sequence S(sigma) (construction):  c = 1,  k0 from 4 k0^2 + 10 ln k0 = ln(1/sigma),  M = k0^2 e^{k0^2} (X = 1/k0^2),  eta = 1/k0^2,
  lam = 2/sqrt(k0)  ->  r M^2 = O(1/k0^2), so  g_LO sqrt(ln 1/sigma) = sqrt(ln(1/sigma)/(4 k0^2)) (1 + O(1/k0^2)) -> 1.
The TRUE gap (paper E's evaluator: point-chunked w_full with exact point fields, true V-envelope; at_tools.sums_envelope adds the AT2 factors)
is computed where the positional precision allows (Lv/period < 1e12: sigma >= ~1e-24).  -> results/av_strip.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from at_tools import sums_envelope, analyse
from as_strip_lo import g_lo
from scipy.optimize import brentq


def seq_params(sig=None, L=None):
    L = np.log(1 / sig) if L is None else L; k0 = brentq(lambda k: 4 * k * k + 10 * np.log(k) - L, 0.3, 1000.0)
    return {'k0': k0, 'c': 1.0, 'M': k0 * k0 * np.exp(min(k0 * k0, 700.0)), 'eta': 1 / k0 ** 2, 'lam': 2 / np.sqrt(k0)}


def true_strip(sig, P, nper=50, coarse=400, n0=31, nv=64):
    lam, M, k0, c = P['lam'], P['M'], P['k0'], P['c']; Lv_sig = P['eta'] * lam / np.sqrt(sig)
    bg = [(1.0, 0.0, lam * M)]; win = [(c, 0.0, lam, 1)]
    T = terms_of(bg, win, k0, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    r = analyse(sums_envelope(T, sig, k0, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv, n0=n0), sig)
    r['g_LO'] = g_lo(sig, k0, lam, M, Lv_sig * sig, c); r['LO_rel_err'] = r['g_LO'] / r['g'] - 1
    return r


if __name__ == '__main__':
    fn = 'results/av_strip.json'; R = json.load(open(fn)) if os.path.exists(fn) else {'sequence': {}, 'sequence_LO': {}, 'remainder': {}}
    # (1) the sequence at leading order, far into small sigma
    for e in (6, 12, 24, 48, 96, 192, 384, 768, 1536):
        sig = 10.0 ** -e if e < 300 else None; L = e * np.log(10); P = seq_params(L=L)
        k0 = P['k0']; X = 1 / k0 ** 2; a = 4 * np.sqrt(2)
        # leading-order value in log-safe form (as_strip_lo.g_lo_log)
        from as_strip_lo import g_lo_log
        g = g_lo_log(-L, k0, np.log(P['lam']), 2 * np.log(k0) + k0 * k0, np.log(P['eta']), 1.0)     # ln M = 2 ln k0 + k0^2 (no overflow)
        R['sequence_LO']['1e-%d' % e] = {'k0': k0, 'g_LO': float(g), 'g_LO_sqrt_ln': float(g * np.sqrt(L)), 'k0sq_over_ln': k0 * k0 / L}
        print('sequence LO sigma 1e-%-5d k0 %.3f  g_LO %.5f  g_LO sqrt(ln) %.4f  k0^2/ln %.4f' % (e, k0, g, g * np.sqrt(L), k0 * k0 / L), flush=True)
    json.dump(R, open(fn, 'w'), indent=1, default=float)
    # (2) the sequence with the TRUE evaluator
    for sig in (1e-6, 1e-12, 1e-18, 1e-24):
        key = '%g' % sig
        if key in R['sequence']: continue
        P = seq_params(sig); t0 = time.time(); r = true_strip(sig, P); r.update(P); r['sigma'] = sig; r['g_sqrt_ln'] = r['g'] * np.sqrt(np.log(1 / sig)); r['seconds'] = time.time() - t0
        R['sequence'][key] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
        print('sequence TRUE sigma %-6g k0 %.3f M %.3g: g %.5f (LO %.5f, rel %.1e)  g sqrt(ln) %.4f | AT2 %.5f  k_eff/k0 %.4f  C_w %.5f  %.0fs' % (
            sig, P['k0'], P['M'], r['g'], r['g_LO'], r['LO_rel_err'], r['g_sqrt_ln'], r['bound_AT2'], r['k_eff'] / P['k0'], r['C_w'], r['seconds']), flush=True)
    # (3) remainders of the leading order: one small parameter at a time around a base state at sigma = 1e-12
    sig = 1e-12; base = {'k0': 2.4, 'c': 1.0, 'M': 3e3, 'eta': 0.1, 'lam': 1.3}
    scans = {'M': [3e2, 1e3, 3e3, 1e4, 3e4], 'k0': [1.6, 2.0, 2.4, 2.8, 3.2], 'eta': [0.4, 0.2, 0.1, 0.05, 0.025]}
    for par, vals in scans.items():
        for v in vals:
            key = '%s=%g' % (par, v)
            if key in R['remainder']: continue
            P = dict(base); P[par] = v
            if par == 'k0': P['M'] = 3e3 * np.exp(v * v - 2.4 * 2.4)          # keep X = e^{k0^2}/M fixed
            r = true_strip(sig, P); r.update(P); r['q'] = P['c'] * np.exp(-P['k0'] ** 2 / 2); R['remainder'][key] = r
            json.dump(R, open(fn, 'w'), indent=1, default=float)
            print('remainder %-10s g %.6f LO %.6f rel %+.2e  (1/M %.1e, q %.1e, eta^2 %.1e)' % (key, r['g'], r['g_LO'], r['LO_rel_err'], 1 / P['M'], r['q'], P['eta'] ** 2), flush=True)
