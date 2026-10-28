"""Stage AP: two independent checks of the two-scale formulas for confined hidden multipoles.

(1) Direct 2-d summation (t = 1 units, closed-form blurred components; no v-reduction): lines at v = -D, +D with w-envelope N(0, l^2),
    rows at x_i with profile (1 + f_i cos(k w)) on the same envelope.  Blurred row = m phi(v - x) N(w; 0, l^2 + 1)
    [1 + f e^{-k^2 l^2/(2(l^2+1))} cos(k l^2 w/(l^2 + 1))].  Gives Lambda = E[(s_v s_w + 2 d_v s_w)^2]/E[s_w^2], J_ww, J_vv on a grid.
(2) True gap of paper E (v3_tools.grid_sums, the cancellation-free pendulum formula with N) on the same state in physical units, sigma = 1e-4.
Both are compared with the two-scale formulas of ap_confined.py:
    sigma^2 J_ww = eps + tJmp,  Lambda = (Jvv eps + Lam_v tJmp)/(eps + tJmp),  eps = sigma^2/(Lw^2 + sigma^2),
    gap/sigma = (2 sigma^2 J_ww + N)/(sigma (Jvv + sigma^2 J_ww)),  N = (1/2)(1 - e^{-Lw^2/4})(Lw^2/(Lw^2 + sigma^2) - 1 + Jvv)
    (N = -sigma^-2 E Cov(x_p, x_z - sin x_z | y); posterior variances Var w = sigma^2 Lw^2/(Lw^2 + sigma^2), E Var v = sigma^2 (1 - Jvv) by Tweedie).
-> results/ap_direct.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, os
import numpy as np
from an_signed1d import quantities
from an_multipole import realise

LN2PI = np.log(2 * np.pi)


def two_scale(xs, lw, f, eps, k=1.0):
    o = quantities(xs, lw, f); tJmp = (k * k * np.exp(-k * k) / 2) * np.exp(-o['lnMI'])
    return {'Jww': eps + tJmp, 'Jvv': o['Jvv'], 'Lambda': (o['Jvv'] * eps + o['Lam_v'] * tJmp) / (eps + tJmp), 'tJmp': tJmp, 'Lam_v': o['Lam_v'], 'lnMI': o['lnMI']}


def direct2d(xs, lw, f, ell, k=1.0, hv=None, per=24, nsd=6.0):
    """t = 1.  xs, lw, f: v-positions, log masses, signs (f = 0: line)."""
    s2 = ell * ell + 1; amp = np.exp(-k * k * ell * ell / (2 * s2)); kk = k * ell * ell / s2
    hv = hv or min(0.01, 0.05 / max(1.0, np.ptp(xs) / 2)); vs = np.arange(xs.min() - 8, xs.max() + 8, hv)
    ws = np.arange(-nsd * np.sqrt(s2), nsd * np.sqrt(s2), 2 * np.pi / kk / per)
    lwr = lw - lw.max(); acc = np.zeros(4)
    W = ws[None, :]; env = np.exp(-W ** 2 / (2 * s2)); denv = -W / s2
    for i0 in range(0, len(vs), max(1, int(2e6 // len(ws)))):
        V = vs[i0:i0 + int(2e6 // len(ws))][:, None]
        p = 0.0; pv = 0.0; pw = 0.0; pvw = 0.0
        for x, l_, ff in zip(xs, lwr, f):
            a = np.exp(l_ - (V - x) ** 2 / 2); av = -(V - x) * a                          # v-factor and its derivative
            c = 1 + ff * amp * np.cos(kk * W); cw = -ff * amp * kk * np.sin(kk * W)           # w-modulation
            bw = env * c; bww = env * (denv * c + cw)                                         # w-factor (with envelope) and d_w
            p = p + a * bw; pv = pv + av * bw; pw = pw + a * bww; pvw = pvw + av * bww
        sv, sw = pv / p, pw / p; dvsw = pvw / p - pv * pw / p ** 2
        acc += [np.sum(p * sw ** 2), np.sum(p * (sv * sw + 2 * dvsw) ** 2), np.sum(p * sv ** 2), np.sum(p)]
    return {'Jww': acc[0] / acc[3], 'Lambda': acc[1] / acc[0], 'Jvv': acc[2] / acc[3], 'n_points': len(vs) * len(ws)}


def v3_gap(xs, lw, f, sig, Lw, k=1.0, ev=1e-3):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
    from v3_tools import grid_sums
    lwr = lw - np.log(np.sum(np.exp(lw - lw.max()))) - lw.max()                          # normalised masses
    S = np.diag([(ev * sig) ** 2, Lw ** 2]); T = []
    for x, l_ in zip(xs, lwr): T.append((l_, np.array([x * sig, 0.0]), S, np.zeros(2)))
    for x, l_, ff in zip(xs, lwr, f):
        if ff == 0: continue
        lc = l_ + np.log(abs(ff) / 2) + (0j if ff > 0 else 1j * np.pi)
        for sgn in (1, -1): T.append((lc, np.array([x * sig, 0.0]), S, np.array([0.0, sgn * k / sig])))
    hv = min(0.0125, 0.1 / max(xs.max(), 1.0)) * sig
    vn = np.arange(xs.min() * sig - 8 * sig, xs.max() * sig + 8 * sig, hv); wn = np.arange(-6 * Lw, 6 * Lw, 2 * np.pi / k * sig / 16)
    o = grid_sums(T, sig, vn, wn, rows=16)
    return {'gap_over_sigma': o['gap_over_sigma'], 'sig2Jww': sig ** 2 * o['J_ww'], 'sig2Jvv': sig ** 2 * o['J_vv'], 'N': o['N_ibp'], 'n_points': o['n_points']}


if __name__ == '__main__':
    out = {'lambda': {}, 'gap': {}}
    M = json.load(open('results/an_multipole.json'))
    # cases where the multipole carries a sizeable share of J_ww (depth matched to the envelope: ln(1/tJmp) ~ ln(1/eps))
    for n, D, d, z, ell in (('2', 2.0, 0.5, 0.5, 100.0), ('2', 3.0, 0.5, 0.5, 1000.0)):
        L = M['leading'][n]; xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max()
        ts = two_scale(xs, lw, f, 1 / (ell * ell + 1)); dr = direct2d(xs, lw, f, ell, per=12)
        key = 'n%s_D%g_l%g' % (n, D, ell)
        out['lambda'][key] = {'two_scale': ts, 'direct': dr, 'rel_Lambda': abs(dr['Lambda'] / ts['Lambda'] - 1), 'rel_Jww': abs(dr['Jww'] / ts['Jww'] - 1)}
        print('%s: Lambda two-scale %.4f direct %.4f (rel %.1e); J_ww %.4e vs %.4e (rel %.1e); multipole share of J_ww %.2f' % (
            key, ts['Lambda'], dr['Lambda'], out['lambda'][key]['rel_Lambda'], ts['Jww'], dr['Jww'], out['lambda'][key]['rel_Jww'], ts['tJmp'] / ts['Jww']), flush=True)
        json.dump(out, open('results/ap_direct.json', 'w'), indent=1, default=float)
    sig = 1e-4
    for n, D, d, z, Lw_rt in (('0', 8.0, 1.0, 6.0, 2.0), ('2', 2.0, 0.5, 1.0, 2.0), ('2', 2.0, 0.5, 1.0, 5.0)):
        Lw = Lw_rt * np.sqrt(sig); eps = sig ** 2 / (Lw ** 2 + sig ** 2)
        if n == '0':                                     # the two lines alone
            xs, lw, f = np.array([-D, D]), np.array([0.0, 0.0]), np.array([0.0, 0.0]); ts = {'Jww': eps, 'Jvv': 1.0, 'tJmp': 0.0}
        else:
            L = M['leading'][n]; xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max(); ts = two_scale(xs, lw, f, eps)
        Nf = 0.5 * (1 - np.exp(-Lw ** 2 / 4)) * (Lw ** 2 / (Lw ** 2 + sig ** 2) - 1 + ts['Jvv'])     # Tweedie: E Var(x_v|y) = sigma^2 (1 - sigma^2 J_vv)
        g_ts = (2 * ts['Jww'] + Nf) / (sig * (ts['Jvv'] + ts['Jww']))
        gv = v3_gap(xs, lw, f, sig, Lw)
        key = 'n%s_D%g_Lw%gsqrtsig' % (n, D, Lw_rt)
        out['gap'][key] = {'two_scale_gap': g_ts, 'v3': gv, 'N_formula': Nf, 'rel_gap': abs(gv['gap_over_sigma'] / g_ts - 1)}
        print('%s: gap/sigma v3 %.5f  two-scale %.5f (rel %.1e); N v3 %.4e formula %.4e; sigma^2 J_ww %.3e (multipole %.1e)' % (
            key, gv['gap_over_sigma'], g_ts, out['gap'][key]['rel_gap'], gv['N'], Nf, gv['sig2Jww'], ts['tJmp']), flush=True)
        json.dump(out, open('results/ap_direct.json', 'w'), indent=1, default=float)
