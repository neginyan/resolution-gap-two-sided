"""Stage AX: exact closed forms for the strip (Lemmas AX1, AX2) and the refined N_phi bound (Lemma AX3), checked on exact point fields.

With the factorisation of Lemma AW1, p = N_V(v) Gamma(w)[1 + q(w) cos(k'v)]:
 Lemma AX1 (proof):  J_ww = 1/L^2 + int Gamma q'^2 (1/sqrt(1-q^2) - 1)/q^2 dw          (avg_t cos^2 t/(1 + q cos t) = (1/sqrt(1-q^2) - 1)/q^2)
                     hence sigma^2/L^2 + sigma^2 B^2/2 <= sigma^2 J_ww <= sigma^2/L^2 + sigma^2 B^2 (1/sqrt(1-qbar^2) - 1)/qbar^2,  B^2 = int Gamma q'^2 = u/(4L^2).
 Lemma AX2 (proof):  J_vv = 1/V + J_vv,per,  J_vv,per = k'^2 int Gamma (1 - sqrt(1 - q^2)) dw     (avg q^2 sin^2/(1 + q cos) = 1 - sqrt(1-q^2))
                     hence x I_q/2 <= W_p := sigma^2 J_vv,per <= x I_q/(1 + sqrt(1 - qbar^2)),  I_q = int Gamma q^2 = q0^2 s/sqrt(s^2 + 2L^2),  x = (k' sigma)^2.
 Lemma AX3 (proof):  <G> = -J_vv,per (integration by parts in v over one period; E_P[s~ s_w] = 0 by parity), so in Lemma AW6
                     E[s_v^2 phi] >= (1 - e - sqrt2 a e sigma^2) J_vv,per + (1 - C)(1/V - gamma)   (C >= 0),  gamma = qbar (x(1+qbar) + sqrt x)/(1-qbar)^2,
                     i.e.  N_phi >= (1/2)[(1 - e - sigma^2) W_p + (1 - C)(b - sigma^2 gamma)],  b = sigma^2/V.
-> results/ax_exact.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from v3_tools import _pre
from u_cgm import trap_weights
from at_tools import point_fields2
from aw_lemmas import profile
from aq_hybrid import strip_params

KEYS = ('svv', 'sww', 'st2', 'G', 'phi_svv', 'omC')


def evaluator(sig, k0, lam, M, c, eta, nper=50, coarse=400, n0=31, nv=64, ext=6.5):
    Lv_sig = eta * lam / np.sqrt(sig); bg = [(1.0, 0.0, lam * M)]; win = [(c, 0.0, lam, 1)]
    T = terms_of(bg, win, k0, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse); Lv = Lv_sig * sig; V = Lv ** 2 + sig ** 2; s2 = sig * sig
    pre = _pre(T, sig); per = 2 * np.pi * sig / k0; ww = trap_weights(wn); v0s = np.linspace(-ext * np.sqrt(V), ext * np.sqrt(V), n0); w0 = trap_weights(v0s)
    acc = {k: 0.0 for k in KEYS}; m = 0.0; npts = max(256, int(2.5e6 // max(1, len(pre)))); a = (1 - s2 / V) / np.sqrt(2)
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2
        V_, W_ = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V_.ravel(), W_.ravel()], 1); dA = cw * np.outer(np.full(nv, 1.0 / nv), ww).ravel()
        for i0 in range(0, len(X), npts):
            x = X[i0:i0 + npts]; p, q = point_fields2(pre, sig, x); pw = p * dA[i0:i0 + npts]; m += pw.sum(); v = x[:, 0]
            sv = q['sv']; stl = sv + v / V; R = np.sqrt(2) * q['mz'] - a * np.sqrt(2) * v
            vals = {'svv': sv * sv, 'sww': q['sww'], 'st2': stl * stl, 'G': -q['aHvv'] / s2 + 1 / V - q['Hvw'], 'phi_svv': sv * sv * 2 * np.sin(q['mz'] / 2) ** 2,
                    'omC': 2 * np.sin(R / (2 * np.sqrt(2))) ** 2}
            for k in KEYS: acc[k] += float(np.sum(pw * vals[k]))
    return {k: v / m for k, v in acc.items()}, V, a


def closed(P, nw=200001):
    L = np.sqrt(P['L2']); s = np.sqrt(P['s2']); w = np.linspace(-12 * L, 12 * L, nw); w = np.unique(np.concatenate([w, np.linspace(-12 * s, 12 * s, nw)]))
    G = np.exp(-w ** 2 / (2 * P['L2'])) / np.sqrt(2 * np.pi * P['L2']); q = P['q0'] * np.exp(-w ** 2 / (2 * P['s2'])); dq = -w / P['s2'] * q
    wt = np.gradient(w); f = np.where(q > 1e-8, (1 / np.sqrt(1 - q * q) - 1) / np.maximum(q * q, 1e-300), 0.5 + 0.375 * q * q)
    Jww = 1 / P['L2'] + np.sum(wt * G * dq ** 2 * f)
    g1 = np.where(q > 1e-8, 1 - np.sqrt(1 - q * q), q * q / 2 + q ** 4 / 8)
    Jper = (P['kp_sigma'] / 1.0) ** 2 * np.sum(wt * G * g1)             # in units sigma^-2: J_vv,per sigma^2 = x int Gamma (1 - sqrt(1-q^2))
    Iq = P['q0'] ** 2 * s / np.sqrt(s * s + 2 * P['L2']); B2 = P['q0'] ** 2 * P['L2'] / (s ** 4 * (1 + 2 * P['L2'] / P['s2']) ** 1.5)
    return {'Jww': Jww, 'Wp': Jper, 'Iq': Iq, 'B2': B2, 'Iq_num': np.sum(wt * G * q * q), 'B2_num': np.sum(wt * G * dq * dq)}


if __name__ == '__main__':
    cases = {}
    for s in (1e-12, 1e-24):
        P = strip_params(s); cases['opt_%g' % s] = (s, P['k0'], P['lam'], P['M'], P['c'], P['Lv_sig'] * np.sqrt(s) / P['lam'])
    cases['seq_1e-18'] = (1e-18, 2.792, 2 / np.sqrt(2.792), 2.792 ** 2 * np.exp(2.792 ** 2), 1.0, 1 / 2.792 ** 2)
    cases['grid_1e-12_k2.4'] = (1e-12, 2.4, 1.3, 1e3, 1.0, 0.1)
    out = {}
    for name, args in cases.items():
        sig = args[0]; P = profile(*args); E, V, a = evaluator(*args); C = closed(P); s2 = sig * sig; x = P['kp_sigma'] ** 2; qb = P['qbar']
        e = np.exp(-a * a * V / 2); ome = -np.expm1(-a * a * V / 2); gam = qb * (x * (1 + qb) + np.sqrt(x)) / (1 - qb) ** 2
        Jper_ev = E['st2']; Wp_ev = s2 * Jper_ev; Jww_ev = E['sww']
        Nphi = s2 / 2 * E['phi_svv']; b = s2 / V
        Nphi_lb = 0.5 * ((ome - s2) * Wp_ev + E['omC'] * (b - s2 * gam))
        r = {'Jww_rel': abs(C['Jww'] / Jww_ev - 1), 'Wp_rel': abs(C['Wp'] / Wp_ev - 1), 'G_mean_over_minus_Jper': E['G'] / (-Jper_ev - 0) , 'Iq_rel': abs(C['Iq'] / C['Iq_num'] - 1), 'B2_rel': abs(C['B2'] / C['B2_num'] - 1),
             'Jww_lower_ok': bool(s2 * Jww_ev >= s2 / P['L2'] + s2 * C['B2'] / 2 * (1 - 1e-9)), 'Wp_bracket_ok': bool(x * C['Iq'] / 2 * (1 - 1e-6) <= Wp_ev <= x * C['Iq'] / (1 + np.sqrt(1 - qb * qb)) * (1 + 1e-6)),
             'Nphi': Nphi, 'Nphi_lower_bound': Nphi_lb, 'Nphi_bound_ok': bool(Nphi >= Nphi_lb), 'Nphi_bound_ratio': Nphi_lb / Nphi, 'qbar': qb, 'x': x, 'gamma': gam}
        out[name] = r
        print('%-16s J_ww closed vs evaluator %.1e | W_p %.1e | <G>/(-J_per) %.6f | I_q %.1e B^2 %.1e | brackets: J_ww %s W_p %s | N_phi %.4e >= %.4e (%s, ratio %.3f)' % (
            name, r['Jww_rel'], r['Wp_rel'], r['G_mean_over_minus_Jper'], r['Iq_rel'], r['B2_rel'], r['Jww_lower_ok'], r['Wp_bracket_ok'], Nphi, Nphi_lb, r['Nphi_bound_ok'], r['Nphi_bound_ratio']), flush=True)
    json.dump(out, open('results/ax_exact.json', 'w'), indent=1, default=float)
