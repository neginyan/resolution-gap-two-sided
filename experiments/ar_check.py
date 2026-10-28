"""Stage AR: Theorem AR2 (the gap is protected in the common-envelope class) checked factor by factor on exact point fields (v3_tools).

Paper E, Theorem 2 (W'):  g = gap/sigma >= (a/den) sqrt(P_w) (1 - eps),  a = sigma^2 J_vv, den = sigma^2 (J_vv + J_ww),
   P_w = 4 sigma^2 E[s_w^2 (1 - phi/4)] E[s_v^2 phi] / a^2,  phi = 1 - cos m_z,  eps = -(B + R3)/(sigma a sqrt(P_w)).
Definitions (exact):  Phi = E[phi],  delta_c = 1 - E[s_v^2 phi]/(J_vv Phi),  delta_w = 1 - E[s_w^2 (1 - phi/4)]/(J_ww (1 - Phi/4)).
Cramer-Rao in w:  sigma^2 J_ww >= sigma^2/(Var_w(rho) + sigma^2).  Hence (Theorem AR2)
   P_w >= kappa^2 / (sigma^2 J_vv),   kappa^2 = 4 Phi (1 - Phi/4)(1 - delta_c)(1 - delta_w) / (Var_w(rho) + sigma^2),
   g >= (a/den)(1 - eps) kappa / sqrt(sigma^2 J_vv)   (>= (a/den)(1 - eps) kappa since sigma^2 J_vv <= 1).
States: the common-envelope class C (two thin w-lines + hidden multipole rows on one Gaussian w-envelope, rows modulated in w with any k)
at sigma = 1e-4 (and the lines alone at sigma = 1e-6).  -> results/ar_check.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from v3_tools import _pre, point_fields
from u_cgm import trap_weights
from an_multipole import realise


def class_C_terms(xs, lw, f, sig, Lw, k, ev=1e-3):
    lwr = lw - np.log(np.sum(np.exp(lw - lw.max()))) - lw.max()
    S = np.diag([(ev * sig) ** 2, Lw ** 2]); T = []
    for x, l_ in zip(xs, lwr): T.append((l_, np.array([x * sig, 0.0]), S, np.zeros(2)))
    for x, l_, ff in zip(xs, lwr, f):
        if ff == 0: continue
        lc = l_ + np.log(abs(ff) / 2) + (0j if ff > 0 else 1j * np.pi)
        for sg in (1, -1): T.append((lc, np.array([x * sig, 0.0]), S, np.array([0.0, sg * k / sig])))
    return T


def sums(T, sig, vn, wn, rows=8):
    pre = _pre(T, sig); wv, ww = trap_weights(vn), trap_weights(wn)
    keys = ('svv', 'sww', 'Nibp', 'N1', 'phi', 'phi_svv', 'sww_w', 'ibp_rest'); acc = {k: 0.0 for k in keys}; m = 0.0
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        p, q = point_fields(pre, sig, X); pw = p * dA; m += pw.sum()
        q = dict(q); q['phi'] = 1 - np.cos(np.sqrt(q['mz2']))
        for k in keys: acc[k] += float(np.sum(pw * q[k]))
    return {k: v / m for k, v in acc.items()}


def analyse(E, sig, Varw):
    s2 = sig * sig; Jvv, Jww = E['svv'], E['sww']; a = s2 * Jvv; den = s2 * (Jvv + Jww); B = a; A = s2 * Jww
    g = (2 * A + E['Nibp']) / (sig * den)
    Pw = 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2
    Bt = s2 / 2 * E['ibp_rest']; R3 = E['Nibp'] - E['N1']; eps = -(Bt + R3) / (sig * a * np.sqrt(Pw))
    Phi = E['phi']; dc = 1 - E['phi_svv'] / (Jvv * Phi); dw = 1 - E['sww_w'] / (Jww * (1 - Phi / 4))
    kappa2 = 4 * Phi * (1 - Phi / 4) * (1 - dc) * (1 - dw) / (Varw + s2)
    bound_W = (a / den) * np.sqrt(Pw) * (1 - eps); bound_AR = (a / den) * (1 - eps) * np.sqrt(kappa2) / np.sqrt(B)
    cr = A * (Varw + s2) / s2                                   # Cramer-Rao ratio (>= 1)
    return {'g': g, 'sig2Jvv': B, 'sig2Jww': A, 'a_over_den': a / den, 'P_w': Pw, 'eps': eps, 'Phi': Phi, 'delta_c': dc, 'delta_w': dw,
            'kappa': float(np.sqrt(kappa2)), 'four_Phi_over_Varw': 4 * Phi / (Varw + s2), 'CR_ratio': cr,
            'bound_ThmW': bound_W, 'bound_AR2': bound_AR, 'one_over_sqrtB': 1 / np.sqrt(B), 'g_over_bound_AR2': g / bound_AR}


if __name__ == '__main__':
    M = json.load(open('results/an_multipole.json')); out = {}
    cases = [('lines', None, 8.0, 0.0, 0.0, 1.0, 1e-4, 2.0), ('lines_Lw1', None, 8.0, 0.0, 0.0, 1.0, 1e-4, 1.0), ('lines_Lw4', None, 8.0, 0.0, 0.0, 1.0, 1e-4, 4.0),
             ('mp_n1_D2_k1', '1', 2.0, 0.5, 1.0, 1.0, 1e-4, 2.0), ('mp_n2_D2_k1', '2', 2.0, 0.5, 1.0, 1.0, 1e-4, 2.0), ('mp_n2_D3_k1', '2', 3.0, 0.5, 0.5, 1.0, 1e-4, 2.0),
             ('mp_n2_D2_k05', '2', 2.0, 0.5, 1.0, 0.5, 1e-4, 2.0), ('mp_n2_D2_k2', '2', 2.0, 0.5, 1.0, 2.0, 1e-4, 2.0), ('mp_n2_D2_k1_Lw4', '2', 2.0, 0.5, 1.0, 1.0, 1e-4, 4.0),
             ('mp_n4_D4_k1', '4', 4.0, 0.5, 1.0, 1.0, 1e-4, 2.0), ('lines_s1e-6', None, 8.0, 0.0, 0.0, 1.0, 1e-6, 2.0)]
    for name, n, D, d, z, k, sig, Lr in cases:
        Lw = Lr * np.sqrt(sig)
        if n is None: xs, lw, f = np.array([-D, D]), np.zeros(2), np.zeros(2)
        else: L = M['leading'][n]; xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max()
        T = class_C_terms(xs, lw, f, sig, Lw, k)
        hv = min(0.0125, 0.1 / max(xs.max(), 1.0)) * sig
        vn = np.arange(xs.min() * sig - 8 * sig, xs.max() * sig + 8 * sig, hv)
        wstep = Lw / 40 if n is None else min(2 * np.pi / k * sig / 16, Lw / 40); wn = np.arange(-6 * Lw, 6 * Lw, wstep)
        E = sums(T, sig, vn, wn); r = analyse(E, sig, Lw ** 2)          # Var_w(rho) = Lw^2 (the modulation changes it by O(e^{-k^2 Lw^2/2 sigma^2}))
        r.update({'sigma': sig, 'Lw_over_sqrt_sigma': Lr, 'k': k, 'n': n, 'D': D})
        out[name] = r
        print('%-16s g %.5f >= Thm2 %.5f >= AR2 %.5f | 1/sqrtB %.5f kappa %.5f (4Phi/Varw %.5f, d_c %+.1e, d_w %+.1e, CR %.4f, eps %+.1e, a/den %.6f)' % (
            name, r['g'], r['bound_ThmW'], r['bound_AR2'], r['one_over_sqrtB'], r['kappa'], r['four_Phi_over_Varw'], r['delta_c'], r['delta_w'], r['CR_ratio'], r['eps'], r['a_over_den']), flush=True)
        json.dump(out, open('results/ar_check.json', 'w'), indent=1, default=float)
