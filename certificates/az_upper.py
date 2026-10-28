"""Stage AZ: the upper side for the strip family — Theorem AZ1 (proof up to paper E's remainder eps, as on the lower side).

Identity (proof): g = (a) + (a') - eps * (Theorem-2 bound) and Theorem-2 bound <= (a) + (a') (AM-GM), so  g <= (1 + |eps|) ((a) + (a')).
Lemma AZ1 (closed-form upper bounds; proof):
  (a)  = 2A_w/(sigma den) <= 2 sigma^2 J_ww,up / (sigma den_lo),   J_ww,up = 1/L^2 + kq B^2 (Lemma AX1),  den_lo = b + x I_q/2 (Lemma AX2)
  (a') = N_phi/(sigma den),  N_phi = (sigma^2/2) E[s_v^2 phi] and, from the exact identity of Lemma AW6 (with <G> = -J_vv,per, Lemma AX3):
     E[s_v^2 phi] <= (1 - e) J_per + <s~^2 R^2>/4 + (1 - e)/V + (1 - C)/V + a^2 e C + gamma (1 - C),
     1 - C <= (L + r)^2/4,  <s~^2 R^2> <= (sqrt<s~^2 w^2> + sqrt<s~^2 (R-w)^2>)^2,  <s~^2 w^2> <= (x/sigma^2) q0^2 L^2 (1 + 2L^2/s^2)^{-3/2}/(1 + sqrt(1 - qbar^2)),
     <s~^2 (R-w)^2> <= 3 Xi J_per,up + 3 sigma^4 <s~^2 w^2>/L^4,  r^2 <= 3(xi^2 + sigma^4/L^2 + sigma^2 qbar^2 (sigma/s)^2/(e(1-qbar)^2)).
The new sequence S' (O(1/k0) terms cancelled): c = 1, 4 k0^2 + 11 ln k0 = ln(1/sigma), M = k0^2 e^{k0^2}, eta = Lv/ell = 1/k0^2, lam = 2/sqrt(k0).
-> results/az_upper.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import brentq
from aw_lemmas import profile

E_ = np.e


def seq2(lnsig, a=11.0):
    k0 = brentq(lambda k: 4 * k * k + a * np.log(k) - lnsig, 0.3, 1000.0)
    return {'k0': k0, 'c': 1.0, 'M': k0 * k0 * np.exp(k0 * k0), 'eta': 1 / k0 ** 2, 'lam': 2 / np.sqrt(k0)}


def U_main(sig, k0, lam, M, c, eta):
    P = profile(sig, k0, lam, M, c, eta); L2 = P['L2']; L = np.sqrt(L2); s2 = P['s2']; s = np.sqrt(s2); V = P['V']; qb = P['qbar']; q0 = P['q0']; x = P['kp_sigma'] ** 2
    s2g = sig * sig; b = s2g / V
    kq = np.expm1(-0.5 * np.log1p(-qb * qb)) / qb ** 2                  # (1/sqrt(1-q^2) - 1)/q^2 without cancellation
    B2 = q0 ** 2 * L2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** 1.5); Iq = q0 ** 2 * s / np.sqrt(s2 + 2 * L2)
    Jww_up = 1 / L2 + kq * B2
    Wp_lo = x * Iq / 2; Wp_up = x * Iq / (1 + np.sqrt(1 - qb * qb)); den_lo = b + Wp_lo
    Jper_up = Wp_up / s2g
    a2 = (1 - b) ** 2 / 2; e = np.exp(-a2 * V / 2); ome = -np.expm1(-a2 * V / 2)
    xi = sig * np.sqrt(x) * qb / (1 - qb); Xi = xi ** 2 + s2g * s2g * qb ** 2 / (E_ * s2 * (1 - qb) ** 2)
    r = np.sqrt(3 * (xi ** 2 + s2g * s2g / L2 + s2g * qb ** 2 * (s2g / s2) / (E_ * (1 - qb) ** 2)))
    gam = qb * (x * (1 + qb) + np.sqrt(x)) / (1 - qb) ** 2
    st2w2 = (x / s2g) * q0 ** 2 * L2 * (1 + 2 * L2 / s2) ** -1.5 / (1 + np.sqrt(1 - qb * qb))
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2 ** 2
    st2R2 = (np.sqrt(st2w2) + np.sqrt(st2Rw)) ** 2
    omC = (L + r) ** 2 / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + a2 * e + gam * omC
    Nphi_up = s2g / 2 * Esv2phi
    a_up = 2 * s2g * Jww_up / (sig * den_lo); ap_up = Nphi_up / (sig * den_lo)
    return {'a_up': a_up, 'ap_up': ap_up, 'U_main': a_up + ap_up, 'qbar': qb, 'x': x, 'L': L, 's': s}


if __name__ == '__main__':
    from av_strip import true_strip
    from as_strip_lo import g_lo
    out = {'true': {}, 'grid': {}}
    for sig in (1e-6, 1e-12, 1e-18, 1e-24):
        P = seq2(np.log(1 / sig)); r = true_strip(sig, P); U = U_main(sig, P['k0'], P['lam'], P['M'], P['c'], P['eta'])
        den = r['sig2Jvv'] + r['sig2Jww']; aw = 2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (sig * den); aN = r['sig2Jvv'] * r['Phi'] * (1 - r['delta_c']) / (2 * sig * den)
        rec = {'k0': P['k0'], 'g_true': r['g'], 'a_true': aw, 'ap_true': aN, 'eps': r['eps'], 'g_LO': g_lo(sig, P['k0'], P['lam'], P['M'], P['Lv_sig'] * sig if 'Lv_sig' in P else P['eta'] * P['lam'] * np.sqrt(sig), 1.0),
               **U, 'g_sqrt_ln': r['g'] * np.sqrt(np.log(1 / sig)), 'U_sqrt_ln': U['U_main'] * np.sqrt(np.log(1 / sig)), 'bound_true_ok': bool(r['g'] <= (1 + abs(r['eps'])) * U['U_main'] and aw <= U['a_up'] and aN <= U['ap_up'])}
        out['true']['%g' % sig] = rec
        print('S\' sigma %-6g k0 %.3f: g %.6f  (a) %.6f <= %.6f  (a\') %.6f <= %.6f  U %.6f  rel %.1e | g sqrt(ln) %.4f  U sqrt(ln) %.4f  eps %.1e  %s' % (
            sig, P['k0'], r['g'], aw, U['a_up'], aN, U['ap_up'], U['U_main'], U['U_main'] / r['g'] - 1, rec['g_sqrt_ln'], rec['U_sqrt_ln'], r['eps'], rec['bound_true_ok']), flush=True)
    # dense grid in ln(1/sigma) (double precision range)
    lns = np.linspace(np.log(1e24), np.log(1e300), 3000); vals = []
    for L in lns:
        P = seq2(L); U = U_main(np.exp(-L), P['k0'], P['lam'], P['M'], P['c'], P['eta'])
        vals.append((L, U['U_main'] * np.sqrt(L), P['k0'], U['qbar']))
    vals = np.array(vals); C = (vals[:, 1] - 1) / (np.log(vals[:, 0]) / vals[:, 0])
    out['grid'] = {'ln_min': float(lns[0]), 'ln_max': float(lns[-1]), 'max_U_sqrt_ln_minus_1_over_lnln_ln': float(C.max()), 'U_sqrt_ln_at': {('%.0f' % L): float(u) for L, u, _, _ in vals[::300]}}
    print('dense grid ln(1/sigma) in [55, 691]: U sqrt(ln) - 1 <= %.3f ln ln/ln (max of the ratio); U sqrt(ln) from %.4f to %.4f' % (C.max(), vals[0, 1], vals[-1, 1]))
    json.dump(out, open('results/az_upper.json', 'w'), indent=1, default=float)


# ---- arbitrary-precision version (Python decimal) of profile + U_main, for the sequence far below the double range ----
def U_main_dec(lnsig, prec=60, a=11.0):
    from decimal import Decimal as D, getcontext
    getcontext().prec = prec
    def em1(y):                                  # expm1 for Decimal (series for tiny |y|)
        return y + y * y / 2 + y ** 3 / 6 if abs(y) < D('1e-15') else y.exp() - 1
    def l1p(y):
        return y - y * y / 2 + y ** 3 / 3 if abs(y) < D('1e-15') else (1 + y).ln()
    L_ = D(repr(float(lnsig))); sig = (-L_).exp()
    k = D(repr(float(brentq(lambda k: 4 * k * k + a * np.log(k) - float(lnsig), 0.3, 1e5))))
    for _ in range(60):                          # Newton refine: 4k^2 + a ln k - L = 0
        f = 4 * k * k + D(repr(a)) * k.ln() - L_; fp = 8 * k + D(repr(a)) / k; k = k - f / fp
    k0 = k; c = D(1); M = k0 ** 2 * (k0 ** 2).exp(); eta = 1 / k0 ** 2; lam = 2 / k0.sqrt()
    ell = lam * sig.sqrt(); Lw = M * ell; Lv = eta * ell; V = Lv ** 2 + sig ** 2; L2 = Lw ** 2 + sig ** 2
    Sb = 1 / (1 / Lw ** 2 + 1 / ell ** 2); qb = c * (-(k0 ** 2) * Lv ** 2 / (2 * V)).exp()
    q0 = qb * (Sb * L2 / (Lw ** 2 * (Sb + sig ** 2))).sqrt(); s2 = 1 / (1 / (Sb + sig ** 2) - 1 / L2); x = (k0 * Lv ** 2 / V) ** 2
    s = s2.sqrt(); L = L2.sqrt(); s2g = sig ** 2; b = s2g / V; E1 = D(1).exp()
    kq = em1(-l1p(-qb * qb) / 2) / qb ** 2
    B2 = q0 ** 2 * L2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** D('1.5')); Iq = q0 ** 2 * s / (s2 + 2 * L2).sqrt()
    Jww_up = 1 / L2 + kq * B2; Wp_lo = x * Iq / 2; Wp_up = x * Iq / (1 + (1 - qb * qb).sqrt()); den_lo = b + Wp_lo; Jper_up = Wp_up / s2g
    a2 = (1 - b) ** 2 / 2; e = (-a2 * V / 2).exp(); ome = -em1(-a2 * V / 2)
    xi = sig * x.sqrt() * qb / (1 - qb); Xi = xi ** 2 + s2g * s2g * qb ** 2 / (E1 * s2 * (1 - qb) ** 2)
    r = (3 * (xi ** 2 + s2g * s2g / L2 + s2g * qb ** 2 * (s2g / s2) / (E1 * (1 - qb) ** 2))).sqrt()
    gam = qb * (x * (1 + qb) + x.sqrt()) / (1 - qb) ** 2
    st2w2 = (x / s2g) * q0 ** 2 * L2 / (1 + 2 * L2 / s2) ** D('1.5') / (1 + (1 - qb * qb).sqrt())
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2 ** 2
    st2R2 = (st2w2.sqrt() + st2Rw.sqrt()) ** 2; omC = (L + r) ** 2 / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + a2 * e + gam * omC
    U = 2 * s2g * Jww_up / (sig * den_lo) + s2g / 2 * Esv2phi / (sig * den_lo)
    return float(U * L_.sqrt()), float(k0), float(qb)
