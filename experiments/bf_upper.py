"""Stage BF §1.2: the upper side for the periodic windows on an explicit comb sequence — Theorem AZ1-P.

Lemma AZ1-P (closed-form upper bounds of the branches on a periodic window; proof — Lemma AZ1 with the P-replacements):
  (a)  <= 2 sigma^2 J_ww,up/(sigma den_lo),  J_ww,up = (1 + u0/(8(1 - Q)))/L^2 (AX1-P),  den_lo = b + x I2/(2(1 + Q)) (AX2-P)
  (a') <= (sigma^2/2) E[s_v^2 phi]/(sigma den_lo) with (exact identity of AW6, <G> = -J_vv,per: integration by parts over one period)
     E[s_v^2 phi] <= (1 - e) J_per,up + <s~^2 R^2>/4 + (1 - e)/V + (1 - C)/V + a^2 e C + gamma_P (1 - C),   1 - C <= (L + r)^2/4,
     J_per,up = x I2/(2(1 - Q) sigma^2),  <s~^2 w^2> <= J_per,up v  (v = L^2 s^2/(s^2 + 2 L^2): the common envelope e^{-w^2/s^2} of all harmonics),
     <s~^2 (R - w)^2> <= 3 Xi_P J_per,up + 3 sigma^4 <s~^2 w^2>/L^4,  r^2 <= 3(xi_P^2 + sigma^4/L^2 + sigma^2 Q^2 (sigma/s)^2/(e (1 - Q)^2)),
     xi_P = sigma sqrt(x) N1/(1 - Q),  Xi_P = xi_P^2 + sigma^4 Q^2/(e s^2 (1 - Q)^2),
     gamma_P = [x (N2 (1 + Q) + N1^2) + sqrt(x) N1 (1 + 2 Q)/sqrt(e)]/(1 - Q)^2   (AX3-P; f even).
  With the branch identity, g <= (1 + |eps|) U_P,  U_P = (a)_up + (a')_up.
The comb sequence S'_C (construction; the image of the strip sequence S' under Proposition BE1's map with a = 2):
  4 k0^2 + 11 ln k0 = ln(256/sigma),  lam = 2/sqrt(k0),  M = k0^2 e^{k0^2}/4,  eta = Lv/ell = 1/k0^2,
  Gaussian teeth with a_n = 2 e^{-n^2 tau^2/2}, tau = k0 st = 1/k0^2  (a_1 = 2 e^{-1/(2 k0^4)} -> 2; (S5) |a_n| <= a_1).
  Lengths: sqrt(sigma) = 16 k0^{-11/2} e^{-2k0^2}, ell = 32 k0^{-6} e^{-2k0^2}, Lw = 8 k0^{-4} e^{-k0^2}, Lv = 32 k0^{-8} e^{-2k0^2} (exact monomials).
-> results/bf_upper.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, os
import numpy as np
from scipy.optimize import brentq
sys.argv = sys.argv[:1]
from be_plemmas import harmonics
from bf_fc import fc_lower_P

E_ = np.e; LN256 = 8 * np.log(2.0)


def seqC(lnsig, nmax=None):
    k0 = brentq(lambda k: 4 * k * k + 11 * np.log(k) - lnsig - LN256, 0.3, 1000.0); tau = 1 / k0 ** 2
    nmax = nmax or max(4, int(np.ceil(np.sqrt(2 * np.log(1e30)) / tau)))
    return {'k0': k0, 'lam': 2 / np.sqrt(k0), 'M': k0 * k0 * np.exp(k0 * k0) / 4, 'eta': 1 / k0 ** 2, 'tau': tau,
            'amps': [2 * np.exp(-(n * tau) ** 2 / 2) for n in range(1, nmax + 1)]}


def U_main_P(sig, H):
    L2, s2, V, x, Q = H['L2'], H['s2'], H['V'], H['x'], H['Qbar']; qb = H['qbar']; n = np.arange(1, len(qb) + 1)
    L = np.sqrt(L2); s = np.sqrt(s2); s2g = sig * sig; b = s2g / V
    N1 = float(np.sum(n * qb)); N2 = float(np.sum(n * n * qb))
    Jww_up = (1 + H['u0'] / (8 * (1 - Q))) / L2
    den_lo = b + x * H['I2'] / (2 * (1 + Q)); Jper_up = x * H['I2'] / (2 * (1 - Q) * s2g)
    a2 = (1 - b) ** 2 / 2; e = np.exp(-a2 * V / 2); ome = -np.expm1(-a2 * V / 2)
    xi = sig * np.sqrt(x) * N1 / (1 - Q); Xi = xi ** 2 + s2g * s2g * Q ** 2 / (E_ * s2 * (1 - Q) ** 2)
    r = np.sqrt(3 * (xi ** 2 + s2g * s2g / L2 + s2g * Q ** 2 * (s2g / s2) / (E_ * (1 - Q) ** 2)))
    gam = (x * (N2 * (1 + Q) + N1 * N1) + np.sqrt(x) * N1 * (1 + 2 * Q) / np.sqrt(E_)) / (1 - Q) ** 2
    v = L2 * s2 / (s2 + 2 * L2); st2w2 = Jper_up * v
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2 ** 2
    st2R2 = (np.sqrt(st2w2) + np.sqrt(st2Rw)) ** 2; omC = (L + r) ** 2 / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + a2 * e + gam * omC
    a_up = 2 * s2g * Jww_up / (sig * den_lo); ap_up = s2g / 2 * Esv2phi / (sig * den_lo)
    return {'a_up': a_up, 'ap_up': ap_up, 'U_main': a_up + ap_up, 'Qbar': Q, 'x': x, 'L': L, 's': s}


def U_of(lnsig, nmax=None):
    P = seqC(lnsig, nmax); sig = np.exp(-lnsig); H = harmonics(sig, P['k0'], P['lam'], P['M'], P['eta'], P['amps']); return U_main_P(sig, H), P, H


def true_comb(sig, P, mmax=4):
    import at_tools
    from w_tools import terms_of, w_nodes
    s = np.sqrt(sig); lam = P['lam']; win = [(a, 0.0, lam, n) for n, a in enumerate(P['amps'][:mmax], 1)]; bg = [(1.0, 0.0, lam * P['M'])]
    T = terms_of(bg, win, P['k0'], sig, P['eta'] * lam / s); wn = w_nodes(bg, win, sig, 50, 400)
    r = at_tools.analyse(at_tools.sums_envelope(T, sig, P['k0'], np.sqrt((P['eta'] * lam * s) ** 2 + sig ** 2), wn, nv=64 * mmax, n0=31), sig)
    den = r['sig2Jvv'] + r['sig2Jww']
    aw = 2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (sig * den); aN = r['sig2Jvv'] * r['Phi'] * (1 - r['delta_c']) / (2 * sig * den)
    return {'g': float(r['g']), 'a': float(aw), 'ap': float(aN), 'eps': float(r['eps'])}


if __name__ == '__main__':
    np.seterr(all='ignore'); fn = os.environ.get('BF_OUT', 'results/bf_upper_full.json'); out = json.load(open(fn)) if os.path.exists(fn) else {'true': {}, 'grid': {}}     # resumable
    for sig in (1e-6, 1e-12, 1e-18, 1e-24):
        if '%g' % sig in out['true']: continue
        L = np.log(1 / sig); P = seqC(L, 4); H = harmonics(sig, P['k0'], P['lam'], P['M'], P['eta'], P['amps']); U = U_main_P(sig, H)
        Ufull = U_of(L)[0]; t = true_comb(sig, P)
        rec = {'k0': P['k0'], 'a1': P['amps'][0], 'x': H['x'], 'Qbar': H['Qbar'], 'g_true': t['g'], 'a_true': t['a'], 'ap_true': t['ap'], 'eps': t['eps'], **U,
               'U_full_comb': Ufull['U_main'], 'g_sqrt_ln': t['g'] * np.sqrt(L), 'U_sqrt_ln': U['U_main'] * np.sqrt(L),
               'bound_true_ok': bool(t['g'] <= (1 + abs(t['eps'])) * U['U_main'] and t['a'] <= U['a_up'] and t['ap'] <= U['ap_up'])}
        out['true']['%g' % sig] = rec; json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('S\'_C sigma %-6g k0 %.3f a1 %.4f x %.2f Q %.3f: g %.6f  (a) %.6f <= %.6f  (a\') %.6f <= %.6f  U %.6f (all teeth %.6f)  U/g-1 %.1e | g sqrt ln %.4f  U sqrt ln %.4f  eps %.1e  %s' % (
            sig, P['k0'], P['amps'][0], H['x'], H['Qbar'], t['g'], t['a'], U['a_up'], t['ap'], U['ap_up'], U['U_main'], Ufull['U_main'], U['U_main'] / t['g'] - 1, rec['g_sqrt_ln'],
            rec['U_sqrt_ln'], t['eps'], rec['bound_true_ok']), flush=True)
    lns = np.linspace(np.log(1e24), np.log(1e150), 2000); vals = []
    for L in lns:
        U, P, H = U_of(L); F = fc_lower_P(np.exp(-L), H)['F_lower_over_1_minus_eps']
        vals.append((L, U['U_main'] * np.sqrt(L), H['Qbar'], H['rho'], H['x'], F, F / min(0.149, 1.2 / np.sqrt(L))))
    vals = np.array(vals); C = (vals[:, 1] - 1) / (np.log(vals[:, 0]) / vals[:, 0])
    out['grid'] = {'ln_min': float(lns[0]), 'ln_max': float(lns[-1]), 'max_coeff': float(C.max()), 'U_sqrt_ln_first_last': [float(vals[0, 1]), float(vals[-1, 1])],
                   'max_Qbar': float(vals[:, 2].max()), 'max_rho': float(vals[:, 3].max()), 'min_x': float(vals[:, 4].min()), 'min_Fc_lower_over_1_minus_eps': float(vals[:, 5].min()),
                   'min_Fc_lower_over_G': float(vals[:, 6].min())}
    print('dense grid ln(1/sigma) in [55, 345] (all teeth): U sqrt(ln) - 1 <= %.3f ln ln/ln; U sqrt(ln) from %.4f to %.4f | the sequence is in the class: '
          'Qbar <= %.4f, rho <= %.4f, x >= %.2f, F_c/(1 - |eps|) >= %.6f (Lemma AY2-P) = %.2f G' % (C.max(), vals[0, 1], vals[-1, 1], vals[:, 2].max(), vals[:, 3].max(),
                                                                                                   vals[:, 4].min(), vals[:, 5].min(), vals[:, 6].min()))
    json.dump(out, open(fn, 'w'), indent=1, default=float)
