"""Stage BE §1.2: the strip lemmas generalised to periodic windows (P-lemmas), checked on exact point fields.

Family P: rho = N(v; Lv^2) N(w; Lw^2)[1 + e^{-w^2/2 ell^2}(f(k0 v/sigma) - 1)], f >= 0 even, 2 pi-periodic, mean 1, f = 1 + sum_n a_n cos(n t),
|a_n| <= |a_1| <= 2 (S5: fundamental-dominated; |a_n| <= 2 holds for every positive f).  Strip: a_1 = c, a_n = 0; comb: a_n = 2c e^{-(n k0 st)^2/2}.
Lemma AW1-P (exact): p = N(v; V) Gamma(w) [1 + sum_n q_n(w) cos(n k' v)],  q_n(w) = q0_n e^{-w^2/2 s^2}  (one s for all n),
   q0_n = |a_n| e^{-n^2 K/2} kappa_s,  qbar_n = |a_n| e^{-n^2 K/2},  K = k0^2 Lv^2/V >= x = (k' sigma)^2,  Qbar = sum_n qbar_n.
Lemma AX1-P:  (1 + u0/(8(1 + Qbar)))/L^2 <= J_ww <= (1 + u0/(8(1 - Qbar)))/L^2,   u0 = 4 L^2 int Gamma sum_n q_n'^2.
Lemma AX2-P:  x I2/(2(1 + Qbar)) <= W_p := sigma^2 J_vv - sigma^2/V <= x I2/(2(1 - Qbar)),   I2 = sum_n n^2 q0_n^2 s/sqrt(s^2 + 2 L^2).
Prop. AX4-P:  S <= 8 kappa2(Qbar, rho) u2/(8 + u0/(1 + Qbar)),   u2 = sum_n n^2 (u0-term of harmonic n)   (S = (k_eff^2 - sigma^2/4V)/x, AW3).
Lemma AW4(i)-P (Wiener majorant):  S <= 1 + sum_{n>=2} (n^2 - 1) (|F_n|/|F_1|)^2  with  |F_n|/||d(sqrt Gamma g)|| <= sum_j |b_j| |kappa_{j,n}| j^{1/4},
   kappa_{j,n} the cos(nt)-coefficient of (sum_m q0_m cos mt)^j,  |kappa_{j,n}| <= 2 h_w^j w^{-n} (h_w = sum_m q0_m (w^m + w^-m)/2), b_j = binom(1/2, j).
-> results/be_plemmas.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
from scipy.special import binom
sys.argv = sys.argv[:1]
import bd_case3 as D
from aw_lemmas import profile
from ba_tools import strip_opt


def harmonics(sig, k0, lam, M, eta, amps):
    P1 = profile(sig, k0, lam, M, 1.0, eta); K = k0 ** 2 * P1['Lv'] ** 2 / P1['V']; ks = P1['q0'] / P1['qbar']
    qbar = np.array([abs(a) * np.exp(-(n ** 2) * K / 2) for n, a in enumerate(amps, 1)]); q0 = qbar * ks
    L2, s2 = P1['L2'], P1['s2']; shape_u = 4 * L2 ** 2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** 1.5)
    n = np.arange(1, len(amps) + 1)
    return {'x': P1['kp_sigma'] ** 2, 'K': K, 'L2': L2, 's2': s2, 'V': P1['V'], 'rho': P1['rho'], 'qbar': qbar, 'q0': q0, 'Qbar': qbar.sum(),
            'u0': shape_u * np.sum(q0 ** 2), 'u2': shape_u * np.sum(n ** 2 * q0 ** 2), 'I2': np.sum(n ** 2 * q0 ** 2) * np.sqrt(s2) / np.sqrt(s2 + 2 * L2)}


def kappa2(Q, rho): return (1 + rho + Q / (2 * (1 - Q))) ** 2 / (8 * (1 - Q))


def S_branch_i(q0, w=2.0, J=80, N=80):
    """Wiener-majorant bound of 1 + sum_{n>=2}(n^2 - 1)(|F_n|/|F_1|)^2 for given q0_m (cos coefficients of the blurred profile)."""
    m = np.arange(1, len(q0) + 1); h = float(np.sum(q0 * (w ** m + w ** -m.astype(float)) / 2))
    assert h < 1
    bj = np.array([abs(binom(0.5, j)) for j in range(J + 1)]); jj = np.arange(J + 1) ** 0.25
    tail = lambda n: 2 * w ** (-n) * float(np.sum(bj[2:] * jj[2:] * h ** np.arange(2, J + 1)))        # j >= 2 part, all n
    F1 = 0.5 * q0[0] - tail(1)
    Fn = [0.5 * (q0[k - 1] if k <= len(q0) else 0.0) + tail(k) for k in range(2, N + 1)]
    return 1 + sum((k * k - 1) * (f / F1) ** 2 for k, f in zip(range(2, N + 1), Fn)), h


def check(sig, k0, lam, M, eta, amps, label):
    s = np.sqrt(sig); H = harmonics(sig, k0, lam, M, eta, amps); ell = lam * s
    win = [(a, 0.0, lam, n) for n, a in enumerate(amps, 1)]
    import at_tools  # noqa: the analyse dict is inside row(); recompute the needed raw numbers
    from w_tools import terms_of, w_nodes
    T = terms_of([(1.0, 0.0, lam * M)], win, k0, sig, eta * lam / s); wn = w_nodes([(1.0, 0.0, lam * M)], win, sig, 40, 300)
    E = at_tools.sums_envelope(T, sig, k0, np.sqrt((eta * lam * s) ** 2 + sig ** 2), wn, nv=32 * len(amps), n0=21); r = at_tools.analyse(E, sig)
    L2, Q, x, V = H['L2'], H['Qbar'], H['x'], H['V']; s2g = sig * sig
    Jww = r['sig2Jww'] / s2g; Wp = r['sig2Jvv'] - s2g / V; S = (r['k_eff'] ** 2 - s2g / (4 * V)) / x
    out = {'label': label, 'sigma': sig, 'x': x, 'Qbar': Q, 'u0': H['u0'], 'rho': H['rho'],
           'AX1P_lo': (1 + H['u0'] / (8 * (1 + Q))) / L2, 'AX1P_up': (1 + H['u0'] / (8 * (1 - Q))) / L2, 'J_ww': Jww,
           'AX2P_lo': x * H['I2'] / (2 * (1 + Q)), 'AX2P_up': x * H['I2'] / (2 * (1 - Q)), 'W_p': Wp,
           'AX4P': 8 * kappa2(Q, H['rho']) * H['u2'] / (8 + H['u0'] / (1 + Q)), 'S': S}
    Si, h = S_branch_i(H['q0']); out.update({'AW4iP': Si, 'h_w': h})
    out['ok'] = bool(out['AX1P_lo'] <= Jww <= out['AX1P_up'] and out['AX2P_lo'] <= Wp <= out['AX2P_up'] and S <= out['AX4P'] and S <= Si)
    print('%-22s x %.3f Qbar %.3f | J_ww %.6e in [%.6e, %.6e] | W_p %.6e in [%.6e, %.6e] | S %.5f <= AX4-P %.5f, AW4(i)-P %.5f  %s' % (
        label, x, Q, Jww, out['AX1P_lo'], out['AX1P_up'], Wp, out['AX2P_lo'], out['AX2P_up'], S, out['AX4P'], Si, out['ok']), flush=True)
    return out


if __name__ == '__main__':
    res = []
    for sg in ('1e-12', '1e-24'):
        Dd = json.load(open('results/bd_opt_comb_%s.json' % sg)); x = np.exp(Dd['x'][:4]); amps = [w[0] for w in Dd['windows']]
        res.append(check(float(sg), Dd['kb'], x[1], x[2], x[3], amps, 'comb optimum %s' % sg))
        P = strip_opt(float(sg)); res.append(check(float(sg), P['k0'], P['lam'], P['M'], P['Lv_sig'] * float(sg) / (P['lam'] * np.sqrt(float(sg))), [P['c']], 'strip optimum %s' % sg))
    for st, k0 in ((0.3, 2.5), (0.15, 2.0), (0.6, 1.6)):           # further combs (c = 0.999, lam = 1.5, M = 500, eta = 0.2) at 1e-12
        amps = [0.999 * 2 * np.exp(-(n * k0 * st) ** 2 / 2) for n in range(1, int(np.ceil(np.sqrt(2 * np.log(1e6)) / (k0 * st))) + 1)]
        res.append(check(1e-12, k0, 1.5, 500.0, 0.2, amps, 'comb st %.2f k0 %.1f' % (st, k0)))
    json.dump(res, open('results/be_plemmas.json', 'w'), indent=1, default=float)
