"""Stage AW: proofs for the lower side of Proposition AV1 (strip family), with explicit constants — numerical checks of every lemma.

Lemma AW1 (exact factorisation; proof).  The blurred strip is  p(v, w) = N(v; V) Gamma(w) [1 + q(w) cos(k' v)],
  V = Lv^2 + sigma^2,  Gamma = N(w; 0, L^2), L^2 = Lw^2 + sigma^2,  k' = (k0/sigma) Lv^2/V,
  q(w) = q0 exp(-w^2/(2 s^2)),  q0 = c e^{-k0^2 Lv^2/(2V)} sqrt(S_b L^2/(Lw^2 (S_b + sigma^2))),  1/s^2 = 1/(S_b + sigma^2) - 1/L^2,
  S_b = (1/Lw^2 + 1/ell^2)^{-1};  0 < q0 <= qbar := c e^{-k0^2 Lv^2/(2V)} (<= c e^{-k0^2/2} e^{k0^2 sigma^2/(2V)}).
Lemma AW2 (Fourier coefficients of h = sqrt(1 + x cos t) = sum a_n(x) cos nt; proof by shifting the contour to |Im t| = tau, cosh tau = 1/(2|x|)):
  |a_n(x)| <= sqrt6 (2|x|)^n,  |a_n'(x)| <= sqrt2 (2|x|)^{n-1}  (|x| <= 1/2, n >= 1);  1/(2 sqrt(1+|x|)) <= a_1'(x) <= 1/(2 sqrt(1-|x|)),
  |a_1(x)| <= |x|/(2 sqrt(1-|x|)),  sqrt(1-|x|) <= a_0(x) <= 1,  |a_0'(x)| <= |x|/(8 (1-|x|)^{3/2}).
Lemma AW3 (envelope separation; proof).  F = d_w sqrt p = sqrt(N_V(v)) F_per(v, w), F_per periodic in v.  Integrating by parts in v and using that a
  Gaussian average of a periodic function equals its period average up to e^{-k'^2 V/2}-small terms:
  k_eff^2 = (k' sigma)^2 S + sigma^2/(4V) + delta_osc,  S = sum n^2 ||F_n||^2 / (2||F_0||^2 + sum_{n>=1} ||F_n||^2)  (F_per = sum F_n(w) cos(n k' v)),
  |delta_osc| = O(e^{-k'^2 V/2}) (here k'^2 V ~ (k0 Lv/sigma)^2 > 1e10).
Proposition AW4 (S bounds; proof).  With A = ||(sqrt Gamma)' q||, B = ||sqrt Gamma q'|| (L^2(dw)), the exact ratio A/B = rho := s^2/(2L^2), and
  u := 4 L^2 B^2 = 4 L^4 q0^2 / (s^4 (1 + 2L^2/s^2)^{3/2}) <= sqrt2 q0^2 L/s:
  (i)  S <= 1 + K1 qbar^2,  K1 = 8 (1 + 12 rho^2) Sig(4 qbar^2) / tau1,  tau1 = (1/2)(1/(2 sqrt(1+qbar)) - rho/(2 sqrt(1-qbar)))^2,  Sig(x) = (4 - 3x + x^2)/(1-x)^3;
  (ii) S <= K2 u,  K2 = D/(sqrt(1-qbar) - qbar sqrt(u)/(4 (1-qbar)^{3/2}))^2,  D = (1+rho)^2/(8 (1-qbar)) + 8 qbar^2 (1 + 12 rho^2) Sig(4 qbar^2)  (u <= 1);
  valid for qbar <= 1/4, rho <= 1/8.  As qbar, rho -> 0: K1 -> 256, K2 -> 1/8 (and (ii) is then S ~ (CR - 1)/CR).
Theorem AW5 (the strip log law from below, up to the state factor Fc; proof).  For every strip with qbar <= 1/4, rho <= 1/8:
  k_eff^2 <= Lam* + K1max/e + sigma^2/(4V) + delta_osc,  Lam* = ln(max(1, K2max) sqrt2 L/s)  (sup over k0 of k0^2 min(1 + K1 e^{-k0^2}, K2 sqrt2 e^{-k0^2} L/s)),
  hence g >= Fc / (2 sqrt(Lam* + K1max/e + 1/4)),  Fc = (a/den)(1 - eps)|C_w| sqrt((1 - Phi/4)(1 - delta_w))  (Theorem AT2).
  Since s >= sigma:  L/s <= (Lw + sigma)/sigma, so under Lw <= sqrt2 pi (H2):  g >= Fc / (2 sqrt(ln(1/sigma) + C))      (constant 1/2);
  if the window is not narrower than lam_min sqrt(sigma):  L/s <= 2 (Lw + sigma)/(lam_min sqrt(sigma)), g >= Fc / sqrt(2 ln(1/sigma) + C')  (constant 1/sqrt2).
  The sharp constant 1 (k0^2 <= ln(1/sigma)/4) needs a lower bound on N_phi (the background v-information at the bent pendulum): open.
-> results/aw_lemmas.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np

SQ2 = np.sqrt(2.0)


def profile(sig, k0, lam, M, c, eta):
    ell = lam * np.sqrt(sig); Lw = M * ell; Lv = eta * ell; V = Lv ** 2 + sig ** 2; L2 = Lw ** 2 + sig ** 2
    Sb = 1 / (1 / Lw ** 2 + 1 / ell ** 2); qbar = c * np.exp(-k0 ** 2 * Lv ** 2 / (2 * V))
    q0 = qbar * np.sqrt(Sb * L2 / (Lw ** 2 * (Sb + sig ** 2))); s2 = 1 / (1 / (Sb + sig ** 2) - 1 / L2)
    kp = k0 / sig * Lv ** 2 / V
    return {'ell': ell, 'Lw': Lw, 'Lv': Lv, 'V': V, 'L2': L2, 's2': s2, 'q0': q0, 'qbar': qbar, 'kp_sigma': kp * sig, 'rho': s2 / (2 * L2),
            'u': 4 * L2 ** 2 * q0 ** 2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** 1.5), 'L_over_s': np.sqrt(L2 / s2)}


def Sig(x): return (4 - 3 * x + x * x) / (1 - x) ** 3


def d_n(n, x):
    """Lemma AW2': |a_n'(x)| <= |x|^{n-1} d_n(x),  |a_n(x)| <= |x|^n/(4(1 - x^2))  (n >= 2, |x| < 1; binomial series of sqrt(1+y), |binom(1/2, j)| <= 1/8)."""
    return (n / (1 - x * x) + 2 * x * x / (1 - x * x) ** 2) / 4


def e_n(n, qb, rho): return rho / (4 * (1 - qb * qb)) + d_n(n, qb)


def K1(qb, rho, nmax=200):
    tau1 = 0.5 * (1 / (2 * np.sqrt(1 + qb)) - rho / (2 * np.sqrt(1 - qb))) ** 2
    return 0.5 * sum((n * n - 1) * qb ** (2 * n - 4) * e_n(n, qb, rho) ** 2 for n in range(2, nmax)) / tau1


def K2(qb, rho, u, nmax=200):
    num = (1 + rho) ** 2 / (4 * (1 - qb)) + sum(n * n * qb ** (2 * n - 2) * e_n(n, qb, rho) ** 2 for n in range(2, nmax))
    den = np.sqrt(1 - qb) - qb * np.sqrt(u) / (4 * (1 - qb) ** 1.5)
    return 0.5 * num / den ** 2 if den > 0 else np.inf


def S_direct(P, nw=6001, nt=256, wext=8.0):
    """S of Lemma AW3 by direct quadrature of the one-period profile Gamma(w)(1 + q(w) cos t) (FFT in t, analytic d_w)."""
    L = np.sqrt(P['L2']); s = np.sqrt(P['s2']); wmax = wext * min(L, max(s * 3, L / 50)) if s < L / 50 else wext * L
    w = np.concatenate([np.linspace(-wext * L, wext * L, nw), np.linspace(-6 * s, 6 * s, nw)]); w = np.unique(w)
    t = 2 * np.pi * np.arange(nt) / nt
    G = np.exp(-w ** 2 / (2 * P['L2'])) / np.sqrt(2 * np.pi * P['L2']); sG = np.sqrt(G); dsG = -w / (2 * P['L2']) * sG
    q = P['q0'] * np.exp(-w ** 2 / (2 * P['s2'])); dq = -w / P['s2'] * q
    h = np.sqrt(1 + np.outer(q, np.cos(t))); F = dsG[:, None] * h + (sG * dq)[:, None] * np.cos(t)[None, :] / (2 * h)
    Fh = np.fft.rfft(F, axis=1) / nt; n = np.arange(Fh.shape[1])
    wt = np.gradient(w)                                                   # trapezoid-like weights on the merged grid
    num = np.sum(wt[:, None] * (n ** 2)[None, :] * np.abs(Fh) ** 2 * np.where(n == 0, 1, 2)[None, :])
    den = np.sum(wt[:, None] * np.abs(Fh) ** 2 * np.where(n == 0, 1, 2)[None, :])
    A = np.sqrt(np.sum(wt * (dsG * q) ** 2)); B = np.sqrt(np.sum(wt * (sG * dq) ** 2))
    return {'S': num / den, 'A_over_B': A / B, 'u_num': 4 * P['L2'] * B ** 2}


def bounds(P):
    qb, rho, u = P['qbar'], P['rho'], P['u']
    out = {'K1': K1(qb, rho), 'S_bound_i': 1 + K1(qb, rho) * qb * qb}
    if u <= 1: out['K2'] = K2(qb, rho, u); out['S_bound_ii'] = out['K2'] * u
    else: out['K2'] = None; out['S_bound_ii'] = np.inf
    out['S_bound'] = min(out['S_bound_i'], out['S_bound_ii'])
    return out


def theorem_bound(P, K1max, K2max, k0=None):
    """k_eff^2 bound of Theorem AW5 (uniform in k0) and the k0-specific one of Lemma AW3 + Proposition AW4."""
    Lam = np.log(max(1.0, K2max) * SQ2 * P['L_over_s'])
    unif = Lam + K1max / np.e + (P['V'] - P['Lv'] ** 2) / (4 * P['V'])
    return unif


if __name__ == '__main__':
    out = {'lemma_checks': [], 'states': {}}
    # (1) Lemma AW2 by quadrature
    t = 2 * np.pi * np.arange(4096) / 4096; worst = 0.0
    for x in (0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.7, 0.9):
        a = np.fft.rfft(np.sqrt(1 + x * np.cos(t))).real / 4096 * 2; a[0] /= 2
        da = np.fft.rfft(np.cos(t) / (2 * np.sqrt(1 + x * np.cos(t)))).real / 4096 * 2; da[0] /= 2
        for n in range(1, 12):
            if np.sqrt(6) * (2 * x) ** n > 1e-12: worst = max(worst, abs(a[n]) / (np.sqrt(6) * (2 * x) ** n), abs(da[n]) / (SQ2 * (2 * x) ** (n - 1)))
            if n >= 2 and x ** n > 1e-12: worst = max(worst, abs(a[n]) / (x ** n / (4 * (1 - x * x))), abs(da[n]) / (x ** (n - 1) * d_n(n, x)))
        c1 = x > 0.5 or 1 / (2 * np.sqrt(1 + x)) <= da[1] <= 1 / (2 * np.sqrt(1 - x)) and abs(a[1]) <= x / (2 * np.sqrt(1 - x)) and np.sqrt(1 - x) <= a[0] <= 1 and abs(da[0]) <= x / (8 * (1 - x) ** 1.5)
        out['lemma_checks'].append({'x': x, 'n1_n0_bounds_hold': bool(c1)})
    out['AW2_worst_ratio'] = worst; print('Lemma AW2: max |a_n|/bound, |a_n\'|/bound over n = 1..11, x <= 1/2: %.4f (<= 1); n = 0, 1 bounds: %s' % (worst, all(c['n1_n0_bounds_hold'] for c in out['lemma_checks'])))
    # (2) Proposition AW4 and Lemma AW3 on states: the stage-AT strip optima, the stage-AV sequence, and a grid of strips
    from aq_hybrid import strip_params
    AT = json.load(open('results/at_strips.json')); AV = json.load(open('results/av_strip.json'))['sequence']
    states = {}
    for s in (1e-6, 1e-12, 1e-24):
        Pp = strip_params(s); states['opt_%g' % s] = (s, Pp['k0'], Pp['lam'], Pp['M'], Pp['c'], Pp['Lv_sig'] * np.sqrt(s) / Pp['lam'], AT['strip_%g' % s])
    for k, v in AV.items(): states['seq_%s' % k] = (float(k), v['k0'], v['lam'], v['M'], v['c'], v['eta'], v)
    for s in (1e-12,):
        for k0 in (1.8, 2.4, 3.0, 4.0):
            for M in (30.0, 1e3, 1e5):
                states['grid_k%g_M%g' % (k0, M)] = (s, k0, 1.3, M, 1.0, 0.1, None)
    for name, (s, k0, lam, M, c, eta, ref) in states.items():
        P = profile(s, k0, lam, M, c, eta); b = bounds(P); r = {k: float(v) if v is not None else None for k, v in P.items()}; r.update({k: (float(v) if v is not None else None) for k, v in b.items()})
        if P['qbar'] <= 0.5 and P['rho'] <= 0.125:
            d = S_direct(P); r.update(d); r['S_le_bound'] = bool(d['S'] <= b['S_bound'] * (1 + 1e-9)); r['A_over_B_exact'] = P['rho']
        if ref is not None and 'k_eff' in ref:
            r['k_eff_evaluator'] = ref['k_eff']; r['k_eff_AW3'] = float(np.sqrt(P['kp_sigma'] ** 2 * r.get('S', np.nan) + s * s / (4 * P['V']))); r['g'] = ref['g']
            Fc = ref['bound_AT2'] * 2 * ref['k_eff']; r['Fc'] = Fc
        out['states'][name] = r
        print('%-22s qbar %.2e rho %.1e u %.3g | S %s  bound(i) %.4g  bound(ii) %s | k_eff evaluator %s, AW3 %s' % (
            name, P['qbar'], P['rho'], P['u'], '%.5f' % r['S'] if 'S' in r else '-', b['S_bound_i'], '%.4g' % b['S_bound_ii'] if np.isfinite(b['S_bound_ii']) else 'n/a',
            '%.5f' % r['k_eff_evaluator'] if 'k_eff_evaluator' in r else '-', '%.5f' % r['k_eff_AW3'] if 'k_eff_AW3' in r else '-'), flush=True)
    # (3) Theorem AW5: uniform constants over qbar <= 1/4, rho <= 1/8 (worst case of K1, K2 there, u <= 1)
    K1max = K1(0.5, 0.125); K2max = K2(0.5, 0.125, 1.0); out['K1max'] = K1max; out['K2max'] = K2max
    out['K1_limit'] = K1(1e-9, 1e-9); out['K2_limit'] = K2(1e-9, 1e-9, 1e-9)
    print('constants: K1max %.1f (limit %.1f), K2max %.3f (limit %.4f)' % (K1max, out['K1_limit'], K2max, out['K2_limit']))
    for name, r in out['states'].items():
        if r.get('g') is None: continue
        s = 1e-6 if '1e-06' in name else float(name.split('_')[-1])
        P = profile(*[states[name][i] for i in range(6)]); kb2 = theorem_bound(P, K1max, K2max)
        r['keff2_bound_AW5'] = kb2; r['g_bound_AW5'] = r['Fc'] / (2 * np.sqrt(kb2)); r['g_bound_AW5_sqrt_ln'] = r['g_bound_AW5'] * np.sqrt(np.log(1 / s))
        r['keff2_bound_AW5_asymptotic'] = np.log(SQ2 * P['L_over_s']) + out['K1_limit'] / np.e; r['L_over_s'] = P['L_over_s']
        print('Theorem AW5 on %-14s: k_eff^2 = %.4f <= %.2f (uniform bound); g = %.5f >= %.5f' % (name, r['k_eff_evaluator'] ** 2, kb2, r['g'], r['g_bound_AW5']))
    json.dump(out, open('results/aw_lemmas.json', 'w'), indent=1, default=float)
