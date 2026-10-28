"""Stage AN, plan B: exact theta decomposition of the staggered double row and its remainder (Lemma AN4), and the assembled bound
(Theorem AN5) for Conjecture 1 on the staggered rows.  Units t = 1.

Exact (Poisson): a dotted row of spacing a, blurred, is (m) phi(v - v_row) Theta(w - o), Theta = 1 + 2 sum_n q^{n^2} cos(n k w),
k = 2 pi/a, q = e^{-k^2/2}.  Rows at v = D -+ d with offsets 0 and a/2 give
  p = P0 (1 + eps),  eps = 2 g O(w) + 2 h E(w),  O = sum_{n odd} q^{n^2} cos(nkw),  E = sum_{n even >= 2} q^{n^2} cos(nkw),
with the v-functions g = (P_+ - P_-)/P0 (signed) and h = (P_+ + P_-)/P0 (unsigned) of an_vproblem.py.  Odd and even harmonics are
orthogonal over a period, so with mu_j = <(d_w^j O)^2>, mu^e_j = <(d_w^j E)^2> the linear parts are exact:
  int P0 (d_w eps)^2 = 4(mu_1 I_g + mu^e_1 I_h),  int P0 (d_w^2 eps)^2 = 4(mu_2 I_g + mu^e_2 I_h),
  int P0 (d_v d_w eps)^2 = 4(mu_1 I_g' + mu^e_1 I_h'),  int P0 (d_v eps)^2 = 4(mu_0 I_g' + mu^e_0 I_h').
Remainder: |eps| <= eb = 2(sum_odd q^{n^2} + sum_even q^{n^2}),  |s_w| <= vs = 2(sum_odd nk q^{n^2} + sum_even nk q^{n^2})/(1 - eb).
-> results/an_theta.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from an_vproblem import exact, proved, staggered_lb


def qsums(a, nmax=41):
    k = 2 * np.pi / a; q = np.exp(-k * k / 2); n = np.arange(1, nmax + 1); odd = n % 2 == 1; ev = (n % 2 == 0)
    S = lambda m, p, sel: float(np.sum((n[sel] * k) ** p * q ** (m * n[sel] ** 2)))
    return {'k': k, 'q': q, 'sig0': S(1, 0, odd), 'sig1': S(1, 1, odd), 'e0': S(1, 0, ev), 'e1': S(1, 1, ev),
            'mu0': S(2, 0, odd) / 2, 'mu1': S(2, 2, odd) / 2, 'mu2': S(2, 4, odd) / 2,
            'me0': S(2, 0, ev) / 2, 'me1': S(2, 2, ev) / 2, 'me2': S(2, 4, ev) / 2}


def assemble(D, a, d=1.0, z=0.1):
    lb = staggered_lb(D, z, d); e = exact(D, d, lb); p = proved(D, d, lb); Q = qsums(a)
    s0 = e['lI_g']; Ig, Ih, Igp, Ihp = (np.exp(e[k] - s0) for k in ('lI_g', 'lI_h', 'lD_g', 'lD_h')); lMs = e['lM'] - s0   # common log scale
    eb = 2 * (Q['sig0'] + Q['e0']); vs = 2 * (Q['sig1'] + Q['e1']) / (1 - eb)
    Jlin = 4 * (Q['mu1'] * Ig + Q['me1'] * Ih)
    Nw2 = 4 * (Q['mu2'] * Ig + Q['me2'] * Ih); Nv2 = 4 * (Q['mu1'] * Igp + Q['me1'] * Ihp); Ev2 = 4 * (Q['mu0'] * Igp + Q['me0'] * Ihp)
    lead_tR = (Nw2 + Nv2) / Jlin; lead_ln = float(lMs - np.log(Jlin))
    c = np.sqrt(1 + eb) / (1 - eb); cl = np.sqrt(1 - eb) / (1 + eb)
    up_w = c * np.sqrt(Nw2) + vs * np.sqrt(Jlin / (1 - eb)); lo_w = max(0.0, cl * np.sqrt(Nw2) - vs * np.sqrt(Jlin / (1 - eb)))
    up_v = c * (np.sqrt(Nv2) + vs * np.sqrt(Ev2)); lo_v = max(0.0, cl * np.sqrt(Nv2) - c * vs * np.sqrt(Ev2))
    tR_up = (up_w ** 2 + up_v ** 2) * (1 + eb) / Jlin; tR_lo = (lo_w ** 2 + lo_v ** 2) * (1 - eb) / Jlin
    ln_lo = float(lMs - np.log(Jlin / (1 - eb))); ln_up = float(lMs - np.log(Jlin / (1 + eb)))
    # Theorem AN5: tR <= B' ln(1/tJ) + A'
    cstar = max(np.sqrt(Q['mu1'] / Q['mu2']), np.sqrt(Q['me1'] / Q['me2']), np.sqrt(Q['mu0'] / Q['mu1']), np.sqrt(Q['me0'] / Q['me1']))
    one_d = (1 + eb) ** 2 / (1 - eb) ** 2 * (1 + vs * cstar) ** 2
    A1 = Q['mu2'] / Q['mu1'] + p['B_g'] * (np.log(Q['mu1']) + p['a_g']); A2 = Q['me2'] / Q['me1'] + p['B_h'] * (np.log(Q['me1']) + p['a_h'])
    B = max(p['B_g'], p['B_h']); Bp = one_d * B; Ap = one_d * (B * (np.log(8) - np.log(1 - eb)) + max(A1, A2))
    r = {'D': D, 'a': a, 'q': Q['q'], 'eps_bar': eb, 'sup_sw': vs, 'lead_tR': lead_tR, 'lead_ln1_tJ': lead_ln, 'tR_bracket': [tR_lo, tR_up],
         'ln1_tJ_bracket': [ln_lo, ln_up], 'weight_even_in_J': 4 * Q['me1'] * Ih / Jlin, 'one_plus_delta': one_d, 'B_g': p['B_g'], 'B_h': p['B_h'],
         'A1': A1, 'A2': A2, 'Bprime_AN5': Bp, 'Aprime_AN5': Ap, 'Rq_g': e['Rq_g'], 'Rq_h': e['Rq_h']}
    return {k: (float(v) if not isinstance(v, list) else [float(u) for u in v]) for k, v in r.items()}


if __name__ == '__main__':
    st = json.load(open('results/al_staggered.json')); out = {'cases': {}, 'large_D': {}}
    for k, v in st.items():
        r = assemble(v['D'], v['a']); o5 = v['taus']['0.5']
        r['exact_tR'] = o5['tR']; r['exact_ln1_tJ'] = o5['ln1_tJ']
        r['in_bracket'] = bool(r['tR_bracket'][0] <= r['exact_tR'] <= r['tR_bracket'][1] and r['ln1_tJ_bracket'][0] <= r['exact_ln1_tJ'] <= r['ln1_tJ_bracket'][1])
        r['rel_lead_tR'] = abs(r['lead_tR'] / r['exact_tR'] - 1); r['abs_lead_ln'] = abs(r['lead_ln1_tJ'] - r['exact_ln1_tJ'])
        r['T2_AN5_holds_on_exact'] = bool(r['exact_tR'] <= r['Aprime_AN5'] + r['Bprime_AN5'] * r['exact_ln1_tJ'])
        r['margin_B1_exact'] = r['exact_tR'] - r['exact_ln1_tJ']
        out['cases'][k] = r
        print(k, 'q %.2e eps %.3g | tR exact %.5f lead %.5f bracket [%.4f, %.4f] | ln exact %.4f lead %.4f bracket [%.4f, %.4f] in %s | even weight %.2e | B_g %.3f B_h %.3f -> B\' %.3f A\' %.2f, margin(B\'=1) %.2f' % (
            r['q'], r['eps_bar'], r['exact_tR'], r['lead_tR'], *r['tR_bracket'], r['exact_ln1_tJ'], r['lead_ln1_tJ'], *r['ln1_tJ_bracket'], r['in_bracket'],
            r['weight_even_in_J'], r['B_g'], r['B_h'], r['Bprime_AN5'], r['Aprime_AN5'], r['margin_B1_exact']), flush=True)
    for a in (2.0, 1.5, 1.2):
        for D in (8.0, 14.0, 20.0, 28.0, 40.0, 60.0):
            r = assemble(D, a); out['large_D']['%g_%g' % (D, a)] = r
            print('  a %g D %g: B\' %.4f A\' %.2f (lead tR %.2f ln %.2f, lead margin B\'=1 %.2f, even weight %.1e)' % (a, D, r['Bprime_AN5'], r['Aprime_AN5'], r['lead_tR'], r['lead_ln1_tJ'], r['lead_tR'] - r['lead_ln1_tJ'], r['weight_even_in_J']))
    json.dump(out, open('results/an_theta.json', 'w'), indent=1)
