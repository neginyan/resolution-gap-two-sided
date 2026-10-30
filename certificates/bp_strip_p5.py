"""Stage BP §2: strips without (P5) — every x > 0 (computer-assisted proof).  With BN1 (1e-12 <= x < 1) and the v1.1 chain (x >= 1) this removes (P5)
for strips completely.  A strip with g < G (c = G/0.9 <= c* = 0.134/0.9; then (a) < c and (a') < c) and 0 < x < 1 leads to a contradiction in
each of four cases:

 Case A1 (tau > 80, x <= 250 sigma^2):  (a) >= c.  tau > 80 gives b <= x/80; the exact W formula (Lemma 20 (iii), one harmonic) gives
     W <= x[(1 + delta_tau) q^2 g_s/(2(1 - Q)) + g1 q e_1]/(1 - g1 q e_1) <= x/72.000 (q <= 1/4, g_s <= 1/3); so b + W <= 0.02639 x.  (a) < c gives
     sigma J_ww < c theta (b + W), and J_ww >= 1/(kappa_w L^2) >= sigma/(kappa_w (1 + sigma^3)) (exact w-marginal Cramer-Rao, (P0)), hence
     x > sigma^2/(0.02639 c theta kappa_w (1 + sigma^3)) = 254.5 sigma^2.
 Case A2 (tau > 80, x > 250 sigma^2):  Lemma BJ1 for strips with cells covering [0, 1] (a cell [0, 1e-30] included; every function has a limit at
     x = 0), and the theta-remainder of Lemma BL3 absorbed into the b-coefficient: (a) < c gives L^2 > 254 sigma/x, so 1 - C >= 0.2 min(254 sigma/x, 1)
     (Lemma BL4), and sigma b E_R <= eps_R b min(1 - C, 1/2) with eps_R <= 5 (A0 + B0 + C0) (E_R <= A0/sqrt(x) + B0 sigma/x + C0 on tau >= 80,
     beta <= sigma tau/sqrt(2x) <= tau/sqrt(500)).  BJ1 is run with eps = eps_in + eps_R; then F_c >= 0.9 A_min and Theorem 1 with Lemma BL5 and AX4-P
     (S_per <= 8 kappa2 (1 + Q)) give g >= 0.9 A_min/(2 sqrt((1 + eps)(8 kappa2 (1 + Q) + 1/4))) > 0.134.
 Case B (tau <= 80, b >= c' sigma, x < 1e-12; c' = 2e-3):  Lemma BK1 for strips (the BO-corrected cell) on b-cells [b_a, b_b] in [2e-27, 1] and x-cells
     covering (0, min(1e-12, 80 b_b)] (one cell starting at x = 0), each evaluated at sigma = min(1e-24, b_a/c') (the cell inequality gets worse as sigma
     grows).  For b < 2e-27: at fixed r = sigma/b and tau = x/b, main/b, Psi(L1) and the 'increasing' test do not decrease, and Ea/b, Eb/b, rhs/b do not
     increase, as b decreases (each is a polynomial in (b, sqrt b) with non-negative coefficients times functions of (r, tau)), so the margin at b is at
     least the margin at (b0, sigma b0/b) with sigma b0/b <= b0/c' = 1e-24: covered by the first b-cell.  ((1e-12 <= x < 1 is BN1.)
 Case C (tau <= 80, b < c' sigma):  Cauchy-Schwarz on every w-slice with g = 2 sin(m_z/2) (the step of the window inequality):
     int P_w s_v^2 phi >= (int P_w cos(m_z/2)(alpha + psi'))^2/(2 int P_w),  int N_V h e^{i alpha v/2} = e^{-alpha^2 V/8} + (q/2)(e^{-(alpha/2-k')^2 V/2}
     + e^{-(alpha/2+k')^2 V/2}) >= e^{-alpha^2 V/8} (q >= 0 for a strip).  (a) < c gives b + W >= sigma^2/(c theta kappa_w (1 + sigma^3)) (exact w-marginal
     Cramer-Rao, any tau) and W <= x/3 <= 80 b/3 (exact W formula), so V = sigma^2/b <= 27.7 c theta kappa_w (1 + sigma^3); then (|J_w| >= |A_w| - E_w on
     every slice, so J_w^2 >= A_w^2 - 2|A_w| E_w also where cos(w/2sqrt2) < 0; stage BQ corrected the error term, which BP had divided by (1 + q)^2)
     2 N_phi/sigma^2 >= alpha^2 e^{-alpha^2 V/4}/(4 (1 + q)^2) - alpha E  and  (a') = N_phi/(sigma den) >= that/(2 sigma 27.7 theta b) >= c for b <= c' sigma.
     Outward-rounded interval arithmetic of the same constants: bq_proofs.case_C.
Paper v1.2 numbering: case B = Lemma 29(ii) with Lemma 30 (b < 2e-27), case C = Lemma 31, case A1 = Lemma 32, case A2 = Lemma 33, assembly = Theorem 34.
-> results/bp_strip_p5.json, results/bp_strip_caseB.jsonl (one line per b-cell, resumable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np

CST = 0.134 / 0.9; SIG = 1e-24; EPS = 1e-6; CP = 2e-3; B0 = 2e-27; X_BN = 1e-12
QS, RS = 0.25, 0.125; G1 = np.sqrt(2 * RS / (1 + 2 * RS)); RSH = 2 * RS / (1 + 2 * RS)
FN = 'results/bp_strip_caseB.jsonl'


def case_A1(c=CST, sig=SIG):
    """(a) < c forces x > T sigma^2 with T > 250 (paper v1.2 Lemma 32; exact w-marginal Cramer-Rao and the exact W formula of Lemma 20 (ii), (iii) for one
    harmonic at tau > 80; interval version: bq_proofs.case_A1)."""
    th = 1 / (1 - c * sig); e1 = np.exp(-40.0); delta = 2 * e1 * (1 + 1e-50)
    kw = (1 + QS * G1 * RSH * e1) / (1 - QS * G1 * e1)
    Wc = ((1 + delta) * QS ** 2 * (1 / 3) / (2 * (1 - QS)) + G1 * QS * e1) / (1 - G1 * QS * e1)     # W <= Wc x (g_s <= 1/3)
    bw = 1 / 80 + Wc
    thr = 1 / (bw * c * th * kw * (1 + sig ** 3))                                  # x > thr sigma^2 if (a) < c
    return {'b_plus_W_over_x': bw, 'threshold_over_sigma2': thr, 'case_A1_x_max_over_sigma2': 250.0, 'ok': bool(thr > 250.0)}


def eps_R_bound(tau=80.0, sig=SIG):
    """E_R <= A0/sqrt(x) + B0 sigma/x + C0 for a strip, x in (0, 1], tau >= 80, beta <= tau/sqrt(500); eps_R <= 5 (A0 + B0 + C0) (see the docstring)."""
    import bl_theta as T
    import bj_lemma as J
    nu = T.nus(tau, tau / np.sqrt(500.0))
    Q = N1 = 0.25
    A, B = J.Gam1_parts(0.5); gmax = 0.25 * (A + B)                                # gamma_P <= q Gamma1(x) <= (A + B)/4 on x <= 1 (strip: A, B constants)
    ccmax = (N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)
    A0 = tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2) * (1 - Q)) + tau * nu[1] * (1 + Q) * gmax / np.sqrt(2)
    B0 = 0.375 * tau * Q * nu[0] + Q * tau ** 2 * nu[2] / 8 + tau * nu[0] * (1 + Q) * gmax + Q * tau ** 3 * nu[4] / 8
    C0 = (1 + Q) * (1 + Q / 2) * ccmax * nu[0] + tau * nu[1] * N1 * ccmax + tau * Q * nu[2] * (1 + Q) * ccmax / 2 + tau * Q * nu[2] / (2 * (1 - nu[0]))
    return {'A0': A0, 'B0': B0, 'C0': C0, 'eps_R': 5 * (A0 + B0 + C0), 'nu0': nu[0]}


def case_A2(sig=SIG, c=None):
    import bj_lemma as J
    J.eta = lambda p, x: 0.0; J.Z2 = lambda x: 0.0
    J.CELLS = np.concatenate([[0.0], np.geomspace(1e-30, 0.01, 6001), np.linspace(0.01, 1.0, 4001)[1:]])
    eR = eps_R_bound()['eps_R']
    import bl_bj1 as BL
    c = BL.CSTAR if c is None else c
    # strip_bj1 sets the same strip functions; its x-cells are replaced by J.CELLS above (re-set after the call it makes)
    r = BL.constants(sig, c, EPS + eR)
    QS_, UP, DN = J.QS, J.UP, J.DN; eps = EPS + eR; th = 1 / (1 - c * sig)
    s2g = np.sqrt(3 * sig ** 4 + 2 * sig); ep = r['step2b']['e_prime']
    Cq = 4 * (1 + QS_) * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * UP
    Aa, Bb = J.Gam1_parts(0.5)
    gst = J.sup_cells(lambda a, b: min(Cq * (np.sqrt(b) * Aa + Bb) ** 2 * UP, J.Gam1_up(a, b) * J.m(a)))
    aw2b = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * DN
    r['step2b'] = {'e_prime': ep, 'C_q': Cq, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c)}
    r['ok'] = bool(r['step2b']['contradiction'] and r['step3a']['contradiction'] and r['step3b']['contradiction'] and r['A_min'] >= 1.02 * c and r['A_min'] > 0.93)
    r['eps_used'] = eps; r['eps_R'] = eR; r['x_cells'] = len(J.CELLS) - 1
    kap2 = (1 + RS + QS / (2 * (1 - QS))) ** 2 / (8 * (1 - QS)); k2 = (1 + eps) * (8 * kap2 * (1 + QS) + 0.25)
    r['conclusion'] = {'k_eff2_max': k2, 'g_lower': 0.9 * r['A_min'] / (2 * np.sqrt(k2)), 'ratio_to_0.134': 0.9 * r['A_min'] / (2 * np.sqrt(k2)) / 0.134}
    return r, BL


def case_B(resume=True, nb_dec=10, nx=40):
    import bk_narrow_lemma as BK
    BK.NN = np.array([1.0])
    have = {}
    if resume and os.path.exists(FN):
        for line in open(FN):
            if line.strip():
                try: d = json.loads(line); have[d['j']] = d
                except json.JSONDecodeError: pass
    for j, (ba, bb) in enumerate(caseB_bcells(nb_dec)):
        if j in have: continue
        d = caseB_cell(j, ba, bb, nx)
        with open(FN, 'a') as fh: fh.write(json.dumps(d) + '\n')
        have[j] = d
    BK.NN = np.arange(1, 40, dtype=float)
    return [have[j] for j in sorted(have)]


def caseB_bcells(nb_dec=10):
    bs = np.geomspace(B0, 1.0, int(round(np.log10(1 / B0) * nb_dec)) + 1)
    return list(zip(bs[:-1], bs[1:]))


def caseB_cell(j, ba, bb, nx=40):
    """one b-cell of case B (one JSON line); expects bk_narrow_lemma.NN = [1] (one harmonic)."""
    import bk_narrow_lemma as BK, bp_audit as AU
    sig = min(SIG, ba / CP)
    xhi = min(X_BN, 80 * bb)
    xs = np.concatenate([[0.0], np.geomspace(xhi * 1e-8, xhi, nx + 1)])
    ok = True; wm = np.inf; n = 0; npt = 0; nviol = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xa / bb > 80: continue                                              # outside tau <= 80
        q = AU.bk_cell_full(BK, xa, xb, ba, bb, True, CST, sig)
        # acceptance: f_big > 0 covers every L with gamma L >= 0.71; if gamma L1 < 0.71 also f(L1) > 0 and f increasing on [L1, 0.71/gamma]
        r = {'ok': bool(q['f_big'] > 0 and (q['gL1'] >= 0.71 or (q['f_L1'] > 0 and q['increasing']))), 'margin': q['margin']}
        ok &= r['ok']; wm = min(wm, r['margin']); n += 1
        # in-cell audit (BP §1 method): the cell margin must not exceed the margin at interior points
        for u, v in ((0.5, 0.5), (1e-9, 1 - 1e-9), (1 - 1e-9, 1e-9), (0.25, 0.75), (0.75, 0.25)):
            x = xa + (xb - xa) * u; b = ba + (bb - ba) * v
            if x / b > 80: continue
            rp = AU.bk_cell_full(BK, x, x, b, b, True, CST, sig); npt += 1
            if rp['margin'] < r['margin'] * (1 - 1e-12) or rp['f_big'] < q['f_big'] * (1 - 1e-12) - 1e-300: nviol += 1
    return {'j': j, 'ba': ba, 'bb': bb, 'sigma': sig, 'x_cells': n, 'ok': bool(ok), 'margin': float(wm), 'audit_points': npt, 'audit_violations': nviol}


def case_C(c=CST, sig=SIG):
    """explicit constants of Case C at the worst corner (each term monotone: larger sigma, c, q, x, V worse)."""
    th = 1 / (1 - c * sig); q = QS
    kw = (1 + q * G1 * RSH) / (1 - q * G1)                                         # H0 <= 1 (strip, any tau)
    bW = 1 + 80 / 3                                                                # b + W <= (1 + 80/3) b  (W <= x/3, x <= 80 b)
    Vmax = bW * c * th * kw * (1 + sig ** 3)                                       # V = sigma^2/b
    b = CP * sig; al = (1 - b) / np.sqrt(2); x = 80 * b                            # worst: b, x at their largest
    Gam1 = (x * (1 + 2 * q) + np.sqrt(x) * (1 + 2 * q) / np.sqrt(np.e)) / (1 - q) ** 2; gam = q * Gam1
    cchi = (np.sqrt(x) * q / (1 - q) + np.sqrt(2 / np.pi) + q * np.exp(-0.5) / (1 - q)) / np.sqrt(2)
    Eterm = (1 / np.sqrt(2)) * (1 + q) * sig * cchi / 2 + (1 + q) * gam / np.sqrt(2)     # alpha <= 1/sqrt2 in the error term
    main = al * al * np.exp(-al * al * Vmax / 4) / (4 * (1 + q) ** 2)
    twoNphi_over_sig2 = main - (1 / np.sqrt(2)) * Eterm                           # J_w^2 >= A_w^2 - 2|A_w| E_w; int Gamma cos^2(w/2sqrt2) >= 1/2; 1 <= Z <= 1 + q
    a_prime_times_b_over_sigma = twoNphi_over_sig2 / (2 * bW * th)                 # (a') >= this * sigma/b
    a_prime_lower = a_prime_times_b_over_sigma / CP                                # at b = c' sigma (worst)
    return {'kappa_w': kw, 'V_max': Vmax, 'main': main, 'E_term': Eterm, 'two_Nphi_over_sigma2': twoNphi_over_sig2,
            'a_prime_lower_at_b_eq_cp_sigma': a_prime_lower, 'c': c, 'ratio': a_prime_lower / c, 'ok': bool(a_prime_lower > c)}


def case_C_numeric():
    """numerical check (not part of the proof): direct quadrature in Case C, (a') against the bound."""
    import bk_direct as Dd
    out = []
    for b_rel in (1e-3, 4e-3):
        for x_rel in (1e-3, 0.5, 79.0):
            for L2 in (1e-2, 1.0, 50.0):
                sig = 1e-24; b = b_rel * sig; x = x_rel * b
                try: r = Dd.branches(sig, b, x, [0.25], L2, L2 / 8)
                except Exception as e: continue
                out.append({'b_over_sigma': b_rel, 'tau': x / b, 'L2': L2, 'a_N': r['a_N'], 'a_w': r['a_w'], 'bound_a_N': case_C()['a_prime_lower_at_b_eq_cp_sigma'] * CP / b_rel})
    return out


if __name__ == '__main__':
    np.seterr(all='ignore')
    out = {'case_A1': case_A1()}; print('A1:', out['case_A1'], flush=True)
    out['eps_R'] = eps_R_bound(); print('eps_R:', out['eps_R'], flush=True)
    for e in (24, 48, 100, 300):
        r, BL = case_A2(10.0 ** (-e)); out['case_A2_1e-%d' % e] = r
        print('A2 sigma 1e-%d: ok %s | A_min %.6f | 2(i) %.2e | 3(i) %.4f | 3(ii) %.2e | g >= %.4f (x %.3f)' % (
            e, r['ok'], r['A_min'], r['step2b']['a_lower_over_c'], r['step3a']['margin'], r['step3b']['W_lower_over_W_cap'], r['conclusion']['g_lower'],
            r['conclusion']['ratio_to_0.134']), flush=True)
    out['case_A2_step3a_interval'] = BL.step3a_interval()
    rows = case_B()
    out['case_B'] = {'b_cells': len(rows), 'cells': sum(d['x_cells'] for d in rows), 'all_ok': all(d['ok'] for d in rows), 'worst_margin': min(d['margin'] for d in rows),
                     'audit_points': sum(d['audit_points'] for d in rows), 'audit_violations': sum(d['audit_violations'] for d in rows)}
    print('B:', out['case_B'], flush=True)
    out['case_C'] = case_C(); print('C:', out['case_C'], flush=True)
    out['case_C_numeric'] = case_C_numeric()
    print('C numeric: min (a\')/c = %.3g over %d states' % (min(d['a_N'] for d in out['case_C_numeric']) / CST, len(out['case_C_numeric'])))
    out['all_ok'] = bool(out['case_A1']['ok'] and all(out['case_A2_1e-%d' % e]['ok'] and out['case_A2_1e-%d' % e]['conclusion']['ratio_to_0.134'] > 1 for e in (24, 48, 100, 300))
                         and out['case_A2_step3a_interval']['g_gt_rhs'] and out['case_B']['all_ok'] and out['case_B']['audit_violations'] == 0 and out['case_C']['ok'])
    print('strips without (P5):', out['all_ok'])
    json.dump(out, open('results/bp_strip_p5.json', 'w'), indent=1, default=float)
