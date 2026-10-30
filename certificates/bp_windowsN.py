"""Stage BP §3: windows with at most N harmonics, 1e-12 <= x < 1 (as Lemma BN1 for strips).  Computer-assisted proof for the N that close.

Claim (Lemma BP-N).  A periodic window with (P0)-(P4), (P6), (P7) whose profile has a_n = 0 for n > N, and 1e-12 <= x < 1, has g >= G(l).
Proof (as BN1; g < G, c = G/0.9; both branches small):
 (1) Lemma BK1 with N harmonics (tau <= 80): the corrected cell (BO fix) with every harmonic sum capped by (P3) Qbar <= 1/4:
     sum q0_n e_n <= min(q sum_{n<=N} dec_n e_n, Q e_1), sum q0_n |1 - n^2 tau| e_n <= min(q sum dec_n sup f, Q max_n sup f),
     sum n^2 q0_n e_n <= min(q sum n^2 dec_n e_n, Q max_{n<=N} n^2 e_n), N1 = sum n q0_n <= min(q sum n dec_n, N Q),
     sum n^2 q0_n^2 <= min(q^2 sum n^2 dec_n^2, N^2 q Q)   (q0_n <= q dec_n by (P2) and K >= x; sum q0_n <= Qbar <= 1/4).
 (2) Lemma BJ1 with N harmonics (tau > 80): bl_bj1.constants with eta_p(x) = sum_{n=2}^N n^p e^{-(n^2-1)x/2}, Z2(x) = sum_{n=2}^N n^2 e^{-(n^2-1)x}
     (finite at x = 0) on x-cells covering [1e-12, 1]; step 2(i) with the per-x product C_q Gamma1(x)^2/x <= C_q (sqrt(b) A(a) + B(a))^2.
     The theta remainder E_R with N1 <= N Q, gamma_P <= q Gamma1(x): sup over [1e-12, 1] at tau >= 80 must be <= eps_in.
 (3) Theorem 1 with Lemma BL5 and AX4-P: u2/u0 <= sum n^2 q0_n^2/sum q0_n^2 <= 1 + Z2(x) (common envelope), so
     k_eff^2 <= (1 + eps)(8 kappa2 (1 + Q) x (1 + Z2(x)) + 1/4) and g >= 0.9 A_min/(2 k_eff) must exceed 0.134 for every x < 1.
Paper v1.2 numbering: Lemma 29(iii) (cells), Lemma 35 (BJ1), Theorem 36, Table 10.
Stage BQ: the factor of theta_tau in W_N is now delta_eff(N, tau) (see there); BP had used 2 sum_{n<60} e_n, valid only through bound (b) at small
tau and only up to the tail at larger tau.  All N were re-run.
-> results/bp_windowsN.json (and results/bp_windowsN_bk1.jsonl, one line per (N, x-cell), resumable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np

QS, RS = 0.25, 0.125; G1 = np.sqrt(2 * RS / (1 + 2 * RS)); RSH = 2 * RS / (1 + 2 * RS); GS = np.sqrt(RS / (1 + RS))
CST = 0.134 / 0.9; SIG = 1e-24; EPS = 1e-6
FN = 'results/bp_windowsN_bk1.jsonl'
f = lambda u: np.abs(1 - u) * np.exp(-u / 2)


def fsup(ua, ub):
    return np.maximum(np.maximum(f(ua), f(ub)), np.where((ua <= 3.0) & (3.0 <= ub), f(3.0), 0.0))


def delta_eff(N, tau):
    """A valid factor 1 + delta with <theta_tau h_t^2> <= (1 + delta) <h_t^2> for a profile with harmonics n <= N (stage BQ).  Two bounds:
    (a) sup theta_tau <= 1 + 2 sum_{n>=1} e_n (all n, explicit tail beyond n = 59);  (b) Toeplitz: <theta_tau sin(nt) sin(mt)> = (e_|n-m| - e_{n+m})/2, so
    with a_n = n q_n (any signs) and Cauchy-Schwarz on shifted sequences, <theta_tau h_t^2> <= (1/2)[1 + 2 sum_{k=1}^{N-1} e_k + sum_{s=3}^{2N-1} e_s] sum a_n^2
    (the diagonal -e_{2n} terms are dropped).  (b) is what makes small tau harmless; BP used 2 sum_{n<60} e_n, which is not an upper bound of (a)."""
    from math import erfc, sqrt, pi
    e = lambda k: float(np.exp(-k * k * tau / 2))
    a = 2 * sum(e(n) for n in range(1, 60)) + (2 * sqrt(pi / (2 * tau)) * erfc(59 * sqrt(tau / 2)) if tau > 0 else np.inf)
    b = 2 * sum(e(k) for k in range(1, N)) + sum(e(t) for t in range(3, 2 * N))
    return min(a, b) * (1 + 1e-12)


def cellN(N, xa, xb, ba, bb, c=CST, sig=SIG):
    n = np.arange(1, N + 1, dtype=float); th = 1 / (1 - c * sig); ta, tb = xa / bb, xb / ba
    q = min(QS, 2 * np.exp(-xa / 2)); dec = np.exp(-(n * n - 1) * xa / 2); en = np.exp(-n * n * ta / 2)
    qH0 = min(q * float(np.sum(dec * en)), QS * float(en.max()))
    qH1 = min(q * float(np.sum(dec * fsup(n * n * ta, n * n * tb))), QS * float(fsup(n * n * ta, n * n * tb).max()))
    qH2 = min(q * float(np.sum(n * n * dec * en)), QS * float((n * n * en).max()))
    N1 = min(q * float(np.sum(n * dec)), N * QS)
    S2 = min(q * q * float(np.sum(n * n * dec * dec)), N * N * q * QS)            # sum n^2 q0_n^2
    delta = delta_eff(N, ta)
    kw = (1 + qH0 * G1 * RSH) / (1 - qH0 * G1); ks = (1 + qH1) / (1 - qH0)
    WN = xb * ((1 + delta) * S2 * GS / (2 * (1 - QS)) + G1 * qH2) / (1 - G1 * qH0)
    L12 = sig / (c * th * kw * (bb + WN)); gam2 = (1 - c * th * kw * (bb + WN) * sig) ** 2 / 2
    Psi1 = -np.expm1(-gam2 * L12 / 2)
    Np = N1 ** 2 / (1 - QS) ** 2; tauN = tb * Np
    d = sig * np.sqrt(xb) * N1 / (1 - QS) + sig * QS * np.exp(-0.5) / (1 - QS)
    Ea = (1 + QS) / np.sqrt(2) * (sig * np.sqrt(2 * bb) * np.sqrt((1 + tauN) * (6 + 2 * tauN)) + 2 * bb * d * (1 + tauN))
    Eb = (1 + QS) * (sig * sig * (3 + tauN) + bb * d * d * (1 + tauN))
    main_fac = ba * (1 - qH0 * G1) / ks
    rhs = 2 * c * sig * th * (bb + WN) * (1 + G1 * qH0)
    gL1 = np.sqrt(gam2 * L12)
    f_L1 = main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb - rhs
    inc = bool(main_fac * gL1 * np.exp(-gam2 * L12 / 2) >= Ea)
    f_big = main_fac * (-np.expm1(-0.71 ** 2 / 2)) - Ea * 0.71 - Eb - rhs
    ok = bool(f_big > 0 and (gL1 >= 0.71 or (f_L1 > 0 and inc)) and qH0 * G1 < 1 and qH0 < 1)
    return {'ok': ok, 'margin': float((main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb) / rhs), 'f_big': float(f_big)}


def bk1N(N, nx=120, nb=121):
    have = {}
    if os.path.exists(FN):
        for line in open(FN):
            if line.strip():
                try: d = json.loads(line); have[(d['N'], d['j'])] = d
                except json.JSONDecodeError: pass
    xs = np.geomspace(1e-12, 1.0, nx + 1); rows = []
    for j, (xa, xb) in enumerate(zip(xs[:-1], xs[1:])):
        if (N, j) in have: rows.append(have[(N, j)]); continue
        d = bk1N_row(N, j, xa, xb, nb)
        with open(FN, 'a') as fh: fh.write(json.dumps(d) + '\n')
        rows.append(d)
    return rows


def bk1N_row(N, j, xa, xb, nb=121):
    bs = np.geomspace(xa / 80, 1.0, nb); ok = True; wm = np.inf; nv = 0; npt = 0
    for ba, bb in zip(bs[:-1], bs[1:]):
        r = cellN(N, xa, xb, ba, bb); ok &= r['ok']; wm = min(wm, r['margin'])
        for u, v in ((0.5, 0.5), (1e-9, 1 - 1e-9), (1 - 1e-9, 1e-9)):            # in-cell audit (BP §1)
            x = xa + (xb - xa) * u; b = ba + (bb - ba) * v; rp = cellN(N, x, x, b, b); npt += 1
            if rp['margin'] < r['margin'] * (1 - 1e-12): nv += 1
    return {'N': N, 'j': j, 'xa': xa, 'xb': xb, 'ok': bool(ok), 'margin': float(wm), 'audit_points': npt, 'audit_violations': nv}


def bj1N(N, sig=SIG):
    import bj_lemma as J
    nn = np.arange(2, N + 1, dtype=float)
    J.eta = lambda p, x: float(np.sum(nn ** p * np.exp(-(nn * nn - 1) * x / 2)))
    J.Z2 = lambda x: float(np.sum(nn * nn * np.exp(-(nn * nn - 1) * x)))
    J.CELLS = np.concatenate([np.geomspace(1e-12, 0.01, 3001), np.linspace(0.01, 1.0, 3001)[1:]])
    import importlib, bl_bj1 as BL
    c = BL.CSTAR; eps = EPS
    r = BL.constants(sig, c, eps)
    th = 1 / (1 - c * sig); s2g = np.sqrt(3 * sig ** 4 + 2 * sig); ep = r['step2b']['e_prime']
    Cq = 4 * (1 + QS) * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * J.UP
    def prod(a, b):
        A, B = J.Gam1_parts(a); return min(Cq * (np.sqrt(b) * A + B) ** 2 * J.UP, J.Gam1_up(a, b) * J.m(a))
    gst = J.sup_cells(prod)
    aw2b = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * J.DN
    r['step2b'] = {'e_prime': ep, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c)}
    r['ok'] = bool(r['step2b']['contradiction'] and r['step3a']['contradiction'] and r['step3b']['contradiction'] and r['A_min'] >= 1.02 * c and r['A_min'] > 0.9)
    # E_R on [1e-12, 1], tau >= 80: N1 <= min(q sum n dec, N Q), gamma <= q Gamma1(x)
    import bl_theta as T
    tau = 80.0; Q = QS; worst = 0.0; xs = np.geomspace(1e-12, 1.0, 2001)
    for a, b in zip(xs[:-1], xs[1:]):
        nu = T.nus(tau, sig * tau / np.sqrt(2 * a)); N1 = min(0.25 * (1 + J.eta(1, a)), N * Q); gam = 0.25 * J.Gam1_up(a, b)
        cc = (np.sqrt(b) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)
        e = (0.375 * sig * tau * Q * nu[0] / a + (1 + Q) * (1 + Q / 2) * cc * nu[0] + Q * sig * tau ** 2 * nu[2] / (8 * a)
             + tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2 * a) * (1 - Q)) + sig * tau * nu[0] * (1 + Q) * gam / a
             + tau * nu[1] * (N1 * cc + (1 + Q) * gam / np.sqrt(2 * a)) + tau * Q * nu[2] * (1 + Q) * cc / 2 + Q * sig * tau ** 3 * nu[4] / (8 * a)
             + sig * tau * Q * nu[2] / (2 * (1 - nu[0])))
        worst = max(worst, e)
    r['E_R_sup'] = worst
    # conclusion: max over x in (0, 1) of x (1 + Z2(x)) (cells: x at b, Z2 at a)
    kap2 = (1 + RS + QS / (2 * (1 - QS))) ** 2 / (8 * (1 - QS))
    xs = np.concatenate([[0.0], np.geomspace(1e-12, 1.0, 4001)])
    xz = max(b * (1 + J.Z2(a)) for a, b in zip(xs[:-1], xs[1:]))
    k2 = (1 + eps) * (8 * kap2 * (1 + QS) * xz + 0.25)
    r['conclusion'] = {'sup_x(1+Z2)': xz, 'k_eff2_max': k2, 'g_lower': 0.9 * r['A_min'] / (2 * np.sqrt(k2)), 'ratio_to_0.134': 0.9 * r['A_min'] / (2 * np.sqrt(k2)) / 0.134}
    return r


if __name__ == '__main__':
    np.seterr(all='ignore')
    out = {}
    for N in (2, 3, 4, 6, 8, 12, 20, 40):
        rows = bk1N(N)
        rb = bj1N(N)
        res = {'bk1_x_cells': len(rows), 'bk1_cells': len(rows) * 120, 'bk1_all_ok': all(d['ok'] for d in rows), 'bk1_worst_margin': min(d['margin'] for d in rows),
               'bk1_audit_points': sum(d['audit_points'] for d in rows), 'bk1_audit_violations': sum(d['audit_violations'] for d in rows),
               'bj1_ok': rb['ok'], 'A_min': rb['A_min'], 'step3a_margin': rb['step3a']['margin'], 'step2i': rb['step2b']['a_lower_over_c'],
               'step3ii': rb['step3b']['W_lower_over_W_cap'], 'E_R_sup': rb['E_R_sup'], 'conclusion': rb['conclusion']}
        res['closes'] = bool(res['bk1_all_ok'] and res['bk1_audit_violations'] == 0 and rb['ok'] and rb['E_R_sup'] <= EPS and rb['conclusion']['ratio_to_0.134'] > 1)
        out['N=%d' % N] = res
        print('N = %2d: BK1 %s (margin %.3f, audit %d/%d) | BJ1 %s A_min %.4f 3(i) %.4f | E_R %.1e | sup x(1+Z2) %.3f -> g >= %.3f (x%.2f) | closes %s' % (
            N, res['bk1_all_ok'], res['bk1_worst_margin'], res['bk1_audit_violations'], res['bk1_audit_points'], rb['ok'], rb['A_min'], rb['step3a']['margin'],
            rb['E_R_sup'], rb['conclusion']['sup_x(1+Z2)'], rb['conclusion']['g_lower'], rb['conclusion']['ratio_to_0.134'], res['closes']), flush=True)
    json.dump(out, open('results/bp_windowsN.json', 'w'), indent=1, default=float)
