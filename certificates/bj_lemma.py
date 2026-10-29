"""Stage BJ §2.1-2.2: Lemma BJ1 — assumption (P8) follows from (P1)-(P7) inside the proof by contradiction (computer-assisted proof).

Lemma BJ1.  Let rho be a periodic window with (P1)-(P5), L_w <= sigma^{-1/2} and sigma <= sigma0 = 1e-24, and let 0 < c <= c* = 0.134/0.9.
If both branches are small, (a) < c and (a') < c, then
    A := (a/den) |C_w| sqrt((1 - Phi/4)(1 - delta_w)) >= A1 A2 A3 >= A_min = 0.93   (A1 A2 A3 of Lemma 5 = AY2-P).
Consequently, for every G <= 0.134 and |eps| <= eps_bar <= 0.1:  g < G  =>  (a) + (a') < G/(1 - |eps|) <= c*  =>  F_c >= (1 - |eps|) A_min >= 0.84 >= 1.02 G.
So (P8) holds for every state of the proof by contradiction, and Theorem 9 (and Lemma 7) hold under (P1)-(P7) alone.

Proof (D = den = b + W + sigma^2 J exactly; b = sigma^2/V; W = W_p; J = J_ww; b + W = sigma^2 J_vv <= 1; q = q0_1; x >= 1).
Tools (all proved in paper F): (a) >= sigma^2 J/(sigma D) (phi <= 2);  2 N_phi >= (1 - e - sigma^2) W + (1 - C)(b - sigma^2 gamma) for every pointwise
bound gamma >= |sigma^2 G| (Lemma AX3-P); here gamma_0 = gamma_P with N_p replaced by N_p^0 = sum n^p q0_n (|h_t| <= N_1^0 etc., since q_n(w) <= q0_n),
so gamma_0 <= q Gamma1(x);  0 <= 1 - C <= 2,  1 - C >= (1 - e^{-L^2/4}) - (L r + r^2/2)/2 with r <= xi_P + sigma^2 sqrt(J) (Lemma 3 = AY1-P);
J >= (1 + u0/(8(1 + Q)))/L^2 >= 1/L^2 (AX1-P);  x q^2 g_s/(2(1 + Q)) <= W <= x q^2 (1 + Z2(x)) g_s/(2(1 - Q)) (AX2-P), g_s = s/sqrt(s^2 + 2L^2),
s >= sigma, L^2 <= 1/sigma + sigma^2  =>  g_s >= g_min = sigma/sqrt(sigma^2 + 2/sigma + 2 sigma^2);  harmonics: q0_n/q0_1 = qbar_n/qbar_1 <= e^{-(n^2-1)x/2}
((P2), K >= x), q <= qbar_1 <= m(x) = min(1/4, 2 e^{-x/2}).
 Step 1 ((a) < c):  sigma^2 J < c sigma D  =>  D < theta (b + W), theta = 1/(1 - c sigma);  sigma^2 J < c sigma theta (b + W);  L^2 > sigma/(c theta (b + W)).
 Step 2 (W >= b).
   2b (b < sigma^2 gamma_0): V > 1/gamma_0 >= 1/Gamma*, so 1 - e >= 1 - exp(-(1 - sigma^2 Gamma*)^2/(4 Gamma*)); with (1 - C)(b - sigma^2 gamma_0) >= -2 sigma^2 gamma_0
       AX3 gives W < 2 sigma^2 gamma_0/e', and with the W-sandwich q < q* = 4(1 + Q*) R1 sigma^2/(e' g_min), gamma_0 <= gamma* = sup_x Gamma1(x) min(q*, m(x));
       then (a) >= sigma/(L_max^2 theta 2W) > e'/(4 theta gamma* (1 + sigma^3)) >= c: contradiction.
   2a (b >= sigma^2 gamma_0): AX3 gives 1 - e < nu = 4 c sigma theta + sigma^2, so V < Vbar = -2 ln(1 - nu)/alpha_min^2 (alpha^2 = (1 - b)^2/2, b <= W <= W_max);
       AM-GM of (1 - e) W >= (alpha^2 (1 - nu)/2)(sigma^2/b) W and (1 - C)(b - sigma^2 gamma_0) gives 1 - C < omega_C, hence L^2 < Lbar^2;
       (a) < c with u0 >= q^2 rho^{-1/2}(1 + rho)^{-3/2} and the W-sandwich gives (sigma/s)^2 < C_s x (1 + Z2) sigma.  Then Lemma 5's A1, A2, A3 >= A_min.
 Step 3 (W < b).
   3a (b >= 8 sigma^2 gamma_0): AX3 gives 1 - C < (8/7)(4 c sigma theta + sigma^2), but L^2 > sigma/(2 c theta b) >= sigma/(2 c theta) gives 1 - C >= g(L1) or g(L_max)
       (g unimodal on [L1, L_max]): contradiction (this is the tight step: it needs 32 c^2 < 7/8).
   3b (b < 8 sigma^2 gamma_0): sigma^2/L_max^2 <= sigma^2 J < 16 c theta sigma^3 gamma_0 gives gamma_0 > gamma_lo = 1/(16 c theta (1 + sigma^3)), so q > gamma_lo/Gamma1(x)
       (x admissible: Gamma1(x) m(x) > gamma_lo), and W > g_min gamma_lo^2 x/(2(1 + Q*) Gamma1(x)^2) exceeds W < b < 8 sigma^2 Gamma*: contradiction.
Every bound is monotone in sigma (worst at sigma0) and in c (worst at c*); the one-variable suprema over x >= 1 are bounded cell by cell from monotone factors
(x, sqrt x increasing; eta_p, Z2, e^{-x} decreasing; for x >= 200 every function decreases).  -> results/bj_lemma.json.  Python 3.11."""
import json, os
import numpy as np

E = np.e; QS, RS = 0.25, 0.125; KAP = (1 + QS) / (1 - QS); CSTAR = 0.134 / 0.9; UP = 1 + 1e-9; DN = 1 - 1e-9
X_END = 200.0


def eta(p, x):                       # sum_{n>=2} n^p e^{-(n^2 - 1) x/2}  (decreasing in x)
    n = np.arange(2, 60, dtype=float); return float(np.sum(n ** p * np.exp(-(n * n - 1) * x / 2)))


def Z2(x):                           # sum_{n>=2} n^2 e^{-(n^2 - 1) x}
    n = np.arange(2, 60, dtype=float); return float(np.sum(n * n * np.exp(-(n * n - 1) * x)))


def m(x): return min(QS, 2 * np.exp(-x / 2))
def Gam1_parts(x):                   # Gamma1(x) = x A(x) + sqrt(x) B(x), A, B decreasing
    A = ((1 + eta(2, x)) * (1 + QS) + QS * (1 + eta(1, x)) ** 2) / (1 - QS) ** 2; B = (1 + eta(1, x)) * (1 + 2 * QS) / (np.sqrt(E) * (1 - QS) ** 2); return A, B
def Gam1(x): A, B = Gam1_parts(x); return x * A + np.sqrt(x) * B
def Gam1_up(a, b): A, B = Gam1_parts(a); return (b * A + np.sqrt(b) * B) * UP     # upper bound of Gamma1 on [a, b]


CELLS = np.concatenate([1 + 0.001 * np.arange(0, 19001), np.geomspace(20.0, X_END, 2001)[1:]])     # cells in x: [1, 20] step 1e-3, then geometric


def cells():
    return zip(CELLS[:-1], CELLS[1:])


def sup_cells(f_up, tail=None):
    s = max(f_up(a, b) for a, b in cells())
    return max(s, tail) if tail is not None else s


def constants(sig, c=CSTAR):
    th = 1 / (1 - c * sig); Lmax2 = 1 / sig + sig * sig
    s2_over_gmin = np.sqrt(3 * sig ** 4 + 2 * sig)          # sigma^2/g_min in closed form (no underflow)
    out = {'sigma': sig, 'c': c, 'theta': th}
    # one-variable suprema (cells); tails x >= X_END bounded by the value of a decreasing majorant at X_END
    Gst = sup_cells(lambda a, b: Gam1_up(a, b) * m(a), tail=Gam1_up(X_END, X_END) * m(X_END))
    R1 = Gam1(1.0) * UP                                                                             # Gamma1(x)/x is decreasing
    Wmax = sup_cells(lambda a, b: b * min(QS * QS, 4 * np.exp(-a)) * (1 + Z2(a)) / (3 * 2 * (1 - QS)) * UP)
    Sxi = sup_cells(lambda a, b: np.sqrt(b) * m(a) * (1 + eta(1, a)) / (1 - QS) * UP)
    Qx = lambda a: min(QS, m(a) * (1 + eta(0, a)))
    T1 = sup_cells(lambda a, b: b * (1 + Z2(a)) * (Qx(a) / (1 - QS) + Qx(a) ** 2 / (E * (1 - QS) ** 2)) * UP)
    T2 = sup_cells(lambda a, b: b * np.sqrt(1 + Z2(a)) * m(a) * (1 + eta(1, a)) * UP)
    out.update({'Gamma_star': Gst, 'R1': R1, 'W_max': Wmax, 'S_xi': Sxi, 'T1': T1, 'T2': T2})
    # Step 2a
    nu = 4 * c * sig * th + sig * sig; al2 = (1 - Wmax) ** 2 / 2; Vbar = -2 * np.log1p(-nu) / al2 * UP
    assert Vbar * Gst < 0.5
    omC = (4 * c * th + sig) ** 2 * Wmax / (2 * al2 * (1 - nu) * (1 - Vbar * Gst)) * UP
    rbar = sig * (Sxi + np.sqrt(c * sig * th)); dr = 0.5 * (np.sqrt(Lmax2) * rbar + rbar * rbar / 2) * UP
    Lbar2 = -4 * np.log1p(-(omC + dr)) * UP
    Cs = 4 * (1 + QS) * c * th * (1 + RS) / (1 - QS)
    l4 = (c * sig * th) ** 2
    dH = (c * sig * th + Cs * sig * T1 + np.sqrt(Cs * sig) * T2 / (np.sqrt(E) * (1 - QS) ** 2)) * UP
    Xi = sig * sig * (Sxi ** 2 + Cs * sig * T1) * UP
    rC = (3 * Vbar / 8 + (1 + 3 * l4) * (1 + KAP / 8) * Lbar2 / 4 + 0.75 * Xi) * UP
    rw = (Vbar / 8 + (1 + 3 * l4) * 0.75 * Lbar2 + 0.75 * Xi) * UP
    A1 = 1 / (1 + KAP ** 2 * c * sig * th) * DN
    Amin = A1 * (1 - dH) * (1 - rC) * np.sqrt(1 - rw) * DN
    out['step2a'] = {'nu': nu, 'alpha2_min': al2, 'V_bar': Vbar, 'one_minus_C_bar': omC, 'r_bar': rbar, 'delta_r': dr, 'L2_bar': Lbar2, 'C_s': Cs,
                     'delta_H': dH, 'r_C': rC, 'r_w': rw, 'A1': A1, 'A_min': Amin}
    # Step 2b
    ep = -np.expm1(-(1 - sig * sig * Gst) ** 2 / (4 * Gst)) - sig * sig - 4 * c * sig * th
    qst = 4 * (1 + QS) * R1 * s2_over_gmin / ep * UP
    gst = sup_cells(lambda a, b: Gam1_up(a, b) * min(qst, m(a)), tail=Gam1_up(X_END, X_END) * m(X_END))
    aw2b = ep / (4 * th * gst * (1 + sig ** 3)) * DN
    out['step2b'] = {'e_prime': ep, 'q_star': qst, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c)}
    # Step 3a (scaled by sigma)
    L1 = np.sqrt(sig / (2 * c * th)); y = L1 * L1 / 4
    gL1 = (-np.expm1(-y) - 0.5 * (L1 * rbar + rbar * rbar / 2)) * DN
    gLmax = (-np.expm1(-Lmax2 / 4) - dr) * DN
    rhs = 8 / 7 * (4 * c * sig * th + sig * sig) * UP
    unimodal = bool(L1 * np.exp(-y) > rbar)
    out['step3a'] = {'g_L1_over_sigma': gL1 / sig, 'g_Lmax': gLmax, 'rhs_over_sigma': rhs / sig, 'margin': min(gL1, gLmax) / rhs,
                     'unimodal': unimodal, 'contradiction': bool(min(gL1, gLmax) > rhs and unimodal)}
    # Step 3b
    glo = 1 / (16 * c * th * (1 + sig ** 3)) * DN
    adm = [(a, b) for a, b in cells() if Gam1_up(a, b) * m(a) >= glo]
    assert Gam1_up(X_END, X_END) * m(X_END) < glo           # the tail x >= X_END is not admissible
    inf_xG = min(a / Gam1_up(a, b) ** 2 for a, b in adm) * DN
    ratio = glo ** 2 * inf_xG / (2 * (1 + QS) * 8 * Gst * s2_over_gmin) * DN          # W_lo/W_cap = (g_min/sigma^2) gamma_lo^2 inf/(2(1+Q*) 8 Gamma*)
    out['step3b'] = {'gamma_lo': glo, 'admissible_x': [float(adm[0][0]), float(adm[-1][1])], 'inf_x_over_Gamma1sq': inf_xG,
                     'W_lower_over_W_cap': ratio, 'contradiction': bool(ratio > 1)}
    out['cells_in_x'] = len(CELLS) - 1
    out['A_min'] = Amin; out['target_1.02c'] = 1.02 * c
    out['F_c_lower'] = 0.9 * Amin
    out['ok'] = bool(out['step2b']['contradiction'] and out['step3a']['contradiction'] and out['step3b']['contradiction'] and Amin >= 1.02 * c and Amin > 0.93)
    return out


if __name__ == '__main__':
    np.seterr(over='ignore', under='ignore')
    res = {}
    for e in (24, 48, 100, 200, 300):
        r = constants(10.0 ** (-e)); res['1e-%d' % e] = r
        print('sigma 1e-%-3d ok %s | Gamma* %.4f R1 %.3f W_max %.5f S_xi %.4f | 2a: Vbar/sigma %.3f, 1-C < %.4f, L2 < %.4f, dH %.1e, rC %.4f, rw %.4f, A_min %.6f'
              ' | 2b: (a)/c >= %.3e | 3a: margin %.4f (%.4f vs %.4f) | 3b: W_lo/W_cap %.2e, x in [%.2f, %.2f]' % (
                  e, r['ok'], r['Gamma_star'], r['R1'], r['W_max'], r['S_xi'], r['step2a']['V_bar'] / r['sigma'], r['step2a']['one_minus_C_bar'],
                  r['step2a']['L2_bar'], r['step2a']['delta_H'], r['step2a']['r_C'], r['step2a']['r_w'], r['A_min'], r['step2b']['a_lower_over_c'],
                  r['step3a']['margin'], r['step3a']['g_L1_over_sigma'], r['step3a']['rhs_over_sigma'], r['step3b']['W_lower_over_W_cap'],
                  *r['step3b']['admissible_x']), flush=True)
    res['all_ok'] = all(v['ok'] for k, v in res.items() if k.startswith('1e-'))
    json.dump(res, open('results/bj_lemma.json', 'w'), indent=1, default=float)
