"""Stage BL §3: Lemma BJ1 ((P8) follows from (P1)-(P7)) redone on the theta_tau-form lemmas BL1-BL6 and Lemma BK1 (tau0 = 80); computer-assisted proof.
Also the polishing of stage BJ §1: sigma- and c-monotonicity table with a grid check, step 3(a) in interval arithmetic, the reasons for x >= 200.

Lemma BJ1' (theta form).  Periodic window with (P1)-(P5), L_w <= sigma^{-1/2}, sigma <= 1e-24, 0 < c <= c* = 0.134/0.9.  If (a) < c and (a') < c, then
tau > 80 (Lemma BK1) and A1 A2 A3 >= A_min (Lemma 5 in theta form, BL6), with every input lemma perturbed by eps = eps_in = 1e-6 (> eps_theta, bl_theta.py).
Changes against bj_lemma.py (v1.0 of the lemma):
  AX3 in theta form (BL3, form A): 2 N_phi >= (1 - e - (1 + eps) sigma^2) W + (1 - eps) b Psi - (1 + eps) sigma^2 gamma_0 Phi - sigma b eps,
     Psi >= Phi_b = min(Phi, 1/2), Phi <= 4 Phi_b.  The case splits move: 2a/2b at b = 4 kappa sigma^2 gamma_0, 3a/3b at b = 32 kappa sigma^2 gamma_0, kappa = (1 + eps)/(1 - eps)
     (so that (1 - eps) b Psi - (1 + eps) sigma^2 gamma_0 Phi >= 0, resp. >= (7/8)(1 - eps) b Phi_b, for every Phi in [0, 2]).
  J_ww >= (1 - eps) J_ww,per (BL1);  (1 - 3 eps) W_per,lo - eps b <= W <= (1 + 3 eps) W_per,up + eps b (BL2);  r of Lemma 3 with c/(1 - eps) (BL4);
  r_C, r_w with (1 + eps) (BL6);  A1 = a/den > 1 - c sigma exactly (sigma^2 J_ww/den < c sigma from (a) < c).
-> results/bl_bj1.json.  Python 3.11."""
import json
import numpy as np
from decimal import Decimal as Dc
from ba_ivl import I as IV
import bj_lemma as J

E = np.e; QS, RS, KAP, CSTAR, UP, DN, X_END = J.QS, J.RS, J.KAP, J.CSTAR, J.UP, J.DN, J.X_END
EPS = 1e-6


def constants(sig, c=CSTAR, eps=EPS):
    th = 1 / (1 - c * sig); Lmax2 = 1 / sig + sig * sig; kap = (1 + eps) / (1 - eps)
    s2g = np.sqrt(3 * sig ** 4 + 2 * sig)
    Gst = J.sup_cells(lambda a, b: J.Gam1_up(a, b) * J.m(a), tail=J.Gam1_up(X_END, X_END) * J.m(X_END))
    R1 = J.Gam1(1.0) * UP
    Wmax = (1 + 3 * eps) * J.sup_cells(lambda a, b: b * min(QS * QS, 4 * np.exp(-a)) * (1 + J.Z2(a)) / (3 * 2 * (1 - QS)) * UP)
    Sxi = J.sup_cells(lambda a, b: np.sqrt(b) * J.m(a) * (1 + J.eta(1, a)) / (1 - QS) * UP)
    Qx = lambda a: min(QS, J.m(a) * (1 + J.eta(0, a)))
    T1 = J.sup_cells(lambda a, b: b * (1 + J.Z2(a)) * (Qx(a) / (1 - QS) + Qx(a) ** 2 / (E * (1 - QS) ** 2)) * UP)
    T2 = J.sup_cells(lambda a, b: b * np.sqrt(1 + J.Z2(a)) * J.m(a) * (1 + J.eta(1, a)) * UP)
    out = {'sigma': sig, 'c': c, 'eps': eps, 'Gamma_star': Gst, 'R1': R1, 'W_max': Wmax, 'S_xi': Sxi, 'T1': T1, 'T2': T2}
    ce = c / (1 - eps)                                                                   # c with the J_ww factor (Lemma 3, L^2 bounds)
    # ---- step 2a: W >= b, b >= 4 kap sigma^2 gamma_0
    nu = 4 * c * sig * th + (1 + eps) * sig * sig + sig * eps
    al2 = (1 - Wmax) ** 2 / 2; Vbar = -2 * np.log1p(-nu) / al2 * UP
    ka = (1 - eps) - 4 * (1 + eps) * Vbar * Gst
    omC = (4 * c * th + (1 + eps) * sig + eps) ** 2 * Wmax / (2 * al2 * (1 - nu) * ka) * UP
    assert omC < 0.5                                                                     # so Phi_b < 1/2, i.e. Phi_b = Phi
    rbar = sig * (Sxi + np.sqrt(ce * sig * th)); dr = 0.5 * (np.sqrt(Lmax2) * rbar + rbar * rbar / 2) * UP
    Lbar2 = -4 * np.log1p(-(omC + dr)) * UP
    Cs = 4 * (1 + QS) * ce * th * (1 + RS) / (1 - QS) * (1 + 3 * eps)
    l4 = (ce * sig * th) ** 2
    dH = (c * sig * th + Cs * sig * T1 + np.sqrt(Cs * sig) * T2 / (np.sqrt(E) * (1 - QS) ** 2)) * UP
    Xi = sig * sig * (Sxi ** 2 + Cs * sig * T1) * UP
    rC = (1 + eps) * (3 * Vbar / 8 + (1 + 3 * l4) * (1 + KAP / 8) * Lbar2 / 4 + 0.75 * Xi) * UP
    rw = (1 + eps) * (Vbar / 8 + (1 + 3 * l4) * 0.75 * Lbar2 + 0.75 * Xi) * UP
    A1 = (1 - c * sig) * DN
    Amin = A1 * (1 - dH) * (1 - rC) * np.sqrt(1 - rw) * DN
    out['step2a'] = {'nu': nu, 'V_bar': Vbar, 'kappa_a': ka, 'one_minus_C_bar': omC, 'r_bar': rbar, 'L2_bar': Lbar2, 'C_s': Cs, 'delta_H': dH, 'r_C': rC, 'r_w': rw,
                     'A1': A1, 'A_min': Amin}
    # ---- step 2b: W >= b, b < 4 kap sigma^2 gamma_0  (V > 1/(4 kap Gamma*))
    a2b = (1 - 4 * kap * sig * sig * Gst) ** 2 / 2
    ep = -np.expm1(-a2b / (2 * 4 * kap * Gst)) - (1 + eps) * sig * sig - sig * eps - 4 * c * sig * th
    qst = 4 * (1 + QS) * R1 * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * UP
    gst = J.sup_cells(lambda a, b: J.Gam1_up(a, b) * min(qst, J.m(a)), tail=J.Gam1_up(X_END, X_END) * J.m(X_END))
    aw2b = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * DN
    out['step2b'] = {'e_prime': ep, 'q_star': qst, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c)}
    # ---- step 3a: W < b, b >= 32 kap sigma^2 gamma_0
    L1 = np.sqrt(sig * (1 - eps) / (2 * c * th)); y = L1 * L1 / 4
    gL1 = (-np.expm1(-y) - 0.5 * (L1 * rbar + rbar * rbar / 2)) * DN
    gLmax = (-np.expm1(-Lmax2 / 4) - dr) * DN
    rhs = 8 / 7 * (4 * c * sig * th + (1 + eps) * sig * sig + sig * eps) / (1 - eps) * UP
    unimodal = bool(L1 * np.exp(-y) > rbar)
    lowb = min(gL1, gLmax, 0.5)
    out['step3a'] = {'g_L1_over_sigma': gL1 / sig, 'g_Lmax': gLmax, 'rhs_over_sigma': rhs / sig, 'margin': lowb / rhs, 'unimodal': unimodal,
                     'contradiction': bool(lowb > rhs and unimodal)}
    # ---- step 3b: W < b, b < 32 kap sigma^2 gamma_0
    glo = (1 - eps) ** 2 / (64 * (1 + eps) * c * th * (1 + sig ** 3)) * DN
    adm = [(a, b) for a, b in J.cells() if J.Gam1_up(a, b) * J.m(a) >= glo]
    assert J.Gam1_up(X_END, X_END) * J.m(X_END) < glo
    inf_xG = min(a / J.Gam1_up(a, b) ** 2 for a, b in adm) * DN
    cap = 32 * kap * Gst                                                                 # W < b < cap sigma^2
    ratio = ((1 - 3 * eps) * glo ** 2 * inf_xG / (2 * (1 + QS)) / s2g) / cap - eps * DN       # W/(cap sigma^2) >= (1 - 3eps) W_lo/(cap sigma^2) - eps (b-part)
    out['step3b'] = {'gamma_lo': glo, 'admissible_x': [float(adm[0][0]), float(adm[-1][1])], 'inf_x_over_Gamma1sq': inf_xG, 'W_lower_over_W_cap': ratio,
                     'contradiction': bool(ratio > 1)}
    out['A_min'] = Amin; out['F_c_lower'] = 0.9 * Amin
    out['ok'] = bool(out['step2b']['contradiction'] and out['step3a']['contradiction'] and out['step3b']['contradiction'] and Amin >= 1.02 * c and Amin > 0.93)
    return out


def step3a_interval(sig_exp=24, c=CSTAR, eps=EPS):
    """step 3(a) at sigma = 10^-sig_exp in outward-rounded 60-digit interval arithmetic (ba_ivl.I, the class of the upper-side certificate):
    g(L1) > rhs and the unimodality test L1 e^{-L1^2/4} > r_bar."""
    s = IV(Dc(10) ** (-sig_exp)); cI = IV(Dc(float(c))); e = IV(Dc(float(eps))); one = IV(Dc(1))
    th = one / (one - cI * s)
    # S_xi: the float certificate's value enlarged by 1e-9 (an upper bound of a supremum certified from monotone factors)
    Sxi = IV(Dc(float(constants(10.0 ** (-sig_exp), c, eps)['S_xi'])) * Dc('1.000000001'))
    ce = cI / (one - e)
    rbar = s * (Sxi + (ce * s * th).sqrt())
    L1 = (s * (one - e) / (IV(Dc(2)) * cI * th)).sqrt(); y = L1 * L1 / IV(Dc(4))
    gL1 = -((-y).em1()) - (L1 * rbar + rbar * rbar / IV(Dc(2))) / IV(Dc(2))
    rhs = IV(Dc(8)) / IV(Dc(7)) * (IV(Dc(4)) * cI * s * th + (one + e) * s * s + s * e) / (one - e)
    uni = L1 * (-y).exp() - rbar
    g_s, r_s, mg = gL1 / s, rhs / s, gL1 / rhs
    return {'g_L1_over_sigma': [float(g_s.lo), float(g_s.hi)], 'rhs_over_sigma': [float(r_s.lo), float(r_s.hi)],
            'margin_lower': float(mg.lo), 'g_gt_rhs': bool((gL1 - rhs).lo > 0), 'unimodal': bool(uni.lo > 0)}


# declared directions: d(quantity)/d(sigma) and d/d(c) (+1 increases = worse for the 'bad-up' rows, -1 decreases), with one-line reasons
MONO = [
    ('theta = 1/(1 - c sigma)', 'theta', +1, +1, 'denominator decreases in c and sigma'),
    ('nu = 4 c sigma theta + (1+eps) sigma^2 + eps sigma (step 2a)', 'nu', +1, +1, 'sum of products of non-decreasing factors'),
    ('V_bar = -2 ln(1 - nu)/alpha_min^2', 'V_bar', +1, +1, '-ln(1 - nu) increases with nu'),
    ('1 - C bound omega_C', 'one_minus_C_bar', +1, +1, '(4c theta + sigma + eps)^2 up, kappa_a and 1 - nu down'),
    ('r_bar = sigma (S_xi + sqrt(c sigma theta/(1 - eps)))', 'r_bar', +1, +1, 'product of non-decreasing factors'),
    ('L2_bar = -4 ln(1 - omega_C - delta_r)', 'L2_bar', +1, +1, 'omega_C and delta_r = (L_max r_bar + r_bar^2/2)/2 increase (L_max r_bar ~ sqrt sigma)'),
    ('C_s (sigma/s bound)', 'C_s', 0, +1, 'proportional to c theta'),
    ('delta_H', 'delta_H', +1, +1, 'terms c sigma theta, C_s sigma, sqrt(C_s sigma)'),
    ('r_C, r_w', 'r_C', +1, +1, 'V_bar, L2_bar, Xi up'),
    ('A_min (lower bound)', 'A_min', -1, -1, '1 - dH, 1 - r_C, 1 - r_w and A1 = 1 - c sigma all decrease'),
    ("e' (step 2b)", 'e_prime', -1, -1, 'minus sigma^2, sigma eps, 4 c sigma theta; V-floor 1/(4 kap Gamma*) independent of c'),
    ('q* (step 2b)', 'q_star', +1, +1, "s2/g_min = sqrt(3 sigma^4 + 2 sigma) increases, e' decreases"),
    ("(a)/c lower bound (step 2b)", 'a_lower_over_c', -1, -1, "e' down, gamma* up (through q*), and the division by c"),
    ('g(L1)/rhs (step 3a)', 'margin3a', -1, -1, 'L1^2 = sigma(1-eps)/(2 c theta): g(L1)/sigma ~ 1/(8 c theta) - r-terms; rhs/sigma ~ (32/7) c theta'),
    ('gamma_lo (step 3b)', 'gamma_lo', 0, -1, '1/(64 kap c theta (1 + sigma^3))'),
    ('W_lo/W_cap (step 3b)', 'W_lower_over_W_cap', -1, -1, 'gamma_lo^2 down, s2/g_min up; inf x/Gamma1^2 over a larger admissible set'),
]


def flat(r):
    return {'theta': 1 / (1 - r['c'] * r['sigma']), 'nu': r['step2a']['nu'], 'V_bar': r['step2a']['V_bar'], 'one_minus_C_bar': r['step2a']['one_minus_C_bar'],
            'r_bar': r['step2a']['r_bar'], 'L2_bar': r['step2a']['L2_bar'], 'C_s': r['step2a']['C_s'], 'delta_H': r['step2a']['delta_H'], 'r_C': r['step2a']['r_C'],
            'A_min': r['A_min'], 'e_prime': r['step2b']['e_prime'], 'q_star': r['step2b']['q_star'], 'a_lower_over_c': r['step2b']['a_lower_over_c'],
            'margin3a': r['step3a']['margin'], 'gamma_lo': r['step3b']['gamma_lo'], 'W_lower_over_W_cap': r['step3b']['W_lower_over_W_cap']}


def mono_grid():
    """grid check of the declared directions (numerical, not part of the proof): sigma in 1e-24..1e-60 at c = c*, and c in [0.05, c*] at sigma = 1e-24."""
    sg = [1e-24, 1e-30, 1e-40, 1e-60]; cg = [0.05, 0.1, 0.13, CSTAR]
    S = [flat(constants(s)) for s in sg]; Cc = [flat(constants(1e-24, c)) for c in cg]
    rows = []
    for name, key, ds, dc, why in MONO:
        vs = [d[key] for d in S]; vc = [d[key] for d in Cc]
        # sigma decreases along sg: a quantity increasing in sigma must be non-increasing along the list
        ok_s = True if ds == 0 else all((vs[i + 1] - vs[i]) * ds <= abs(vs[i]) * 1e-9 for i in range(len(vs) - 1))
        ok_c = True if dc == 0 else all((vc[i + 1] - vc[i]) * dc >= -abs(vc[i]) * 1e-9 for i in range(len(vc) - 1))
        rows.append({'quantity': name, 'sigma_dir': ds, 'c_dir': dc, 'reason': why, 'ok_sigma': bool(ok_s), 'ok_c': bool(ok_c), 'along_sigma': vs, 'along_c': vc})
    return rows


X200 = [
    ('Gamma1(x)/x = A(x) + B(x)/sqrt x', 'A, B are positive combinations of eta_1(x), eta_2(x), which decrease (each term e^{-(n^2-1)x/2})'),
    ('eta_p(x) = sum_{n>=2} n^p e^{-(n^2-1)x/2}', 'every term decreases in x'),
    ('Z2(x) = sum_{n>=2} n^2 e^{-(n^2-1)x}', 'every term decreases in x'),
    ('m(x) = min(1/4, 2 e^{-x/2})', 'equals 2 e^{-x/2} for x >= 2 ln 64, decreasing'),
    ('x e^{-x}, x min(1/16, 4 e^{-x})(1 + Z2)', 'd/dx ln(x e^{-x}) = 1/x - 1 < 0 for x > 1; (1 + Z2) decreases'),
    ('Gamma1(x) m(x) = 2 e^{-x/2}(x A + sqrt x B)', 'd/dx ln(x e^{-x/2}) = 1/x - 1/2 < 0 for x > 2, A, B decrease'),
    ('sqrt x m(x)(1 + eta_1)/(1 - Q)', 'd/dx ln(sqrt x e^{-x/2}) = 1/(2x) - 1/2 < 0 for x > 1'),
    ('x (1 + Z2) Q(x)/(1 - Q), x sqrt(1 + Z2) m(x)(1 + eta_1)', 'Q(x) <= m(x)(1 + eta_0) ~ e^{-x/2}: x e^{-x/2} decreases for x > 2'),
    ('x / Gamma1(x)^2 (inf in step 3b)', 'not needed beyond 200: the admissible set {Gamma1 m >= gamma_lo} ends below x = 16'),
]


if __name__ == '__main__':
    np.seterr(over='ignore', under='ignore')
    res = {'eps_in': EPS}
    for e in (24, 48, 100, 200, 300):
        r = constants(10.0 ** (-e)); res['1e-%d' % e] = r
        print('sigma 1e-%-3d ok %s | 2a: A_min %.6f (1-C < %.4f, L2 < %.4f) | 2b: (a)/c >= %.3e | 3a: margin %.4f | 3b: W_lo/W_cap %.2e, x in [%.2f, %.2f]' % (
            e, r['ok'], r['A_min'], r['step2a']['one_minus_C_bar'], r['step2a']['L2_bar'], r['step2b']['a_lower_over_c'], r['step3a']['margin'],
            r['step3b']['W_lower_over_W_cap'], *r['step3b']['admissible_x']), flush=True)
    res['all_ok'] = all(v['ok'] for k, v in res.items() if k.startswith('1e-'))
    res['step3a_interval'] = step3a_interval(); print('step 3(a) in interval arithmetic:', res['step3a_interval'], flush=True)
    res['monotonicity'] = mono_grid(); print('monotonicity table: %d rows, all directions confirmed on the grid: %s' % (
        len(res['monotonicity']), all(r['ok_sigma'] and r['ok_c'] for r in res['monotonicity'])), flush=True)
    res['x_ge_200'] = [{'function': f, 'reason': w} for f, w in X200]
    json.dump(res, open('results/bl_bj1.json', 'w'), indent=1, default=float)
