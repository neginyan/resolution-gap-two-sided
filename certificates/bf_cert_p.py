"""Stage BF §1.1: Theorem AY1-P — the periodic-window lower bound in final form (stage-AY improvements applied to Theorem AX1-P):
 (1) no assumption on Lw (Lemma AY1-P),  (2) |eps| <= eps_max in the conclusion,  (3) F_c bounded from the window's parameters (Lemma AY2-P),
 every supremum certified cell by cell from monotone factors.

Lemma AY1-P (proof).  The period average of p has the exact w-marginal Gamma = N(0, L^2) (avg_t f = 1), and R - w = sigma^2 (s~ + s_w) with
 |sigma^2 s~| <= xi_P = sigma sqrt(x) N1/(1 - Qbar) pointwise and E[(sigma^2 s_w)^2] = sigma^4 J_ww < eta G theta sigma^3 inside the case g < G, so
 r := sqrt(E(R - w)^2) <= xi_P + sigma^{3/2} sqrt(eta G theta) and, with |cos a - cos b| <= |b||a - b| + (a - b)^2/2,
   1 - C >= (1 - e^{-L^2/4}) - (L r + r^2/2)/2 >= phi1 min(L^2, 1),  phi1 = 1 - e^{-1/4} - max(r/(2 L_min) + r^2/(4 L_min^2), (sigma^{-1/2} r + r^2/2)/2)
 for L_min^2 = sigma/(eta G theta) <= L^2 <= 1/sigma.  If L > 1, the AM-GM step of (P5) forces W_p > phi1 omega^2/Gam2^2, above sup W_p: impossible.
Lemma AY2-P (proof).  F_c >= (1 - |eps|) A1 A2 A3 as Lemma AY2, with the P-replacements (one envelope e^{-w^2/2s^2} for all harmonics, f in [1 - Qbar, 1 + Qbar]):
   dH = sigma^2/L^2 + (sigma/s)^2 [Qbar/(1 - Qbar) + Qbar^2/(e (1 - Qbar)^2)] + sqrt(x)(sigma/s) N1/(sqrt(e)(1 - Qbar)^2)      (|f_ww| <= Qbar/s^2, |f_w| <= Qbar e^{-1/2}/s,
                                                                                                            |f_tw| <= N1 e^{-1/2}/s, |f_t| <= N1)
   r_C = V/2 + (1 + 3 l4)(L^2 + s^2)/4 + (3/4) Xi_P,   r_w = V/8 + (1 + 3 l4)(3 L^2/4 + (3/8) kQ s^2) + (3/4) Xi_P,   kQ = (1 + Qbar)/(1 - Qbar),
   Xi_P = xi_P^2 + sigma^2 (sigma/s)^2 Qbar^2/(e (1 - Qbar)^2);  the exact identities <s_w^2 w^2> = 3 + int Gamma w^2 avg(f_w^2/f) and
   J_ww = 1/L^2 + int Gamma avg(f_w^2/f) (avg_t f_w = 0 kills the cross terms) and the common envelope give <window part of s_w^2 w^2>/J_ww,win <= (3 s^2/2) kQ,
   <s~^2 w^2>/J_vv,per <= (s^2/2) kQ  (kQ = 5/3 <= 2 keeps Lemma AY2's coefficient of s^2 in r_C).
Bootstrap for F_c (rigorous form of stage AY's iteration).  The chain needs a lower bound F <= F_c (through U = (F^2/(4G^2) - 1/4)/kappa2 ...).
 Step k: from F_c >= F_k the chain gives x >= x_low(F_k) = max(1, U0/S_i(1), U0 - S1_P), L^2 <= L2max, s^2 >= sigma/(Gam3_P x), x <= X; Lemma AY2-P then gives
 F_c >= F_{k+1} = (1 - |eps|) F_int(F_k).  Every step is an implication, so the iteration may only go UP from a starting value F_0 that is assumed
 ((S7) F_c >= F_0).  The certificate closes from F_0 = 1.02 G (= 1.22/sqrt(ln(1/sigma)) below 1e-28): the first step already gives F_c >= 0.95,
 the next ones 0.99..., so (S7) only excludes states whose AT2 numerator is below the target gap itself.
 (Stage AY iterated downward from F = 1, i.e. it used the fixed point; the upward bootstrap is the rigorous reading; both end at the same F_int.)
Tail of the suprema: x gamma_P(x), gamma_P(x), sqrt(x) N1(x) and x min(1/16, 4 e^{-x}) decrease for x >= 8 (x^2 e^{-x/2} factors), so cells over
 [x_lo, max(x_max, 60)] contain their suprema; for ln(1/sigma) > 690 the sigma-small factors are evaluated at sigma = 1e-300 (all increase with sigma).
-> results/bf_cert_p.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import brentq
from be_cert_p import r_m, eta, zeta, q1, kap2, Qb, gamma_P, cell_sup, S1_P, MMAX, QB, RHO
from be_plemmas import S_branch_i
import functools, be_cert_p as _B
_B.eta = functools.lru_cache(maxsize=None)(_B.eta); _B.zeta = functools.lru_cache(maxsize=None)(_B.zeta)     # same values, cached
eta, zeta = _B.eta, _B.zeta

SQ2 = np.sqrt(2.0); E = np.e
F0_OVER_G = 1.02                       # (S7): F_c >= 1.02 G, the a-priori start of the bootstrap
def N1(x): return float(q1(x)) * (1 + eta(1, x))
def Si_at(x):
    q = float(q1(x)); return S_branch_i(np.array([q * r_m(x, m) for m in range(1, MMAX)]))[0]


def chain(sig, lnsig, G, eta_, F, x_lo, S1):
    """Theorem AX1-P's chain with Lemma AY1-P (no Lw assumption) and lower bound F of F_c; returns all certified constants and Lemma AY2-P's F_int."""
    th = 1 / (1 - eta_ * G * sig); U0 = F * F / (4 * G * G) - 0.25
    if U0 <= 0: return {'ok': False, 'why': 'U0 <= 0'}
    z0, z2 = zeta(0, x_lo), zeta(2, x_lo); kp = kap2(); UP = U0 / (kp * (1 + z2)); Ueff = UP / (1 + QB)
    # a-priori x range with L <= sigma^{-1/2} (no Lw assumption): x e^{-x} > U_P sigma^{3/2}/(4 sqrt2 (1 + zeta0))
    lrhs = np.log(UP / (SQ2 * 4 * (1 + z0))) - 1.5 * lnsig
    if lrhs > -1: return {'ok': False, 'why': 'x range'}
    x_max = brentq(lambda x: np.log(x) - x - lrhs, 1.0, 1e9); xr = max(x_max, 60.0); xe = max(min(xr, 400.0), x_lo + 20.0)   # right end of the cells (decreasing tails)
    sup_xg = cell_sup(lambda a, b: b * gamma_P(a, b), x_lo, xe)
    xi = sig * cell_sup(lambda a, b: np.sqrt(b) * N1(a) / (1 - Qb(a)), x_lo, xe)
    L2min = sig / (eta_ * G * th)
    r = xi + sig ** 1.5 * np.sqrt(eta_ * G * th)
    dR = max(r / (2 * np.sqrt(L2min)) + r * r / (4 * L2min), (r / np.sqrt(sig) + r * r / 2) / 2)
    phi1 = 1 - np.exp(-0.25) - dR                                                   # Lemma AY1-P: 1 - C >= phi1 min(L^2, 1)
    Phi_min = phi1 * L2min; Phi_max = 2.0
    tau0 = 2 * eta_ * G * sig * th; tau = tau0 + sig * sig
    condA = Phi_min >= 2 * (2 * tau0 + sig * sig)
    sig_caseB = Ueff / (32 * eta_ * G * th * (1 + sig * sig) * sup_xg)
    r_wp = sig * 16 * eta_ * G * th * (1 + sig * sig) * sup_xg / Ueff
    beta = r_wp * Phi_min / (Phi_min - tau0) + tau / (Phi_min - tau0)
    Wp_max = cell_sup(lambda a, b: b * min(QB ** 2, 4 * np.exp(-a)), x_lo, max(60.0, x_lo + 20.0)) * (1 + z2) * (1 / 3) / (2 * (1 - QB))
    b_max = beta * Wp_max; nu = sig * sig + tau0 + Phi_max * r_wp
    a2 = (1 - b_max) ** 2 / 2; V_max = 2 * (-np.log1p(-nu)) / a2
    om_term = cell_sup(lambda a, b: sig * gamma_P(a, b) * np.sqrt(Phi_max) * np.sqrt(16 * eta_ * G * th * b * (1 + sig * sig) / (sig * Ueff)), x_lo, xe)
    omega = (1 - b_max) * np.exp(-V_max / 8) - om_term
    Gam2 = 2 * eta_ * G * th * (1 + beta) + sig
    L_gt_1_excluded = bool(phi1 * omega ** 2 / Gam2 ** 2 > Wp_max)
    Gam3P = 2 * eta_ * G * th * (1 + beta) * (1 + QB) * (1 + z2) * (1 + RHO) ** 1.5 / (1 - QB)
    K = np.log(256 * Gam2 ** 2 * Gam3P * (1 + z2) * (1 + z0) ** 3 / ((1 - QB) * UP ** 3 * phi1 * omega ** 2))
    X = brentq(lambda X: 4 * X - lnsig - 5 * np.log(X) - K, 1.0, 1e9) + 1e-9
    # Lemma AY2-P inside the case g < G (x in [x_lo, X], L^2 <= L2max, s^2 in [sigma/(Gam3_P x), L^2/4])
    Qx = Qb(x_lo); kQ = (1 + Qx) / (1 - Qx)
    xq = cell_sup(lambda a, b: b * min(QB ** 2, 4 * np.exp(-a)), x_lo, max(X, x_lo + 1.0), 2001)
    L2max = min(1.0, 0.5 * Gam2 ** 2 * xq * (1 + z2) / (2 * SQ2 * (1 - QB) * phi1 * omega ** 2))
    s2max = L2max / 4; sos_max = np.sqrt(sig * Gam3P * X)                          # (sigma/s)_max
    xiX = sig * cell_sup(lambda a, b: np.sqrt(b) * N1(a) / (1 - Qb(a)), x_lo, max(X, x_lo + 1.0), 2001)
    sxN1 = cell_sup(lambda a, b: np.sqrt(b) * N1(a), x_lo, max(X, x_lo + 1.0), 2001)
    l4 = (eta_ * G * th * sig) ** 2
    Xi = xiX ** 2 + sig ** 2 * sos_max ** 2 * Qx ** 2 / (E * (1 - Qx) ** 2)
    dH = sig * eta_ * G * th + sos_max ** 2 * (Qx / (1 - Qx) + Qx * Qx / (E * (1 - Qx) ** 2)) + sxN1 * sos_max / (np.sqrt(E) * (1 - Qx) ** 2)
    rC = 0.5 * V_max + (1 + 3 * l4) * (L2max + s2max) / 4 + 0.75 * Xi
    rw = V_max / 8 + (1 + 3 * l4) * (0.75 * L2max + 0.375 * kQ * s2max) + 0.75 * Xi
    F_int = (1 - eta_ * G * sig * th) * (1 - dH) * (1 - rC) * np.sqrt(max(0.0, 1 - rw))
    ok = bool(condA and sig <= sig_caseB and omega > 0 and 0 < beta < 1 and nu < 1 and phi1 > 0 and L_gt_1_excluded and F_int > 0)
    return {'ok': ok, 'U0': U0, 'U_P': UP, 'x_lo': x_lo, 'x_max': x_max, 'sup_xgamma': sup_xg, 'xi_over_sigma': xi / sig, 'r_over_sigma': r / sig, 'phi1': phi1,
            'beta': beta, 'nu': nu, 'V_max': V_max, 'omega': omega, 'Gam2': Gam2, 'Gam3P': Gam3P, 'Wp_max': Wp_max, 'W_p_forced_if_L_gt_1': phi1 * omega ** 2 / Gam2 ** 2,
            'L_gt_1_excluded': L_gt_1_excluded, 'cond_PhiC': bool(condA), 'sigma_caseB_limit': sig_caseB, 'K': K, 'X': X, 'X_minus_quarter_ln': X - lnsig / 4,
            'L2max': L2max, 'sigma_over_s_max': sos_max, 'dH': dH, 'rC': rC, 'rw': rw, 'F_int': F_int}


def certificate(lnsig, eps=0.01, F0=None, cG=1.2, S1=None, Si1=None):
    sig = np.exp(-lnsig) if lnsig < 690 else 1e-300
    eta_ = 1 / (1 - abs(eps)); G = min(0.149 / eta_, cG / np.sqrt(lnsig)); S1 = S1_P() if S1 is None else S1; Si1 = Si_at(1.0) if Si1 is None else Si1
    F = 2.0 * G if F0 is None else F0; steps = []
    x_lo = 1.0
    for k in range(40):
        c = chain(sig, lnsig, G, eta_, F, x_lo, S1)
        if not c['ok']: return {'ln1_sigma': lnsig, 'ok': False, 'F0': F0, 'steps': steps, 'fail': c}
        Fn = (1 - abs(eps)) * c['F_int']; steps.append({'F': F, 'x_lo': x_lo, 'F_next': Fn})
        U0n = Fn * Fn / (4 * G * G) - 0.25; x_new = max(1.0, U0n / Si1, U0n - S1)
        if Fn <= F * (1 + 1e-13) and x_new <= x_lo * (1 + 1e-13): break
        F = max(F, Fn); x_lo = max(x_lo, x_new)                                    # each step is an implication: only go up
    g = min(G, F / (2 * np.sqrt(c['X'] + S1 + 0.25)))
    c.update({'ln1_sigma': lnsig, 'eps': eps, 'G': G, 'F0': steps[0]['F'], 'F_final': F, 'F_int_final': F / (1 - abs(eps)), 'n_steps': len(steps), 'S1_P': S1,
              'S_i(1)': Si1, 'g_bound': g, 'g_bound_sqrt_ln': g * np.sqrt(lnsig), 'g_sqrt_ln_over_1_minus_eps': g * np.sqrt(lnsig) / (1 - abs(eps)), 'steps': steps})
    return c


def smallest_F0(lnsig, eps, S1, Si1):
    G = min(0.149 * (1 - abs(eps)), 1.2 / np.sqrt(lnsig)); best = None
    for f in (1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        r = certificate(lnsig, eps, F0=f * G, S1=S1, Si1=Si1)
        if r['ok']: return float(f), r
    return None, None


if __name__ == '__main__':
    S1 = S1_P(); Si1 = Si_at(1.0); print('S1_P = %.4f, S_i(1) = %.4f' % (S1, Si1), flush=True)
    res = {'S1_P': S1, 'S_i_at_1': Si1}
    for eps in (0.0, 0.01, 0.1):
        for e in (24, 32, 48, 64, 96, 128, 256, 512, 1024, 4096, 16384):
            L = e * np.log(10); f = F0_OVER_G; G = min(0.149 * (1 - abs(eps)), 1.2 / np.sqrt(L)); r = certificate(L, eps, F0=f * G, S1=S1, Si1=Si1)
            if not r['ok']: res['eps%g_1e-%d' % (eps, e)] = {'ok': False}; print('eps %g 1e-%d: no F0 closes' % (eps, e)); continue
            r['F0_over_G'] = f; r['F0_times_sqrt_ln'] = f * r['G'] * np.sqrt(L); res['eps%g_1e-%d' % (eps, e)] = r
            print('eps %-4g sigma 1e-%-6d ok %s | F0 = %.2f G (= %.3f/sqrt ln) -> F_int %.8f in %d steps (x_lo %.2f, L2max %.1e) | phi1 %.4f omega %.4f beta %.3f K %.2f X-ln/4 %+.2f'
                  ' | W_p forced %.2f > %.3f | g >= %.5f  x sqrt(ln) %.4f  /(1-eps) %.4f' % (
                      eps, e, r['ok'], f, f * r['G'] * np.sqrt(L), r['F_int_final'], r['n_steps'], r['x_lo'], r['L2max'], r['phi1'], r['omega'], r['beta'], r['K'],
                      r['X_minus_quarter_ln'], r['W_p_forced_if_L_gt_1'], r['Wp_max'], r['g_bound'], r['g_bound_sqrt_ln'], r['g_sqrt_ln_over_1_minus_eps']), flush=True)
    json.dump(res, open('results/bf_cert_p.json', 'w'), indent=1, default=float)
