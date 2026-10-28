"""Stage AY: Theorem AY1 — the strip family's lower bound with F_c replaced by an explicit 1 - o(1), no assumption on Lw, the remainder eps
kept in the conclusion, and every supremum certified by interval bounds on grid cells (monotone factors).

Assumptions: strip (paper E) with qbar <= 1/4, rho = s^2/(2L^2) <= 1/8; paper E's remainder |eps| <= 0.1 (a state quantity; enters the conclusion).
Theorem AY1:  g >= min( G, (1 - |eps|) F_int / (2 sqrt(X + S1 + 1/4)) ),  4X = ln(1/sigma) + 5 ln X + K(eps),  G = min(0.149(1 - |eps|), 1.2/sqrt(ln(1/sigma))),
  F_int = (1 - eta G sigma)(1 - dH)(1 - rC) sqrt(1 - rw) -> 1 (explicit; Lemma AY2), so g sqrt(ln(1/sigma)) >= (1 - |eps|)(1 - O(ln ln / ln)).
Changes against Theorem AX1 (stage AX):
 (1) Lw: Lemma AY1 gives 1 - C >= (1 - e^{-L^2/4}) - dR >= phi1 min(L^2, 1), phi1 = 1 - e^{-1/4} - O(sqrt sigma) (for L <= sigma^{-1/2}); L > 1 is then impossible when g < G
     (it would force W_p > phi1 omega^2/Gam2^2 > 0.115 >= sup W_p) — no assumption on Lw is needed.
 (2) eps: eta = 1/(1 - |eps|) enters only through K(eps) (Gam2^2 Gam3 ~ eta^3) and the conclusion's factor (1 - |eps|).
 (3) F_c: Lemma AY2 bounds a/den, |C_w|, sqrt((1 - Phi/4)(1 - delta_w)) from below inside the case g < G.
 (4) suprema over x and L are bounded cell by cell from the monotone factors (q(x) = min(1/4, e^{-x/2}) decreasing, x increasing,
     K1 increasing in q, kappa2 increasing in q), so no maximum can escape between grid points.
-> results/ay_certificate.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import brentq
from aw_lemmas import K1

SQ2 = np.sqrt(2.0); QB, RHO = 0.25, 0.125; E = np.e


def kap2(q=QB, rho=RHO): return (1 + rho + q / (2 * (1 - q))) ** 2 / (8 * (1 - q))


def qx(x): return np.minimum(QB, np.exp(-np.asarray(x, float) / 2))


def cell_sup(f_upper, x0, x1, n=20001):
    """sup over [x0, x1] of a function bounded on each cell [a, b] by f_upper(a, b) (monotone-factor bound)."""
    xs = np.linspace(x0, x1, n); return float(np.max(f_upper(xs[:-1], xs[1:])))


def xgamma_up(a, b):        # x gamma(x) = x q (x(1+q) + sqrt x)/(1-q)^2: x-factors increasing (use b), q-factors increasing in q (use q(a))
    q = qx(a); return b * q * (b * (1 + q) + np.sqrt(b)) / (1 - q) ** 2


def gamma_up(a, b):
    q = qx(a); return q * (b * (1 + q) + np.sqrt(b)) / (1 - q) ** 2


def S1_up(a, b, rho=RHO):
    q = qx(a); return b * np.array([K1(t, rho, 60) for t in q]) * q * q


def certificate(lnsig, eps=0.01, G0=None, cG=1.2, F=None, iterate=True):
    """lnsig = ln(1/sigma).  Returns the certified constants and the bound.  sigma-small factors use sigma = exp(-lnsig) (underflow-safe)."""
    sig = np.exp(-lnsig) if lnsig < 700 else 0.0
    eta = 1 / (1 - abs(eps)); G0 = 0.149 / eta if G0 is None else G0; G = min(G0, cG / np.sqrt(lnsig)); th = 1 / (1 - eta * G * sig); kp = kap2()
    F = 1.0 if F is None else F
    out = {'ln1_sigma': lnsig, 'G': G, 'eta': eta, 'eps': eps}
    for it in range(4 if iterate else 1):
        U0 = F * F / (4 * G * G) - 0.25; U = U0 / kp
        # a-priori x range from u <= sqrt2 qbar^2 L/s <= sqrt2 e^{-x} L/s and L/s <= 1/(2 sqrt rho_min)... use L/s <= L/sigma, L <= Lmax (L^2 < 1 shown below)
        # x e^{-x} > U sigma/sqrt2  ->  x < x_max  (in logs)
        lr = np.log(U / SQ2) - lnsig
        x_max = brentq(lambda x: np.log(x) - x - lr, 1.0, 1e7) + 1e-9
        sup_xg = cell_sup(xgamma_up, 1e-9, x_max)
        # Lemma AY1: |cos(R/sqrt2) - cos(w/sqrt2)| <= |w||R-w|/2 + (R-w)^2/4, sqrt E(R-w)^2 <= r <= 1.8 sigma, so
        # dR/L^2 <= 0.9 sigma/L + 0.81 sigma^2/L^2 <= 0.9 sqrt(sigma eta G theta) + 0.81 sigma eta G theta  (L^2 > sigma/(eta G theta)); for 1 < L <= sigma^{-1/2} dR <= 0.9 sqrt(sigma) + ...
        dRrel = 0.9 * np.sqrt(sig * eta * G * th) + 0.81 * sig * eta * G * th + 0.9 * np.sqrt(sig)
        phi1 = 1 - np.exp(-0.25) - dRrel
        L2min_over_sig = 1 / (eta * G * th)                                  # L^2 > sigma/(eta G theta)  (P4)
        Phi_min_over_sig = phi1 * L2min_over_sig                              # Phi_C >= phi1 min(L^2, 1) >= phi1 L2min
        tau0_over_sig = 2 * eta * G * th; tau_over_sig = tau0_over_sig + sig
        condA = Phi_min_over_sig >= 2 * (2 * tau0_over_sig + sig)
        caseB_limit = U / (32 * eta * G * th * sup_xg)                        # need sigma below this
        r_wp_over_sig = 16 * eta * G * th * sup_xg / U                        # sup sigma^2 gamma / W_p,min, divided by sigma
        beta = r_wp_over_sig * sig * Phi_min_over_sig / (Phi_min_over_sig - tau0_over_sig) + tau_over_sig / (Phi_min_over_sig - tau0_over_sig)
        Wp_max = np.exp(-1) * (17 / 16) / (2 * np.sqrt(3)); b_max = beta * Wp_max
        nu = sig * sig + tau0_over_sig * sig + 2.0 * r_wp_over_sig * sig          # Phi_C <= 2
        a2 = (1 - b_max) ** 2 / 2; V_max = 2 * (-np.log1p(-nu)) / a2
        om_term = 0.0 if sig == 0 else cell_sup(lambda a, b: sig * gamma_up(a, b) * np.sqrt(2.0) * np.sqrt(16 * eta * G * th * b / (sig * U)), 1e-9, x_max)
        omega = (1 - b_max) * np.exp(-V_max / 8) - om_term
        Gam2 = 2 * eta * G * th * (1 + beta) + sig; Gam3 = 2 * eta * G * th * (1 + beta) * (1 + QB * QB) * (1 + RHO) ** 1.5
        L_gt_1_excluded = bool(phi1 * omega ** 2 / Gam2 ** 2 > Wp_max)
        K = np.log(Gam2 ** 2 * Gam3 * (1 + QB * QB) / (U ** 3 * phi1 * omega ** 2))
        X = brentq(lambda X: 4 * X - lnsig - 5 * np.log(X) - K, 1.25, 1e9) + 1e-9
        S1 = cell_sup(S1_up, -2 * np.log(QB), 60.0, 3001)
        # Lemma AY2 inside the case g < G: worst case over x in [x_low, X], L^2 <= L2max(x), s^2 in [sigma/(Gam3 x), L^2/4]
        K1max = K1(QB, RHO, 60); x_low = U0 / (1 + K1max / 16)
        L2max = min(1.0, 0.5 * Gam2 ** 2 * x_low * (17 / 16) * np.exp(-x_low) / (2 * SQ2 * phi1 * omega ** 2)) if x_low > 1 else 1.0   # x e^{-x} decreasing for x > 1
        s2max = L2max / 4; sigma_over_s_max = 0.0 if sig == 0 else np.sqrt(sig * Gam3 * X)
        qb = min(QB, np.exp(-x_low / 2)); xi = 0.82 * sig
        l4 = (eta * G * th * sig) ** 2                                         # sigma^4/L^4 <= (eta G theta sigma)^2
        rw = V_max / 8 + (1 + 3 * l4) * (0.75 * L2max + 0.75 * 0.53 * s2max) + 0.75 * (xi ** 2 + sig ** 2 * sigma_over_s_max ** 2 * qb ** 2 / (E * (1 - qb) ** 2))
        rC = 0.5 * V_max + (1 + 3 * l4) * (L2max + s2max) / 4 + 0.75 * (xi ** 2 + sig ** 2 * sigma_over_s_max ** 2 * qb ** 2 / (E * (1 - qb) ** 2))
        dH = sig * eta * G * th + sigma_over_s_max ** 2 * (qb / (1 - qb) + qb * qb / (E * (1 - qb) ** 2)) + np.sqrt(X) * sigma_over_s_max * qb / (np.sqrt(E) * (1 - qb) ** 2)
        F_int = (1 - eta * G * sig) * (1 - dH) * (1 - rC) * np.sqrt(max(0.0, 1 - rw))
        Fnew = (1 - abs(eps)) * F_int
        if not iterate or abs(Fnew - F) < 1e-12 or Fnew >= F: break
        F = Fnew
    g = min(G, (1 - abs(eps)) * F_int / (2 * np.sqrt(X + S1 + 0.25)))
    ok = bool(condA and (sig == 0 or sig <= caseB_limit) and omega > 0 and 0 < beta < 1 and nu < 1 and L_gt_1_excluded and F_int > 0 and (1 - abs(eps)) * F_int >= F - 1e-12)
    out.update({'U': U, 'x_max': x_max, 'sup_xgamma': sup_xg, 'phi1': phi1, 'beta': beta, 'nu': nu, 'V_max': V_max, 'omega': omega, 'Gam2': Gam2, 'Gam3': Gam3,
                'K': K, 'X': X, 'X_minus_quarter_ln': X - lnsig / 4, 'S1': S1, 'x_low': x_low, 'L2max': L2max, 'rw': rw, 'rC': rC, 'dH': dH, 'F_int': F_int,
                'F_used': F, 'g_bound': g, 'g_bound_sqrt_ln': g * np.sqrt(lnsig), 'condA': bool(condA), 'caseB_limit': caseB_limit, 'L_gt_1_excluded': L_gt_1_excluded, 'ok': ok})
    return out


if __name__ == '__main__':
    res = {}
    for eps in (0.0, 0.01, 0.1):
        for e in (24, 48, 96, 256, 1024, 4096, 16384):
            r = certificate(e * np.log(10), eps=eps); res['eps%g_1e-%d' % (eps, e)] = r
            print('eps %-4g sigma 1e-%-6d ok %s G %.4f K %.2f X-ln/4 %+.2f | F_int %.6f (x_low %.1f, L2max %.1e) | g >= %.5f  x sqrt(ln) %.4f  /(1-eps) %.4f' % (
                eps, e, r['ok'], r['G'], r['K'], r['X_minus_quarter_ln'], r['F_int'], r['x_low'], r['L2max'], r['g_bound'], r['g_bound_sqrt_ln'], r['g_bound_sqrt_ln'] / (1 - eps)), flush=True)
    json.dump(res, open('results/ay_certificate.json', 'w'), indent=1, default=float)
