"""Stage AX: Theorem AX1 (the sharp strip constant) — an explicit certificate of every constant in the three-branch argument.

Strip assumptions (H_s): qbar <= 1/4, rho = s^2/(2L^2) <= 1/8, Lw <= 1, |eps| <= eps_max (paper E's remainder, a state quantity), F_c >= F.
Proposition AX4 (proof): S <= 8 kappa2 u/(8 + u) for every u, kappa2 = (1 + rho + qbar/(2(1-qbar)))^2/(8(1-qbar)) -> 1/8
   (|d_w d_t sqrt p| <= [|(sqrt Gamma)' q| + sqrt Gamma |q'|(1 + qbar/(2(1-qbar)))]/(2 sqrt(1-qbar)) |sin t|, and int F^2 = J_ww/4 >= (1/L^2 + B^2/2)/4 by Lemma AX1).
Theorem AX1 (proof, constants below):  for every strip satisfying (H_s) and sigma <= sigma_0(G),
   g >= min( G, F_c / (2 sqrt(X + S1 + 1/4)) ),   X the largest root of  4X = ln(1/sigma) + ln(Gam2^2 Gam3 (17/16) X^5 / (U^3 phi omega^2)),
   so  g sqrt(ln(1/sigma)) >= F_c (1 - (5/2) ln ln(1/sigma)/ln(1/sigma) - O(1/ln(1/sigma))).
Chain (all inequalities proved in material §7.18):
 (P1) g >= (1-|eps|)((a) + (a'));  g < G  =>  sigma^2 J_ww <= eta G sigma D,  N_phi <= eta G sigma D,  eta = 1/(1-|eps|), D = den
 (P3) D <= theta (b + W_p),  theta = 1/(1 - eta G sigma);   b + W_p = sigma^2 J_vv <= 1
 (P4) Lemma AX1: sigma^2/L^2 (1 + u/8) <= sigma^2 J_ww  =>  L^2 > sigma/(eta G theta),  u < 8 eta G theta L^2 (b + W_p)/sigma
 (P5) Lemma AX3: (1-e-sigma^2) W_p + Phi_C (b - sigma^2 gamma) <= 2 eta G sigma theta (b + W_p)
 (P6) AT2 + AW3 + AX4: g < G  =>  u > U/x,  U = (F^2/(4G^2) - 1/4)/kappa2
 case Wp <= b excluded for sigma <= sigma_0;  b <= beta W_p;  1 - e <= nu (V tiny);  AM-GM in (P5)  =>  L^3/s < Gam2^2 x (1+qbar^2) q0^2/(2 sqrt2 phi omega^2)
 (P6) => L/s > U/(sqrt2 x q0^2);  (P4) + closed forms  =>  s^2 > sigma/(Gam3 x);  combine with q0^2 <= e^{-x}.
-> results/ax_theorem.json.  Python 3.11."""
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

SQ2 = np.sqrt(2.0)
QB, RHO = 0.25, 0.125


def kappa2(qb=QB, rho=RHO): return (1 + rho + qb / (2 * (1 - qb))) ** 2 / (8 * (1 - qb))


def gamma_x(x, qb=QB):
    q = min(qb, np.exp(-x / 2)); return q * (x * (1 + q) + np.sqrt(x)) / (1 - q) ** 2


def certificate(sig, G=0.15, eps_max=0.01, F=1.0):
    eta = 1 / (1 - eps_max); th = 1 / (1 - eta * G * sig); kap = kappa2(); U0 = F * F / (4 * G * G) - 0.25; U = U0 / kap
    out = {'sigma': sig, 'G': G, 'eta': eta, 'theta': th, 'kappa2': kap, 'U': U}
    # a priori x range: x e^{-x} > U sigma /(sqrt2 sqrt(1+sigma^2))  (u <= sqrt2 qbar^2 L/s, L <= sqrt(1+sigma^2), s >= sigma)
    rhs = U * sig / (SQ2 * np.sqrt(1 + sig * sig))
    if rhs >= np.exp(-1): out['ok'] = False; out['why'] = 'sigma too large for the a-priori x range'; return out
    x_max = brentq(lambda x: np.log(x) - x - np.log(rhs), 1.0, 1e6); out['x_max'] = x_max
    xs = np.linspace(1e-6, x_max, 200001); gx = np.array([gamma_x(x) for x in xs[::100]]); xg = xs[::100]
    sup_xgamma = float(np.max(xg * gx)); sup_gamma = float(np.max(gx))
    L2min = sig / (eta * G * th); Lmax2 = 1 + sig * sig
    xi = sig * 0.82                                                     # sigma^2 |s~| <= sigma sqrt(x) qbar/(1-qbar) <= 0.82 sigma
    Ls = np.sqrt(np.exp(np.linspace(np.log(L2min), np.log(Lmax2), 2001)))
    Phi_lo = (np.sqrt(np.maximum(Ls ** 2 - 2 * sig * sig, 0)) - xi) ** 2 / 4 - (3 ** 0.25 * Ls + xi) ** 4 / 96
    phi = float(np.min(Phi_lo / Ls ** 2)); Phi_min = phi * L2min; Phi_max = (np.sqrt(Lmax2) + xi) ** 2 / 4
    tau0 = 2 * eta * G * sig * th; tau = tau0 + sig * sig
    out.update({'phi': phi, 'Phi_min': Phi_min, 'tau0': tau0})
    # condition for excluding case W_p <= b, and for D1
    condA = Phi_min >= 2 * (2 * tau0 + sig * sig)
    sig_caseB = U / (32 * eta * G * th * (1 + sig * sig) * sup_xgamma)                    # need sigma below this
    condB = sig <= sig_caseB
    # beta, nu, V, omega
    r_wp = sig * 16 * eta * G * th * (1 + sig * sig) * sup_xgamma / U                          # sup_x sigma^2 gamma(x) / W_p,min(x)
    beta = (r_wp * Phi_min / (Phi_min - tau0)) + tau / (Phi_min - tau0)
    Wp_max = np.exp(-1) * (17 / 16) / (2 * np.sqrt(3)); b_max = beta * Wp_max
    nu = sig * sig + tau0 + Phi_max * r_wp
    a2 = (1 - b_max) ** 2 / 2; V_max = 2 * (-np.log1p(-nu)) / a2
    Wp_min_sqrt_ratio = np.sqrt(sig * 16 * eta * G * th * (1 + sig * sig) * max(xg[xg > 0] * 0 + 1) / U)   # placeholder, replaced below
    # sigma gamma sqrt(Phi) / sqrt(W_p,min): W_p,min(x) = sigma U/(16 eta G theta x (1+sigma^2)) -> sigma gamma(x) sqrt(Phi_max) sqrt(16 eta G theta x (1+sigma^2)/(sigma U))
    om_term = float(np.max(sig * gx * np.sqrt(Phi_max) * np.sqrt(16 * eta * G * th * xg * (1 + sig * sig) / (sig * U))))
    omega = (1 - b_max) * np.exp(-V_max / 8) - om_term
    Gam2 = 2 * eta * G * th * (1 + beta) + sig; Gam3 = 2 * eta * G * th * (1 + beta) * (1 + QB * QB) * (1 + RHO) ** 1.5
    K = np.log(Gam2 ** 2 * Gam3 * (1 + QB * QB) / (U ** 3 * phi * omega ** 2))
    f = lambda X: 4 * X - np.log(1 / sig) - 5 * np.log(X) - K
    X = brentq(f, 1.0, 1e7)                                                              # largest root (f increasing for X > 5/4)
    xs1 = np.linspace(-2 * np.log(QB), 60, 6001); q = np.exp(-xs1 / 2); S1 = float(np.max(xs1 * np.array([K1(a, RHO, 60) for a in q]) * q * q))
    g_bound = min(G, F / (2 * np.sqrt(X + S1 + 0.25)))
    out.update({'cond_PhiC': bool(condA), 'cond_caseB': bool(condB), 'sigma_caseB_limit': sig_caseB, 'beta': beta, 'nu': nu, 'V_max': V_max, 'omega': omega,
                'Gam2': Gam2, 'Gam3': Gam3, 'K': K, 'X': X, 'X_minus_quarter_ln': X - np.log(1 / sig) / 4, 'S1': S1, 'g_bound': g_bound,
                'g_bound_sqrt_ln': g_bound * np.sqrt(np.log(1 / sig)), 'ok': bool(condA and condB and omega > 0 and 0 < beta < 1 and nu < 1)})
    return out


if __name__ == '__main__':
    res = {}
    LO = json.load(open('results/as_strip_lo.json'))
    for e in (20, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 4096, 16384):
        sig = 10.0 ** (-e) if e < 300 else None
        if sig is None:                                   # work in logs: only ln(1/sigma) enters X; small-sigma factors are ~1 (use 1e-300 for them)
            r = certificate(1e-300); L = e * np.log(10)
            X = brentq(lambda X: 4 * X - L - 5 * np.log(X) - r['K'], 1.0, 1e9); r['X'] = X; r['X_minus_quarter_ln'] = X - L / 4
            r['g_bound'] = min(r['G'], 1.0 / (2 * np.sqrt(X + r['S1'] + 0.25))); r['g_bound_sqrt_ln'] = r['g_bound'] * np.sqrt(L); r['sigma'] = '1e-%d' % e
        else:
            r = certificate(sig)
        key = '1e-%d' % e; lo = LO.get(key, {}).get('g_sqrt_ln')
        r['strip_min_LO_sqrt_ln'] = lo; res[key] = r
        print('sigma 1e-%-6d ok %s  beta %.3f omega %.4f K %.2f | X = ln/4 %+.2f | g >= %.4f  (x sqrt(ln) %.4f; strip family minimum (leading order) x sqrt(ln) %s)' % (
            e, r['ok'], r['beta'], r['omega'], r['K'], r['X_minus_quarter_ln'], r['g_bound'], r['g_bound_sqrt_ln'], '%.4f' % lo if lo else '-'), flush=True)
    json.dump(res, open('results/ax_theorem.json', 'w'), indent=1, default=float)
