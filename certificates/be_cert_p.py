"""Stage BE §1.2: Theorem AX1-P — the strip family's sharp lower bound extended to periodic windows (strip, comb, any even fundamental-dominated
profile), with every supremum certified cell by cell (monotone factors, as Theorem AY1).

Family P (be_plemmas.py): window profile f = 1 + sum_n a_n cos(n t) >= 0, even, with
 (P-S1) Qbar = sum_n qbar_n <= 1/4,  (S2) rho <= 1/8,  Lw <= 1,  |eps| <= eps_max,  F_c >= F (AT2's state factor, as in Theorem AX1),
 (S5) |a_n| <= |a_1| (<= 2 for every positive profile; the comb has a_1 -> 2),  (S6) x = (k' sigma)^2 >= 1.
With r_m(x) = e^{-(m^2-1)x/2}:  qbar_m <= qbar_1 r_m,  q0_m <= q0_1 r_m;  q1(x) = min(1/4, 2 e^{-x/2}) >= qbar_1;  q0_1^2 <= 4 e^{-x}.
 eta_p = sum_{m>=2} m^p r_m (linear sums: sum m^p qbar_m <= qbar_1(1 + eta_p)),  zeta_p = sum_{m>=2} m^p r_m^2 (square sums).
P-lemmas used (proved in material §7.24): AW1-P (exact), AW3 (general), AX1-P lower J_ww >= (1 + u0/(8(1 + Qbar)))/L^2, AX2-P W_p <= x I2/(2(1 - Qbar)),
 AX4-P S <= kappa2(Qbar, rho)(1 + zeta_2) u0, AX3-P with gamma_P = [x(N2(1 + Qbar) + N1^2) + sqrt(x) N1 (1 + 2 Qbar)/sqrt(e)]/(1 - Qbar)^2,
 AW4(i)-P (Wiener majorant) for the final k_eff^2 <= x S_i(x) + 1/4, and AW6/AX3 (even profile: the parity step).
The chain is Theorem AX1's with: U -> U_P = U0/(kappa2 (1 + zeta_2)), the (P4) factor (1 + Qbar), Gam3 -> Gam3_P, (1 + qbar^2) -> (1 + zeta_2)/(1 - Qbar),
 q0^2 <= e^{-x} -> 4 e^{-x} (1 + zeta_0): K_P = ln(256 Gam2^2 Gam3_P (1 + zeta_2)(1 + zeta_0)^3 /((1 - Qbar) U_P^3 phi omega^2)).
-> results/be_cert_p.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import binom
from be_plemmas import S_branch_i

SQ2 = np.sqrt(2.0); QB, RHO = 0.25, 0.125; E = np.e; MMAX = 60


def r_m(x, m): return np.exp(-(m * m - 1) * x / 2)
def eta(p, x): return sum(m ** p * r_m(x, m) for m in range(2, MMAX))
def zeta(p, x): return sum(m ** p * r_m(x, m) ** 2 for m in range(2, MMAX))
def q1(x): return np.minimum(QB, 2 * np.exp(-np.asarray(x, float) / 2))
def kap2(Q=QB, rho=RHO): return (1 + rho + Q / (2 * (1 - Q))) ** 2 / (8 * (1 - Q))
def Qb(x): return min(QB, float(q1(x)) * (1 + eta(0, x)))


def gamma_P(xa, xb):
    """cell bound of gamma_P on [xa, xb]: increasing factors at xb, decreasing (q-sums) at xa."""
    q = float(q1(xa)); N1 = q * (1 + eta(1, xa)); N2 = q * (1 + eta(2, xa)); Q = Qb(xa)
    return (xb * (N2 * (1 + Q) + N1 * N1) + np.sqrt(xb) * N1 * (1 + 2 * Q) / np.sqrt(E)) / (1 - Q) ** 2


def cell_sup(f, x0, x1, n=4001):
    xs = np.linspace(x0, x1, n); return float(max(f(a, b) for a, b in zip(xs[:-1], xs[1:])))


def S1_P(x0=1.0, x1=200.0, n=2000):
    """sup_x x (S_i(x) - 1): S_i from the Wiener majorant with the worst-case q0's at the cell's left end (decreasing in x), x at the right end."""
    xs = np.linspace(x0, x1, n); best = 0.0
    for a, b in zip(xs[:-1], xs[1:]):
        q = float(q1(a)); q0 = np.array([q * r_m(a, m) for m in range(1, MMAX)])
        Si, h = S_branch_i(q0); best = max(best, b * (Si - 1))
    return best


def certificate(sig, G=0.15, eps_max=0.01, F=1.0, x_lo=1.0):
    eta_ = 1 / (1 - eps_max); th = 1 / (1 - eta_ * G * sig); U0 = F * F / (4 * G * G) - 0.25
    z0, z2 = zeta(0, x_lo), zeta(2, x_lo); kp = kap2(); UP = U0 / (kp * (1 + z2)); Ueff = UP / (1 + QB)     # Ueff: (P4-P) factor 1 + Qbar
    out = {'sigma': sig, 'G': G, 'eta': eta_, 'U_P': UP, 'zeta0': z0, 'zeta2': z2}
    rhs = UP * sig / (SQ2 * 4 * (1 + z0) * np.sqrt(1 + sig * sig))
    x_max = brentq(lambda x: np.log(x) - x - np.log(rhs), 1.0, 1e7) if rhs < np.exp(-1) else None
    if x_max is None: out['ok'] = False; return out
    sup_xg = cell_sup(lambda a, b: b * gamma_P(a, b), x_lo, x_max)
    sup_g = cell_sup(gamma_P, x_lo, x_max)
    xi = sig * cell_sup(lambda a, b: np.sqrt(b) * float(q1(a)) * (1 + eta(1, a)) / (1 - Qb(a)), x_lo, max(x_max, 60.0))   # sigma^2 |s~| <= sigma sqrt(x) N1/(1 - Qbar)
    L2min = sig / (eta_ * G * th); Lmax2 = 1 + sig * sig
    Ls = np.sqrt(np.exp(np.linspace(np.log(L2min), np.log(Lmax2), 2001)))
    Phi_lo = (np.sqrt(np.maximum(Ls ** 2 - 2 * sig * sig, 0)) - xi) ** 2 / 4 - (3 ** 0.25 * Ls + xi) ** 4 / 96
    phi = float(np.min(Phi_lo / Ls ** 2)); Phi_min = phi * L2min; Phi_max = (np.sqrt(Lmax2) + xi) ** 2 / 4
    tau0 = 2 * eta_ * G * sig * th; tau = tau0 + sig * sig
    condA = Phi_min >= 2 * (2 * tau0 + sig * sig)
    sig_caseB = Ueff / (32 * eta_ * G * th * (1 + sig * sig) * sup_xg)
    r_wp = sig * 16 * eta_ * G * th * (1 + sig * sig) * sup_xg / Ueff
    beta = r_wp * Phi_min / (Phi_min - tau0) + tau / (Phi_min - tau0)
    Wp_max = cell_sup(lambda a, b: b * min(QB ** 2, 4 * np.exp(-a)), x_lo, max(x_max, 60.0)) * (1 + z2) * (1 / 3) / (2 * (1 - QB))
    b_max = beta * Wp_max; nu = sig * sig + tau0 + Phi_max * r_wp
    a2 = (1 - b_max) ** 2 / 2; V_max = 2 * (-np.log1p(-nu)) / a2
    om_term = cell_sup(lambda a, b: sig * gamma_P(a, b) * np.sqrt(Phi_max) * np.sqrt(16 * eta_ * G * th * b * (1 + sig * sig) / (sig * Ueff)), x_lo, x_max)
    omega = (1 - b_max) * np.exp(-V_max / 8) - om_term
    Gam2 = 2 * eta_ * G * th * (1 + beta) + sig
    Gam3P = 2 * eta_ * G * th * (1 + beta) * (1 + QB) * (1 + z2) * (1 + RHO) ** 1.5 / (1 - QB)
    K = np.log(256 * Gam2 ** 2 * Gam3P * (1 + z2) * (1 + z0) ** 3 / ((1 - QB) * UP ** 3 * phi * omega ** 2))
    out.update({'x_max': x_max, 'sup_xgamma': sup_xg, 'xi_over_sigma': xi / sig, 'phi': phi, 'beta': beta, 'nu': nu, 'V_max': V_max, 'omega': omega,
                'Wp_max': Wp_max, 'K': K, 'cond_PhiC': bool(condA), 'cond_caseB': bool(sig <= sig_caseB), 'sigma_caseB_limit': sig_caseB,
                'ok': bool(condA and sig <= sig_caseB and omega > 0 and 0 < beta < 1 and nu < 1)})
    return out


if __name__ == '__main__':
    S1 = S1_P(); print('S1_P = sup_x x(S_i(x) - 1) = %.4f (strip: 1.91)' % S1, flush=True)
    ax = json.load(open('results/ax_theorem.json')); res = {'S1_P': S1}
    for e in (24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 4096, 16384):
        sig = 10.0 ** (-e) if e < 300 else 1e-300; L = e * np.log(10); r = certificate(sig)
        X = brentq(lambda X: 4 * X - L - 5 * np.log(X) - r['K'], 1.0, 1e9); g = min(r['G'], 1.0 / (2 * np.sqrt(X + S1 + 0.25)))
        r.update({'X': X, 'X_minus_quarter_ln': X - L / 4, 'g_bound': g, 'g_bound_sqrt_ln': g * np.sqrt(L), 'strip_AX1_sqrt_ln': ax.get('1e-%d' % e, {}).get('g_bound_sqrt_ln')})
        res['1e-%d' % e] = r
        print('sigma 1e-%-6d ok %s beta %.3f omega %.4f K_P %.2f | X - ln/4 %+.2f | g >= %.5f  x sqrt(ln) %.4f  (strip AX1: %s)' % (
            e, r['ok'], r['beta'], r['omega'], r['K'], X - L / 4, g, g * np.sqrt(L), '%.4f' % r['strip_AX1_sqrt_ln'] if r['strip_AX1_sqrt_ln'] else '-'), flush=True)
    json.dump(res, open('results/be_cert_p.json', 'w'), indent=1, default=float)
