"""Stage BQ §2.1: the constants of the paper-level proofs of Appendix E (v1.2), in outward-rounded interval arithmetic (ba_ivl, 60 decimal digits).

  Lemma 32 (case A1)   tau > 80 and x <= 250 sigma^2  =>  (a) >= c.          threshold T: (a) < c forces x > T sigma^2, T > 250.
  Lemma 31 (case C)    tau <= 80 and b < c' sigma       =>  (a') >= c.         lower bound of (a') / c > 1.
  Lemma 33 (case A2)   the theta remainder E_R of Lemma BL3 is absorbed:      eps_R = 5 (A0 + B0 + C0).
  Theorem 34           the window-inequality conclusion for strips:            g >= 0.9 A_min / (2 sqrt(k2)) > 0.134.
  Lemma 35 (N <= 6)    the same conclusion with sup_x x (1 + Z2(x)) and 2 E_R + eps_theta <= eps_in.

Every input that the proofs take from a computer-assisted certificate (A_min of Lemmas 33 and 35, sup x(1 + Z2), sup E_R) is read from the stored
results and rounded down (resp. up) in the last printed digit; every other constant is computed here in interval arithmetic.

Numerical illustrations (not part of the proofs; they test the two monotonicity lemmas E1(c) and E2 on random cells and states):
  sigma_lemma   Lemma 20 (vi): a cell certified at sigma_hat also closes at every sigma < sigma_hat (margins compared with relative tolerance 1e-9: float rounding)
  b_scaling     Lemma 30: at fixed r = sigma/b and tau, F_b(Lambda)/b does not decrease and Lambda_1(b) does not decrease as b decreases
-> results/bq_proofs.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
from decimal import Decimal as D
import numpy as np
from ba_ivl import I

PI = I('3.14159265358979323846264338327950288419716939937510582097494', '3.14159265358979323846264338327950288419716939937510582097495')
ONE = I(1)
CSTAR = I('0.134') / I('0.9')
SIG = I('1e-24')
Q = I('0.25'); RHO = I('0.125')
CP = I('0.002')                                    # c'
TAU0 = I(80)


def f(x): return float(x.lo), float(x.hi)


def e1(tau): return (-(tau / 2)).exp()             # e_1 = e^{-tau/2}


def sum_e(tau, k=0, beta=I(0), nmax=12):
    """2 sum_{n>=1} n^k e^{-n^2 tau/2 + n beta}, with an explicit tail bound (successive terms fall by more than 1/2 for n >= nmax, tau >= 80)."""
    s = I(0)
    for n in range(1, nmax + 1):
        s = s + I(n ** k) * (-(I(n * n) * tau / 2) + I(n) * beta).exp()
    last = I((nmax + 1) ** k) * (-(I((nmax + 1) ** 2) * tau / 2) + I(nmax + 1) * beta).exp()
    return 2 * (s + I(0, (2 * last).hi))


def geom():
    """the window geometry of (P4): rho = s^2/(2L^2) <= 1/8."""
    g1 = (2 * RHO / (1 + 2 * RHO)).sqrt()          # s/sqrt(s^2 + L^2) <= 1/sqrt5
    rs = 2 * RHO / (1 + 2 * RHO)                   # s^2/(s^2 + L^2) <= 1/5
    gs = (RHO / (1 + RHO)).sqrt()                  # s/sqrt(s^2 + 2L^2) <= 1/3
    return g1, rs, gs


# ---------------------------------------------------------------- Lemma 32: case A1
def case_A1(c=CSTAR, sig=SIG):
    g1, rs, gs = geom()
    th = 1 / (1 - c * sig)
    e = e1(TAU0)                                   # e_1 <= e^{-40} for tau >= 80
    delta = sum_e(TAU0)                            # delta_tau = 2 sum e_n
    kw = (1 + Q * g1 * rs * e) / (1 - Q * g1 * e)  # exact w-marginal Cramer-Rao (Lemma 20 (ii)), H0 = e_1 for a strip
    Wc = ((1 + delta) * Q * Q * gs / (2 * (1 - Q)) + g1 * Q * e) / (1 - g1 * Q * e)       # W <= Wc x (Lemma 20 (iii), one harmonic)
    bw = 1 / TAU0 + Wc                             # b + W <= bw x  (b = x/tau < x/80)
    T = 1 / (bw * c * th * kw * (1 + sig * sig * sig))    # (a) < c  =>  x > T sigma^2   and   L^2 > T (1 + sigma^3) sigma/x
    return {'kappa_w': f(kw), 'W_coef': f(Wc), 'b_plus_W_over_x': f(bw), 'T': f(T), 'T_over_250_lower': float((T / 250).lo),
            'ok': bool(T.lo > 250), 'L2_times_x_over_sigma_lower': float((T * (1 + sig * sig * sig)).lo)}


# ---------------------------------------------------------------- Lemma 31: case C
def case_C(c=CSTAR, sig=SIG, cp=CP):
    g1, rs, gs = geom()
    th = 1 / (1 - c * sig)
    kw = (1 + Q * g1 * rs) / (1 - Q * g1)          # H0 = e^{-tau/2} <= 1 for every tau
    bW = 1 + TAU0 / 3                              # b + W <= (1 + tau/3) b, W <= x/3 = tau b/3 (W <= x (q^2/(1-q) + q))
    Vmax = bW * c * th * kw * (1 + sig * sig * sig)       # V = sigma^2/b < Vmax
    al_lo = (1 - cp * sig) / I(2).sqrt(); al_hi = 1 / I(2).sqrt()
    assert (al_hi.sq() * Vmax / 4).hi < 1          # alpha^2 e^{-alpha^2 V/4} increases in alpha^2: worst at the smallest alpha
    x = TAU0 * cp * sig                            # x <= 80 b < 80 c' sigma
    A = (1 + 2 * Q) / (1 - Q).sq(); B = (1 + 2 * Q) / (I(1).exp().sqrt() * (1 - Q).sq())
    gam = Q * (x * A + x.sqrt() * B)               # gamma_P <= Qbar Gamma1(x) (strip), increasing in x
    cchi = (x.sqrt() * Q / (1 - Q) + (2 / PI).sqrt() + Q * (-I('0.5')).exp() / (1 - Q)) / I(2).sqrt()
    E = al_hi * (1 + Q) * sig * cchi / 2 + (1 + Q) * gam / I(2).sqrt()
    main = al_lo.sq() * (-(al_lo.sq()) * Vmax / 4).exp() / (4 * (1 + Q).sq())
    two_Nphi = main - al_hi * E                    # 2 N_phi / sigma^2 >= main - alpha E   (derivation in App. E)
    a_prime = two_Nphi / (2 * th * bW * cp)        # (a') >= (2 N_phi/sigma^2) sigma/(2 theta (1 + 80/3) b) > that / c'
    return {'kappa_w': f(kw), 'V_max': f(Vmax), 'main': f(main), 'E': f(E), 'two_Nphi_over_sigma2_lower': float(two_Nphi.lo),
            'a_prime_lower': float(a_prime.lo), 'ratio_lower': float((a_prime / c).lo), 'ok': bool(a_prime.lo > c.hi)}


# ---------------------------------------------------------------- Lemma 33: eps_R (case A2), and the conclusion
def eps_R():
    tau = TAU0; beta = TAU0 / I(500).sqrt()        # x > 250 sigma^2 gives beta <= sigma tau/sqrt(2x) < tau/sqrt(500)
    nu = {k: sum_e(tau, k, beta) for k in range(5)}
    N1 = Q
    A = (1 + 2 * Q) / (1 - Q).sq(); B = (1 + 2 * Q) / (I(1).exp().sqrt() * (1 - Q).sq())
    gmax = Q * (A + B)                             # gamma_P <= Qbar Gamma1(x) <= Qbar (A + B) on x <= 1
    cc = (N1 / (1 - Q) + (2 / PI).sqrt() + Q * (-I('0.5')).exp() / (1 - Q)) / I(2).sqrt()
    r2 = I(2).sqrt()
    A0 = tau.sq() * N1.sq() * nu[1] / (r2 * (1 - Q)) + tau * nu[1] * (1 + Q) * gmax / r2
    B0 = I('0.375') * tau * Q * nu[0] + Q * tau.sq() * nu[2] / 8 + tau * nu[0] * (1 + Q) * gmax + Q * (tau.sq() * tau) * nu[4] / 8
    C0 = (1 + Q) * (1 + Q / 2) * cc * nu[0] + tau * nu[1] * N1 * cc + tau * Q * nu[2] * (1 + Q) * cc / 2 + tau * Q * nu[2] / (2 * (1 - nu[0]))
    eR = 5 * (A0 + B0 + C0)
    # tau-monotonicity of tau^j e^{-n^2 tau/2 + n tau/sqrt500}, j <= 3, n >= 1, tau >= 80: d/dtau log = j/tau - n^2/2 + n/sqrt500 < 0
    slope = I(3) / tau - I('0.5') + 1 / I(500).sqrt()
    # the absorption: (i) 254 sigma/x <= 1: (A0 sqrt x + B0 sigma + C0 x) <= (A0 + B0 + C0) <= 50.8 eps_R; (ii) otherwise sigma/sqrt x < 1/sqrt250
    case_i = (A0 + B0 + C0) / (I('0.2') * 254 * eR)
    case_ii = (A0 / I(250).sqrt() + B0 / 250 + C0) / (I('0.2') * eR)
    # beta-dependent theta corrections that enter Lemma BL3's Phi-proportional remainder: 2 Qbar (nu_0 + tau nu_2) (relative, Phi <= 4 min(Phi, 1/2))
    phi_rem = 2 * Q * (nu[0] + tau * nu[2])
    return {'nu': {k: f(v) for k, v in nu.items()}, 'A0': f(A0), 'B0': f(B0), 'C0': f(C0), 'eps_R_upper': float(eR.hi), 'slope_upper': float(slope.hi),
            'absorb_case_i_upper': float(case_i.hi), 'absorb_case_ii_upper': float(case_ii.hi), 'phi_remainder_upper': float(phi_rem.hi),
            'ok': bool(slope.hi < 0 and case_i.hi <= 1 and case_ii.hi <= 1 and eR.hi < 1e-9 and phi_rem.hi < 1e-9)}


def conclusion(A_min_lo, eps, sup_xz=I(1)):
    kap2 = (1 + RHO + Q / (2 * (1 - Q))).sq() / (8 * (1 - Q))
    k2 = (1 + eps) * (8 * kap2 * (1 + Q) * sup_xz + I('0.25'))
    g = I('0.9') * A_min_lo / (2 * k2.sqrt())
    return {'kappa2': f(kap2), 'k_eff2_max': f(k2), 'g_lower': float(g.lo), 'ratio_to_0.134_lower': float((g / I('0.134')).lo), 'ok': bool(g.lo > D('0.134'))}


def floor(x, d=6): return D(str(np.floor(x * 10 ** d) / 10 ** d))
def ceil(x, d=6): return D(str(np.ceil(x * 10 ** d) / 10 ** d))


# ---------------------------------------------------------------- numerical illustrations of Lemma 20 (vi) and Lemma 30
def sigma_lemma(n=400, seed=5):
    import bk_narrow_lemma as BK, bp_audit as AU
    rng = np.random.default_rng(seed); out = {}
    acc = lambda q: q['f_big'] > 0 and (q['gL1'] >= 0.71 or (q['f_L1'] > 0 and q['increasing']))
    for name, NN, xs in (('general', np.arange(1, 40, dtype=float), np.geomspace(1, 80, 241)), ('strip', np.array([1.0]), np.geomspace(1e-12, 1, 481))):
        BK.NN = NN; bad = 0; worse = 0; cnt = 0
        for _ in range(n):
            i = rng.integers(0, len(xs) - 1); xa, xb = xs[i], xs[i + 1]; bs = np.geomspace(xa / 80, 1, 241); j = rng.integers(0, 240)
            m0 = AU.bk_cell_full(BK, xa, xb, bs[j], bs[j + 1], True)['margin']
            for s in (1e-30, 1e-60, 1e-150, 1e-300):
                q = AU.bk_cell_full(BK, xa, xb, bs[j], bs[j + 1], True, None, s); cnt += 1
                bad += not acc(q); worse += q['margin'] < m0 * (1 - 1e-9)
        out[name] = {'cell_sigma_pairs': cnt, 'not_closing': bad, 'margin_below_sigma_hat': worse}
    BK.NN = np.arange(1, 40, dtype=float)
    # case-B cells (large Lambda_1 at sigma_hat): the cell at sigma_hat against direct point evaluations at sigma < sigma_hat
    BK.NN = np.array([1.0]); rows = [json.loads(l) for l in open('results/bp_strip_caseB.jsonl') if l.strip()]; bad = 0; cnt = 0
    for d in rows[::9]:
        ba, bb = d['ba'], d['bb']
        for s in (1e-26, 1e-28, 1e-30):
            if ba < 2e-3 * s: continue
            for xr in (1e-3, 1.0, 40.0):
                F = point_F(BK, xr * ba, ba, s); cnt += 1; bad += not F
    out['caseB_points_below_sigma_hat'] = {'points': cnt, 'not_closing': bad}
    BK.NN = np.arange(1, 40, dtype=float)
    return out


def point_F(BK, x, b, sig, c=0.134 / 0.9):
    """F(Lambda) > 0 for every Lambda >= Lambda_1 at a single state (x, b, sigma) of a strip (point form of the BK1 inequality)."""
    import bp_audit as AU
    q = AU.bk_cell_full(BK, x, x, b, b, True, c, sig)
    L = np.concatenate([[q['gL1']], np.geomspace(max(q['gL1'], 1e-300), 0.71, 400), [0.71]])
    F = q['main_fac'] * (-np.expm1(-L ** 2 / 2)) - q['Ea'] * np.minimum(L, 0.71) - q['Eb'] - q['rhs']
    return bool(np.all(F[L >= q['gL1']] > 0) and q['f_big'] > 0)


def b_scaling(n=2000, seed=7):
    import bk_narrow_lemma as BK, bp_audit as AU
    BK.NN = np.array([1.0]); rng = np.random.default_rng(seed); bad_F = 0; bad_L = 0; cnt = 0
    for _ in range(n):
        r = 10 ** rng.uniform(-3, np.log10(500)); tau = 10 ** rng.uniform(-4, np.log10(80)); b0 = 2e-27
        b = b0 * 10 ** (-rng.uniform(0, 40)); Lam = 10 ** rng.uniform(-6, 0.5)
        P0 = AU.bk_cell_full(BK, tau * b0, tau * b0, b0, b0, True, None, r * b0); P = AU.bk_cell_full(BK, tau * b, tau * b, b, b, True, None, r * b)
        F = lambda p, bb: (p['main_fac'] * (-np.expm1(-Lam ** 2 / 2)) - p['Ea'] * min(Lam, 0.71) - p['Eb'] - p['rhs']) / bb
        cnt += 1; bad_F += F(P, b) < F(P0, b0) - 1e-12 * abs(F(P0, b0)); bad_L += P['gL1'] < P0['gL1'] * (1 - 1e-12)
    BK.NN = np.arange(1, 40, dtype=float)
    return {'states': cnt, 'F_over_b_decreased': bad_F, 'Lambda1_decreased': bad_L}


if __name__ == '__main__':
    np.seterr(all='ignore')
    out = {'case_A1': case_A1(), 'case_C': case_C(), 'eps_R': eps_R()}
    S = json.load(open('results/bp_strip_p5.json'))
    A_strip = floor(S['case_A2_1e-24']['A_min'])
    epsA2 = I('1e-6') + I(D(str(out['eps_R']['eps_R_upper'])))
    out['conclusion_strip'] = {'A_min_used': str(A_strip), **conclusion(I(A_strip), epsA2)}
    W = json.load(open('results/bp_windowsN.json'))
    out['conclusion_N'] = {}
    for N in (2, 3, 4, 6):
        w = W['N=%d' % N]; A = floor(w['A_min']); xz = ceil(w['conclusion']['sup_x(1+Z2)']); ER = ceil(w['E_R_sup'], 9)
        out['conclusion_N'][N] = {'A_min_used': str(A), 'sup_xZ_used': str(xz), 'E_R_used': str(ER),
                                  'two_ER_plus_eps_theta_upper': float((2 * I(ER) + I('7.1e-14')).hi), **conclusion(I(A), I('1e-6'), I(xz))}
        out['conclusion_N'][N]['ok'] = out['conclusion_N'][N]['ok'] and out['conclusion_N'][N]['two_ER_plus_eps_theta_upper'] <= 1e-6
    BN = json.load(open('results/bn_strip_x.json'))
    out['BN1_E_R'] = {'E_R_sup': BN['E_R_sup'], 'two_ER_plus_eps_theta_upper': float((2 * I(ceil(BN['E_R_sup'], 12)) + I('7.1e-14')).hi)}
    for k in ('case_A1', 'case_C', 'eps_R', 'conclusion_strip'): print(k, out[k], flush=True)
    for N, v in out['conclusion_N'].items(): print('N =', N, v, flush=True)
    out['sigma_lemma_numerical'] = sigma_lemma(); print('sigma lemma (numerical):', out['sigma_lemma_numerical'], flush=True)
    out['b_scaling_numerical'] = b_scaling(); print('b scaling (numerical):', out['b_scaling_numerical'], flush=True)
    out['all_ok'] = bool(out['case_A1']['ok'] and out['case_C']['ok'] and out['eps_R']['ok'] and out['conclusion_strip']['ok']
                         and all(v['ok'] for v in out['conclusion_N'].values()) and out['BN1_E_R']['two_ER_plus_eps_theta_upper'] <= 1e-6)
    print('all ok:', out['all_ok'])
    json.dump(out, open('results/bq_proofs.json', 'w'), indent=1, default=str)
