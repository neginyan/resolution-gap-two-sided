"""Stage BR: the cell of Lemma 20 / Lemma 29 for windows with at most N harmonics (N = None: any number), with the improvements of stage BR.

Cell: x in [xa, xb], b in [ba, bb], tau = x/b in [ta, tb] = [xa/bb, xb/ba]; sigma = sig (the cell is valid for every smaller sigma by Lemma 20 (vi):
M and E_a/sigma do not depend on sigma, E_b/sigma and R/sigma grow, u = Lambda_1^2/2 grows and u/sigma falls; the same structure as Lemma 20).

Harmonic sums.  y := q0_1 in (0, m(x)], m(x) = min(Qbar, 2e^{-x/2}) ((P3), a_1 <= 2, K >= x);  for n >= 2: |q0_n| <= y dec_n, dec_n = e^{-(n^2-1)x/2}
((P2), K >= x; dec_n := 1 when dec_one), and sum_n |q0_n| <= Qbar = 1/4.  For a linear sum sum_{n>=2} c_n |q0_n| the largest value under these
constraints is the greedy fill G(c) (largest c_n first, each up to y dec_n, total budget Qbar - y); every sum below is bounded by its own supremum
over y, taken on y-cells [ya, yb] with the caps at yb and the budget Qbar - ya (rigorous upper bounds):
  H0 = sum |q0_n| e_n <= y e_1 + G(e_n),  H1 = sum |q0_n| f_n <= y f_1 + G(f_n) (f_n = sup over the cell of |1 - n^2 tau| e^{-n^2 tau/2}),
  H2 = sum n^2 |q0_n| e_n <= y e_1 + G(n^2 e_n),  N1 = sum n |q0_n| <= y + G(n),  N2 = sum n^2 |q0_n| <= y + G(n^2),
  S2 = sum n^2 q0_n^2 <= y^2 + y G(n^2)   (|q0_n| <= y).
The window's v-information (Lemma 20 (iii)):  W <= W_N = x [g_s P/(1 - Qbar) + g1 H2]/(1 - g1 H0), with P bounding int Gamma <theta_tau h_t^2>/g_s:
  P <= (1 + delta) S2/2   (delta = min(sup theta_tau - 1, Toeplitz delta_N) for finite N; sup theta_tau - 1 for N = None), and
  P <= tau N2^2           (stage BR: <theta_tau sin nt sin mt> = (e_|n-m| - e_{n+m})/2 <= n m tau, since e^{-a} - e^{-b} <= b - a).
min(b + W_N, 1) (stage BR): b + W = sigma^2 J_vv <= 1 for every state (Lemma 20 (i)) and W <= W_N, so b + W <= min(b + W_N, 1); every place where Lemma
20 uses b + W_N needs an upper bound of b + W (L_1^2 = sigma/(c theta kappa_w (b + W)), gamma, R), so the minimum may replace it.
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from math import erfc, sqrt, pi

QS, RS = 0.25, 0.125
G1 = np.sqrt(2 * RS / (1 + 2 * RS)); RSH = 2 * RS / (1 + 2 * RS); GS = np.sqrt(RS / (1 + RS))
CST = 0.134 / 0.9; SIG = 1e-24
NINF = 4000                                                       # harmonics kept explicitly for N = None (the rest by a tail bound)


def fsup(ua, ub):
    f = lambda u: np.abs(1 - u) * np.exp(-u / 2)
    return np.maximum(np.maximum(f(ua), f(ub)), np.where((ua <= 3.0) & (3.0 <= ub), f(3.0), 0.0))


def greedy(c, cap, budget):
    """max sum c_n z_n with 0 <= z_n <= cap_n, sum z_n <= budget (c >= 0)."""
    o = np.argsort(-c); cs = np.cumsum(cap[o]); full = cs <= budget
    val = float(np.sum((c[o] * cap[o])[full]))
    k = int(np.sum(full))
    if k < len(o): val += float(c[o][k] * (budget - (cs[k - 1] if k > 0 else 0.0)))
    return val


def delta_sup(tau):
    if tau <= 0: return np.inf
    s = 2 * sum(np.exp(-n * n * tau / 2) for n in range(1, 60))
    return s + 2 * sqrt(pi / (2 * tau)) * erfc(59 * sqrt(tau / 2))


def delta_toeplitz(N, tau):
    e = lambda k: np.exp(-k * k * tau / 2)
    return 2 * sum(e(k) for k in range(1, N)) + sum(e(s) for s in range(3, 2 * N))


def sums(N, xa, ta, tb, ny=24, dec_one=False):
    """suprema over y of the harmonic sums for the cell (x >= xa, tau in [ta, tb])."""
    n = np.arange(2, (N if N is not None else NINF) + 1, dtype=float)
    dec = np.ones_like(n) if dec_one else np.exp(-(n * n - 1) * xa / 2)
    en = np.exp(-n * n * ta / 2); fn = fsup(n * n * ta, n * n * tb)
    e1 = np.exp(-ta / 2); f1 = float(fsup(np.array([ta]), np.array([tb]))[0])
    m = min(QS, 2 * np.exp(-xa / 2)) if not dec_one else QS
    ys = np.linspace(0.0, m, ny + 1)
    out = {'H0': 0.0, 'H1': 0.0, 'H2': 0.0, 'N1': 0.0, 'N2': 0.0, 'S2': 0.0}
    for ya, yb in zip(ys[:-1], ys[1:]):
        cap = yb * dec; B = QS - ya
        Gn2 = greedy(n * n, cap, B)
        v = {'H0': yb * e1 + greedy(en, cap, B), 'H1': yb * f1 + greedy(fn, cap, B), 'H2': yb * e1 + greedy(n * n * en, cap, B),
             'N1': yb + greedy(n, cap, B), 'N2': yb + Gn2, 'S2': yb * yb + yb * Gn2}
        for k in out: out[k] = max(out[k], v[k])
    if N is None:                                                  # harmonics beyond NINF: |q0_n| <= y dec_n with dec_n <= e^{-(NINF^2 - 1) x/2}
        assert dec_one is False and xa * NINF * NINF > 2000, 'N = None needs x large enough for the tail bound'
    out['m'] = m
    return out


def cellG(N, xa, xb, ba, bb, c=CST, sig=SIG, dec_one=False, min1=True, small_tau=True, ny=24):
    th = 1 / (1 - c * sig); ta, tb = xa / bb, xb / ba
    h = sums(N, xa, ta, tb, ny, dec_one)
    H0, H1, H2, N1, N2, S2 = h['H0'], h['H1'], h['H2'], h['N1'], h['N2'], h['S2']
    delta = delta_sup(ta) if N is None else min(delta_sup(ta), delta_toeplitz(N, ta))
    P = (1 + delta) * S2 / 2
    if small_tau: P = min(P, tb * N2 * N2)
    kw = (1 + H0 * G1 * RSH) / (1 - H0 * G1); ks = (1 + H1) / (1 - H0)
    WN = xb * (GS * P / (1 - QS) + G1 * H2) / (1 - G1 * H0)
    bW = min(bb + WN, 1.0) if min1 else bb + WN
    L12 = sig / (c * th * kw * bW); gam2 = (1 - c * th * kw * bW * sig) ** 2 / 2
    Psi1 = -np.expm1(-gam2 * L12 / 2)
    Np = N1 ** 2 / (1 - QS) ** 2; tauN = tb * Np
    d = sig * np.sqrt(xb) * N1 / (1 - QS) + sig * QS * np.exp(-0.5) / (1 - QS)
    Ea = (1 + QS) / np.sqrt(2) * (sig * np.sqrt(2 * bb) * np.sqrt((1 + tauN) * (6 + 2 * tauN)) + 2 * bb * d * (1 + tauN))
    Eb = (1 + QS) * (sig * sig * (3 + tauN) + bb * d * d * (1 + tauN))
    main_fac = ba * (1 - H0 * G1) / ks
    rhs = 2 * c * sig * th * bW * (1 + G1 * H0)
    gL1 = np.sqrt(gam2 * L12)
    f_L1 = main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb - rhs
    inc = bool(main_fac * gL1 * np.exp(-gam2 * L12 / 2) >= Ea)
    f_big = main_fac * (-np.expm1(-0.71 ** 2 / 2)) - Ea * 0.71 - Eb - rhs
    ok = bool(f_big > 0 and (gL1 >= 0.71 or (f_L1 > 0 and inc)) and H0 * G1 < 1 and H0 < 1)
    return {'ok': ok, 'margin': float((main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb) / rhs), 'f_big': float(f_big), 'f_L1': float(f_L1), 'gL1': float(gL1),
            'increasing': inc, 'H0': H0, 'H1': H1, 'H2': H2, 'N1': N1, 'N2': N2, 'S2': S2, 'P': P, 'WN': WN, 'bW': bW, 'kw': kw, 'ks': ks,
            'main_fac': main_fac, 'rhs': rhs, 'Ea': Ea, 'Eb': Eb}
