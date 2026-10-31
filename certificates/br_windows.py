"""Stage BR: windows with at most N harmonics without (P5) (§2.1: N <= 6 down to x -> 0) and the extension to N >= 8 (§2.2).

The proof of Theorem 34 (strips) has four cases.  For a profile with harmonics n <= N (any signs of a_n; (P1)-(P4), (P0), |eps| <= 0.1,
sigma <= 1e-24) each case is redone below; what the strip used and what replaces it:
  - one harmonic, q(w) >= 0        -> |q0_n| <= y = q0_1 ((P2)), sum |q0_n| <= Qbar = 1/4 ((P3)); so 1 - Qbar <= h <= 1 + Qbar and Z in [1 - Qbar, 1 + Qbar]
  - eta_p = Z2 = 0                 -> finite sums over n = 2..N (finite at x = 0)
  - N1 = q, N2 = q, S2 = q^2       -> N1 <= N Qbar, N2 <= N^2 Qbar, S2 = sum n^2 q0_n^2 <= max_y [y^2 + N^2 y (Qbar - y)] = N^4 Qbar^2/(4(N^2 - 1))
  - delta_1 = 0 (theta factor)     -> min(sup theta_tau - 1, Toeplitz delta_N), and P <= tau N2^2 (small tau)
  - X_w >= e^{-alpha^2 V/8} (q>=0) -> X_w >= e^{-alpha^2 V/8} - Qbar
  - W <= x/3 (case C)              -> W/b <= tau [g_s min((1 + delta_N) S2/2, tau N2^2)/(1 - Qbar) + g1 sum n^2|q0_n| e_n]/(1 - g1 Qbar)
Parts:
  caseB(N)    tau <= 80, x < 1e-12, b in [2e-27, 1] (b >= c' sigma):  br_cells.cellG with dec_n := 1 (an upper bound for every x < 1e-12), sigma_hat = 1e-24
              (Lemma 20 (vi)), b-cells x x-cells, in-cell audit.  b < 2e-27: the scaling of Lemma 30 (with dec_n := 1 every ingredient is a function of
              (b, r, tau), and min(b + W_N, 1) = b + W_N there).
  bk1(N)      tau <= 80, 1e-12 <= x <= 1: br_cells.cellG (min(b + W_N, 1), harmonic sums with the budget), in-cell audit.
  caseC(N)    tau <= 80, b < c' sigma: Cauchy-Schwarz on every w-slice, constants in interval arithmetic.
  caseA1(N)   tau > 80, x <= T_s sigma^2: (a) >= c; T from the exact Cramer-Rao and the exact W, interval arithmetic.
  caseA2(N)   tau > 80, T_s sigma^2 < x <= 1: Lemma 28 with N harmonics on x-cells covering [0, 1]; the theta remainder absorbed (eps_R, interval).
  bj1(N, x0)  tau > 80, x0 <= x <= 1 (x0 = 1e-12 for §2.2): Lemma 28 with eps = eps_in + 2 sup E_R.
  conclusion  g >= 0.9 A_min/(2 k_eff), k_eff^2 <= (1 + eps)(8 kappa2 (1 + Qbar) sup_x x f*(x) + 1/4), f*(x) = sup u2/u0 under |q0_n| <= y dec_n.
-> results/br_windows.json, results/br_cells_<part>_N<N>[_cp<c'>].jsonl (one line per b- or x-cell, resumable).
Runs behind the stored results/br_windows.json (BR_OUT: the JSON file, BR_DIR: the folder of the cell files, BR_CP: c'):
  br_windows.py 6 8;  BR_CP=0.0002 br_windows.py 12;  BR_CP=0.00002 br_windows.py 16;  br_windows.py 20 24 --large-x-only;
  BR_CP=0.0000001 br_windows.py 20 --small-x-attempt   (N = 20 below x = 1e-12: cases B, C, A1 close, the last case does not; stored under 'small_x_attempt')
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np
from decimal import Decimal as D
import br_cells as C
from ba_ivl import I

QS = 0.25; CSTAR = 0.134 / 0.9; SIG = 1e-24; EPS = 1e-6
CP_TXT = os.environ.get('BR_CP', '0.002')                      # c' (case B: b >= c' sigma; case C: b < c' sigma); B0 = c' 1e-24
CP = float(CP_TXT); B0 = CP * 1e-24
QI = I('0.25'); RHO = I('0.125'); CI = I('0.134') / I('0.9'); SI = I('1e-24'); CPI = I(CP_TXT)
PI = I('3.14159265358979323846264338327950288419716939937510582097494', '3.14159265358979323846264338327950288419716939937510582097495')
BR_DIR = os.environ.get('BR_DIR', 'results')                  # where the one-line-per-cell files live (a temporary folder for recomputations)
PHASE = C.PHASE                                                 # v1.4 (BR_PHASE=1): sine components allowed, e kept in the theta remainders
FNT = os.path.join(BR_DIR, 'br_cells_%s_N%s' + ('' if CP_TXT == '0.002' else '_cp' + CP_TXT) + ('_ph' if PHASE else '') + '.jsonl')


def f(x): return (float(x.lo), float(x.hi))


def geomI():
    g1 = (2 * RHO / (1 + 2 * RHO)).sqrt(); rs = 2 * RHO / (1 + 2 * RHO); gs = (RHO / (1 + RHO)).sqrt(); return g1, rs, gs


def S2maxI(N): return I(N ** 4) * QI * QI / (4 * I(N * N - 1)) if N > 1 else QI * QI


def load(fn):
    have = {}
    if os.path.exists(fn):
        for line in open(fn):
            if line.strip():
                try: d = json.loads(line); have[d['j']] = d
                except json.JSONDecodeError: pass
    return have


def audit_row(N, cells, dec_one, sig):
    """in-cell audit (stage BP method): the cell's margin and f_big must not exceed those at interior points."""
    nv = 0; npt = 0
    for xa, xb, ba, bb, r in cells:
        for u, v in ((0.5, 0.5), (1e-9, 1 - 1e-9), (1 - 1e-9, 1e-9), (0.3, 0.7), (0.7, 0.3)):
            x = xa + (xb - xa) * u; b = ba + (bb - ba) * v
            if x / b > 80: continue
            p = C.cellG(N, x, x, b, b, sig=sig, dec_one=dec_one); npt += 1
            if p['margin'] < r['margin'] * (1 - 1e-12) or p['f_big'] < r['f_big'] * (1 - 1e-12) - 1e-300: nv += 1
    return npt, nv


# ---------------------------------------------------------------- tau <= 80
def caseB(N, nb_dec=10, nx=40):
    fn = FNT % ('caseB', N); have = load(fn)
    bs = np.geomspace(B0, 1.0, int(round(np.log10(1 / B0) * nb_dec)) + 1)
    for j, (ba, bb) in enumerate(zip(bs[:-1], bs[1:])):
        if j in have: continue
        xhi = min(1e-12, 80 * bb); xs = np.concatenate([[0.0], np.geomspace(xhi * 1e-8, xhi, nx + 1)])
        cells = []; ok = True; wm = np.inf
        for xa, xb in zip(xs[:-1], xs[1:]):
            if xa / bb > 80: continue
            r = C.cellG(N, xa, xb, ba, bb, sig=SIG, dec_one=True); ok &= r['ok']; wm = min(wm, r['margin']); cells.append((xa, xb, ba, bb, r))
        npt, nv = audit_row(N, cells, True, SIG)
        d = {'j': j, 'ba': ba, 'bb': bb, 'x_cells': len(cells), 'ok': bool(ok), 'margin': float(wm), 'audit_points': npt, 'audit_violations': nv}
        with open(fn, 'a') as fh: fh.write(json.dumps(d) + '\n')
        have[j] = d
    rows = [have[j] for j in sorted(have)]
    return {'b_cells': len(rows), 'cells': sum(d['x_cells'] for d in rows), 'all_ok': all(d['ok'] for d in rows), 'worst_margin': min(d['margin'] for d in rows),
            'audit_points': sum(d['audit_points'] for d in rows), 'audit_violations': sum(d['audit_violations'] for d in rows)}


def bk1(N, nx=120, nb=121):
    fn = FNT % ('bk1', N); have = load(fn)
    xs = np.geomspace(1e-12, 1.0, nx + 1)
    for j, (xa, xb) in enumerate(zip(xs[:-1], xs[1:])):
        if j in have: continue
        bs = np.geomspace(xa / 80, 1.0, nb); cells = []; ok = True; wm = np.inf
        for ba, bb in zip(bs[:-1], bs[1:]):
            r = C.cellG(N, xa, xb, ba, bb, sig=SIG); ok &= r['ok']; wm = min(wm, r['margin']); cells.append((xa, xb, ba, bb, r))
        npt, nv = audit_row(N, cells[::4], False, SIG)
        d = {'j': j, 'xa': xa, 'xb': xb, 'b_cells': len(cells), 'ok': bool(ok), 'margin': float(wm), 'audit_points': npt, 'audit_violations': nv}
        with open(fn, 'a') as fh: fh.write(json.dumps(d) + '\n')
        have[j] = d
    rows = [have[j] for j in sorted(have)]
    return {'x_cells': len(rows), 'cells': sum(d['b_cells'] for d in rows), 'all_ok': all(d['ok'] for d in rows), 'worst_margin': min(d['margin'] for d in rows),
            'worst_x': min(rows, key=lambda d: d['margin'])['xa'], 'audit_points': sum(d['audit_points'] for d in rows),
            'audit_violations': sum(d['audit_violations'] for d in rows)}


def W_over_b_sup(N, ncell=4000):
    """sup over tau in (0, 80] of W/b (interval cells in tau): (tau/Z)[g_s min((1 + delta_N(tau)) S2/2, tau N2^2)/(1 - Qbar) + g1 sum n^2 |q0_n| e_n]."""
    g1, rs, gs = geomI(); S2 = S2maxI(N); N2 = I(N * N) * QI
    taus = [D(0)] + [D(80) * (D(k) / D(ncell)) ** 3 for k in range(1, ncell + 1)]
    best = I(0)
    for ta, tb in zip(taus[:-1], taus[1:]):
        TA, TB = I(ta), I(tb)
        dN = I(0)
        for k in range(1, N): dN = dN + 2 * (-(I(k * k) * TA / 2)).exp()
        for s in (range(2, 2 * N + 1) if PHASE else range(3, 2 * N)): dN = dN + (-(I(s * s) * TA / 2)).exp()
        P = (1 + dN) * S2 / 2; P2 = TB * N2 * N2
        Pm = P if (PHASE or P.hi <= P2.hi) else P2
        h2 = I(0)                                                       # sum n^2 |q0_n| e_n <= Qbar max_n n^2 e_n, tau n^2 e_n <= 2/e
        v = TB * gs * Pm / (1 - QI) + g1 * QI * 2 / I(1).exp()
        v = v / (1 - g1 * QI)
        if v.hi > best.hi: best = v
    return best


def caseC(N, c=CI, sig=SI, cp=CPI):
    g1, rs, gs = geomI(); th = 1 / (1 - c * sig)
    kw = (1 + QI * g1 * rs) / (1 - QI * g1)                             # Cramer-Rao for the exact w-marginal, signs allowed
    Wb = W_over_b_sup(N); bW = 1 + Wb
    Vmax = bW * c * th * kw * (1 + sig * sig * sig)
    al = I(((1 - cp * sig) / I(2).sqrt()).lo, (1 / I(2).sqrt()).hi)
    Xmin = (-(al * al) * Vmax / 8).exp() - QI
    assert Xmin.lo > 0
    if PHASE:
        # X_w is complex: |Re(X_w e^{i theta})| >= (X0 |cos theta| - Qbar)_+, theta = w/(2 sqrt2), X0 = e^{-alpha^2 V/8}, d = X0 - Qbar.  With
        # W = sqrt(d/X0) and s = |theta|/W <= 1: X0 cos theta - Qbar >= d - X0 theta^2/2 = d (1 - s^2/2), and (1 - s^2/2)^2 >= 1 - s, so the square
        # is >= d^2 (1 - |theta|/W)_+ (a triangle, pi-periodised).  A triangle's Fourier coefficients are >= 0, hence its Gaussian average over theta
        # is >= its uniform average d^2 W/pi for every L:  int Gamma |Re X_w e^{i theta}|^2 >= G := d^2 sqrt(d/X0)/pi.
        X0 = Xmin + QI; G = Xmin * Xmin * (Xmin / X0).sqrt() / PI
        main = al * al * G / (2 * (1 + QI) * (1 + QI))
    else:
        main = al * al * Xmin * Xmin / (4 * (1 + QI) * (1 + QI))
    x = I(80) * cp * sig; N1 = I(N) * QI; N2 = I(N * N) * QI
    gam = (x * (N2 * (1 + QI) + N1 * N1) + x.sqrt() * N1 * (1 + 2 * QI) / I(1).exp().sqrt()) / ((1 - QI) * (1 - QI))
    cchi = (x.sqrt() * N1 / (1 - QI) + (2 / PI).sqrt() + QI * (-I('0.5')).exp() / (1 - QI)) / I(2).sqrt()
    err = al * (1 + QI) * (al * sig * cchi / 2 + gam / I(2).sqrt()) / (1 - QI)
    two = main - err
    a_prime = two / (2 * th * bW * cp)
    return {'W_over_b_max': f(Wb), 'kappa_w': f(kw), 'V_max': f(Vmax), 'X_min': f(Xmin), 'main': f(main), 'err': f(err), 'a_prime_lower': float(a_prime.lo),
            'ratio_lower': float((a_prime / c).lo), 'ok': bool(a_prime.lo > c.hi)}


# ---------------------------------------------------------------- tau > 80
def caseA1(N, c=CI, sig=SI):
    g1, rs, gs = geomI(); th = 1 / (1 - c * sig); tau = I(80)
    e1 = (-tau / 2).exp()
    dN = I(0)
    for k in range(1, max(N, 2)): dN = dN + 2 * (-(I(k * k) * tau / 2)).exp()
    for s in (range(2, 2 * N + 1) if PHASE else range(3, 2 * N)): dN = dN + (-(I(s * s) * tau / 2)).exp()
    kw = (1 + QI * g1 * rs * e1) / (1 - QI * g1 * e1)
    h2 = QI * e1                                                        # sum n^2 |q0_n| e_n <= Qbar max_n n^2 e^{-40 n^2} = Qbar e^{-40}
    Wc = ((1 + dN) * S2maxI(N) * gs / (2 * (1 - QI)) + g1 * h2) / (1 - g1 * QI * e1)
    bw = 1 / tau + Wc
    T = 1 / (bw * c * th * kw * (1 + sig * sig * sig))
    return {'W_coef': f(Wc), 'b_plus_W_over_x': f(bw), 'T': f(T), 'T_lower': float(T.lo)}


def eps_R(N, Ts, T):
    """E_R <= A0/sqrt(x) + B0 sigma/x + C0 at tau >= 80 with beta <= tau/sqrt(2 Ts) (x > Ts sigma^2); absorbed into the b-coefficient through
    1 - C >= 0.2 min(T sigma/x, 1) (Lemma 32's analogue: (a) < c gives L^2 > T sigma/x)."""
    tau = I(80); beta = tau / (2 * I(Ts)).sqrt()
    def nu(k):
        s = I(0)
        if PHASE:     # Lemma BL3': every beta-dependent remainder carries e = e^{-beta^2/(2 tau)}: e e_n e^{n beta} = exp(-tau (n - r)^2/2), plus the theta_tau part
            r = beta / tau
            for n in range(1, 13): s = s + I(n ** k) * ((-(tau * (I(n) - r) * (I(n) - r)) / 2).exp() + (-(I(n * n) * tau / 2)).exp())
            last = I(13 ** k) * ((-(tau * (I(13) - r) * (I(13) - r)) / 2).exp() + (-(I(169) * tau / 2)).exp())
            return 2 * (s + I(0, (2 * last).hi))
        for n in range(1, 13): s = s + I(n ** k) * (-(I(n * n) * tau / 2) + I(n) * beta).exp()
        last = I(13 ** k) * (-(I(169) * tau / 2) + I(13) * beta).exp()
        return 2 * (s + I(0, (2 * last).hi))
    nv = {k: nu(k) for k in range(5)}
    N1 = I(N) * QI; N2 = I(N * N) * QI
    gmax = (N2 * (1 + QI) + N1 * N1 + N1 * (1 + 2 * QI) / I(1).exp().sqrt()) / ((1 - QI) * (1 - QI))    # gamma_P on x <= 1
    cc = (N1 / (1 - QI) + (2 / PI).sqrt() + QI * (-I('0.5')).exp() / (1 - QI)) / I(2).sqrt()
    r2 = I(2).sqrt(); Q = QI
    A0 = tau * tau * N1 * N1 * nv[1] / (r2 * (1 - Q)) + tau * nv[1] * (1 + Q) * gmax / r2
    if PHASE: A0 = A0 + tau * tau * Q * nv[3] / (2 * r2)                  # a_10: the sinh(n beta) sin(phi_n) term of Theta acting on h'' (sine components)
    B0 = I('0.375') * tau * Q * nv[0] + Q * tau * tau * nv[2] / 8 + tau * nv[0] * (1 + Q) * gmax + Q * tau * tau * tau * nv[4] / 8
    C0 = (1 + Q) * (1 + Q / 2) * cc * nv[0] + tau * nv[1] * N1 * cc + tau * Q * nv[2] * (1 + Q) * cc / 2 + tau * Q * nv[2] / (2 * (1 - nv[0]))
    S = A0 + B0 + C0
    case_i = S / (I('0.2') * I(T)); case_ii = (A0 / I(Ts).sqrt() + B0 / I(Ts) + C0) / I('0.2')
    eR = case_i if case_i.hi >= case_ii.hi else case_ii
    slope = (I(3) / tau - (1 - 1 / (2 * I(Ts)).sqrt()) * (1 - 1 / (2 * I(Ts)).sqrt()) / 2) if PHASE else (I(3) / tau - I('0.5') + 1 / (2 * I(Ts)).sqrt())
    phi_rem = 2 * Q * (nv[0] + tau * nv[2])
    return {'beta_over_tau': float(beta.hi / 80), 'A0': f(A0), 'B0': f(B0), 'C0': f(C0), 'eps_R': float(eR.hi), 'slope': float(slope.hi), 'phi_remainder': float(phi_rem.hi),
            'ok': bool(slope.hi < 0)}


def setJ(N, cells):
    import bj_lemma as J
    nn = np.arange(2, N + 1, dtype=float)
    J.eta = lambda p, x: float(np.sum(nn ** p * np.exp(-(nn * nn - 1) * x / 2)))
    J.Z2 = lambda x: float(np.sum(nn * nn * np.exp(-(nn * nn - 1) * x)))
    J.CELLS = cells
    return J


def bj1(N, x0, eps, sig=SIG):
    """Lemma 28 with N harmonics on x in [x0, 1] (x0 = 0: a first cell [0, 1e-30]) with perturbation eps; step 2(i) with the per-x product."""
    if x0 == 0: cells = np.concatenate([[0.0], np.geomspace(1e-30, 0.01, 6001), np.linspace(0.01, 1.0, 4001)[1:]])
    else: cells = np.concatenate([np.geomspace(x0, 0.01, 3001), np.linspace(0.01, 1.0, 3001)[1:]])
    J = setJ(N, cells)
    import bl_bj1 as BL
    c = BL.CSTAR
    r = BL.constants(sig, c, eps)
    th = 1 / (1 - c * sig); s2g = np.sqrt(3 * sig ** 4 + 2 * sig); ep = r['step2b']['e_prime']
    Cq = 4 * (1 + QS) * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * J.UP
    def prod(a, b):
        A, B = J.Gam1_parts(a); return min(Cq * (np.sqrt(b) * A + B) ** 2 * J.UP, J.Gam1_up(a, b) * J.m(a))
    gst = J.sup_cells(prod)
    aw2b = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * J.DN
    r['step2b'] = {'e_prime': ep, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c)}
    r['step3a_interval'] = BL.step3a_interval(24, c, eps)
    r['steps_ok'] = bool(r['step2b']['contradiction'] and r['step3a']['contradiction'] and r['step3b']['contradiction'] and r['A_min'] >= 1.02 * c
                         and r['step3a_interval']['g_gt_rhs'] and r['step3a_interval']['unimodal'])     # what App. F needs (g >= 0.9 A_min/(2 k_eff))
    r['ok'] = bool(r['steps_ok'] and r['A_min'] > 0.9)                                                    # + the bootstrap-start flag of Lemma 28
    r['x_cells'] = len(cells) - 1
    return r


def E_R_sup(N, x0=1e-12, sig=SIG):
    import bl_theta as T
    import bj_lemma as J
    setJ(N, J.CELLS)
    tau = 80.0; Q = QS; worst = 0.0; xs = np.geomspace(x0, 1.0, 2001)
    for a, b in zip(xs[:-1], xs[1:]):
        nu = T.nus(tau, sig * tau / np.sqrt(2 * a)); N1 = min(0.25 * (1 + J.eta(1, a)), N * Q)
        gam = min(0.25 * J.Gam1_up(a, b), (b * (N * N * Q * (1 + Q) + (N * Q) ** 2) + np.sqrt(b) * N * Q * (1 + 2 * Q) / np.sqrt(np.e)) / (1 - Q) ** 2)
        cc = (np.sqrt(b) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)
        e = (0.375 * sig * tau * Q * nu[0] / a + (1 + Q) * (1 + Q / 2) * cc * nu[0] + Q * sig * tau ** 2 * nu[2] / (8 * a)
             + tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2 * a) * (1 - Q)) + sig * tau * nu[0] * (1 + Q) * gam / a
             + tau * nu[1] * (N1 * cc + (1 + Q) * gam / np.sqrt(2 * a)) + tau * Q * nu[2] * (1 + Q) * cc / 2 + Q * sig * tau ** 3 * nu[4] / (8 * a)
             + sig * tau * Q * nu[2] / (2 * (1 - nu[0])))
        if PHASE: e += tau ** 2 * Q * nu[3] / (2 * np.sqrt(2 * a))                 # a_10 (sine components)
        worst = max(worst, e)
    return worst


def fstar(N, xa):
    """sup of u2/u0 = sum n^2 q0_n^2 / sum q0_n^2 over |q0_n| <= y dec_n(xa) (n = 2..N), q0_1 = y: the weighted mean of n^2 with weights w_n^2,
    w_1 = 1, 0 <= w_n <= dec_n (scale-free in y; the budget only lowers it).  The maximiser takes w_n = dec_n for every n^2 above the value."""
    n = np.arange(2, N + 1, dtype=float); d2 = np.exp(-(n * n - 1) * xa)
    o = np.argsort(-n); best = 1.0; num = 1.0; den = 1.0
    for k in o:
        num += n[k] ** 2 * d2[k]; den += d2[k]; best = max(best, num / den)
    return best


def conclusion(N, A_min, eps, ncell=4000):
    kap2 = (1 + 0.125 + QS / (2 * (1 - QS))) ** 2 / (8 * (1 - QS))
    xs = np.concatenate([[0.0], np.geomspace(1e-14, 1.0, ncell)])
    xf = max(b * fstar(N, a) for a, b in zip(xs[:-1], xs[1:]))
    k2 = (1 + eps) * (8 * kap2 * (1 + QS) * xf + 0.25)
    g = 0.9 * A_min / (2 * np.sqrt(k2))
    return {'sup_x_fstar': xf, 'k_eff2_max': k2, 'g_lower': g, 'ratio_to_0.134': g / 0.134, 'ok': bool(g > 0.134)}


def numeric_caseC(N, a_prime_lower):
    """numerical check (not part of the proof): (a') by direct quadrature (bk_direct) for profiles with N harmonics of mixed signs in case C,
    against the bound (a') >= a_prime_lower c'/(b/sigma)."""
    import bk_direct as Dd
    profs = {2: [[0.15, -0.1], [0.125, 0.125]], 6: [[0.1, -0.05, 0.05, -0.03, 0.02, 0.0], [0.04] * 6, [0.06, 0.0, 0.0, 0.0, 0.0, -0.06]],
             8: [[0.03] * 8, [0.05, 0, 0, 0, 0, 0, 0, -0.05]], 12: [[0.02] * 12, [0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03]],
             16: [[0.015] * 16]}
    out = []
    for q0 in profs.get(N, [[0.25 / N] * N]):
        for b_rel in (CP * 0.5, CP * 0.9):
            for tau in (1e-3, 1.0, 50.0):
                for L2 in (1e-2, 1.0, 30.0):
                    sig = 1e-24; b = b_rel * sig; x = tau * b
                    try: r = Dd.branches(sig, b, x, q0, L2, L2 / 8)
                    except Exception: continue
                    out.append({'q0': q0, 'b_over_sigma': b_rel, 'tau': tau, 'L2': L2, 'a_N': r['a_N'], 'bound': a_prime_lower * CP / b_rel})
    return out


def run(N, small_x):
    out = {'N': N, 'c_prime': CP, 'B0': B0}
    out['bk1_x_ge_1e-12'] = bk1(N); print('N=%d bk1:' % N, out['bk1_x_ge_1e-12'], flush=True)
    ER = E_R_sup(N); epsb = EPS + 2 * ER + 7.1e-14
    rb = bj1(N, 1e-12, epsb); out['bj1_x_ge_1e-12'] = {'E_R_sup': ER, 'eps': epsb, 'A_min': rb['A_min'], 'ok': rb['ok'], 'step3a_margin': rb['step3a_interval']['margin_lower'],
                                                       'step2i': rb['step2b']['a_lower_over_c'], 'step3ii': rb['step3b']['W_lower_over_W_cap']}
    out['conclusion_x_ge_1e-12'] = conclusion(N, rb['A_min'], epsb)
    print('N=%d bj1:' % N, out['bj1_x_ge_1e-12'], out['conclusion_x_ge_1e-12'], flush=True)
    if small_x:
        out['caseB'] = caseB(N); print('N=%d caseB:' % N, out['caseB'], flush=True)
        out['caseC'] = caseC(N); print('N=%d caseC:' % N, out['caseC'], flush=True)
        nc = numeric_caseC(N, out['caseC']['a_prime_lower'])
        out['caseC_numeric'] = {'states': len(nc), 'all_above_bound': all(d['a_N'] >= d['bound'] for d in nc), 'min_ratio': min(d['a_N'] / d['bound'] for d in nc) if nc else None}
        a1 = caseA1(N); Ts = float(np.floor(a1['T_lower'] * 0.9)); a1['T_split'] = Ts; a1['ok'] = bool(Ts >= 1 and a1['T_lower'] > Ts)
        out['caseA1'] = a1
        er = eps_R(N, Ts, a1['T_lower']); out['eps_R'] = er
        eps2 = EPS + er['eps_R'] + er['phi_remainder']
        r2 = bj1(N, 0, eps2)
        out['caseA2'] = {'eps': eps2, 'A_min': r2['A_min'], 'ok': (r2['steps_ok'] if PHASE else r2['ok']) and er['ok'],
                         'step3a_margin': r2['step3a_interval']['margin_lower'],
                         'step2i': r2['step2b']['a_lower_over_c'], 'step3ii': r2['step3b']['W_lower_over_W_cap'], 'x_cells': r2['x_cells']}
        if PHASE: out['caseA2']['A_min_gt_0.9'] = bool(r2['A_min'] > 0.9)        # the bootstrap-start flag (not needed for g >= 0.134)
        out['conclusion_x_to_0'] = conclusion(N, r2['A_min'], eps2)
        print('N=%d A1:' % N, a1, 'eps_R:', er, 'A2:', out['caseA2'], out['conclusion_x_to_0'], flush=True)
    return out


def attempt(N):
    """x < 1e-12 for an N whose last case does not close: cases B, C, A1 and the size of eps_R (no assertion on the last case)."""
    out = {'c_prime': CP}
    out['caseB'] = caseB(N); out['caseC'] = caseC(N)
    a1 = caseA1(N); Ts = float(np.floor(a1['T_lower'] * 0.9)); a1['T_split'] = Ts; out['caseA1'] = a1
    er = eps_R(N, Ts, a1['T_lower']); out['eps_R'] = er
    out['caseA2'] = ('fails: eps_R = %.2f, the theta remainder at tau = 80 with beta <= tau/sqrt(2 T_s) is not small' % er['eps_R'] if er['eps_R'] > 0.5
                     else 'eps_R = %.3g: run the full case' % er['eps_R'])
    print('N=%d attempt:' % N, out, flush=True)
    return out


if __name__ == '__main__':
    np.seterr(all='ignore')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; small = '--large-x-only' not in sys.argv
    Ns = [int(a) for a in args] or [6]
    out = os.environ.get('BR_OUT', 'results/br_windows.json')
    for N in Ns:
        res = json.load(open(out)) if os.path.exists(out) else {}
        if '--small-x-attempt' in sys.argv:
            res.setdefault('N=%d' % N, {})['small_x_attempt'] = attempt(N)
        else:
            r = run(N, small_x=small)
            res = json.load(open(out)) if os.path.exists(out) else {}
            key = 'N=%d' % N + ('' if CP_TXT == '0.002' else ', c\'=' + CP_TXT) + (', phase' if PHASE else '')
            if 'small_x_attempt' in res.get(key, {}): r['small_x_attempt'] = res[key]['small_x_attempt']
            res[key] = r
        json.dump(res, open(out, 'w'), indent=1, default=float)
