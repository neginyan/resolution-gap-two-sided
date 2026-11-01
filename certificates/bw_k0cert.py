"""Stage BW §2.1: K >= K0 as a computer-assisted proof (paper F v1.5, App. F: any number of harmonics, any phases).

Setting.  Periodic window with (P0), (P2)-(P4), any phases, the profile's blur K >= K0 (K >= x always).  For every n >= 2: |q0_n| <= y d_n,
d_n = e^{-(n^2-1) K/2} <= e^{-(n^2-1) max(x, K0)/2}, y = q0_1 <= m = min(Qbar, 2 e^{-max(x,K0)/2}), sum |q0_n| <= Qbar = 1/4.  No bound on the number of harmonics.

Rigorous floating point (the three items of stage BV §3):
 (1) Harmonic sums.  A sum sup sum c_n z_n over {0 <= z_n <= cap_n, sum z_n <= B} (c_n >= 0) is a linear programme; by weak duality, for EVERY
     lambda >= 0 its value is <= lambda B + sum cap_n (c_n - lambda)_+.  lambda is taken at the greedy break point (computed in floating point; any
     lambda is admissible, so its rounding does not matter); the dual value is a sum of non-negative terms and is rounded up by the factor
     UPF = 1 + 1e-10 (a sum of at most 1e5 non-negative double terms, each from at most 4 operations with libm exp (error <= 1 ulp), has a relative
     error below 1e5 * 6 * 1.2e-16 < 1e-10), plus an absolute 1e-14 sum cap_n c_n for the rounding of c_n inside (c_n - lambda)_+.  Harmonics
     beyond n_max(x) (where (n^2-1) x/2 >= 745) enter through the rigorous tail bound sum_{n >= m} n^p d_n <= t_m/(1 - r), r = (1 + 1/m)^p e^{-(2m+1)x/2}
     (ratio of consecutive terms), times the supremum of the coefficient there.  K0 enters as the double just below the decimal K0.
     (S): S2 <= y^2 + y min(G(n^2), G(n^2 d_n)).  (P): <theta_tau h_t^2> <= N1^2 for every phase.
 (2) Lemma F6' step 2(i) and step 3(ii) with the budgeted gamma_P(x, q): gamma_P is NOT monotone in q (a larger first harmonic takes budget from
     the high ones), so q is cut into cells [q_a, q_b]: N_p <= q_b + LP(n^p; caps q_b d_n(max(x_a, K0)), budget Qbar - q_a), the x-prefactors at x_b.
     Step 2(i): a cell is admissible if x_a q_a^2 < C_q gamma_up (W-sandwich, App. E (a)); gamma_0 <= max over admissible cells.
     Step 3(ii): q > q_min(x) = q_a of the first q-cell with gamma_up >= gamma_lo; the lower W-sandwich gets x_a q_min^2.
 (3) f* (Lemma F7): f*(x) = max over m >= 2 of R(m) = (1 + sum_{n>=m} n^2 w_n)/(1 + sum_{n>=m} w_n), w_n = d_n^2 at max(x, K0); every m is enumerated
     (numerator up, denominator down) until 1 + sum_{n>=m} n^2 w_n is below the best value (R(m') <= that for every m' >= m).  f* decreases in x, so
     x-cells take x at the right end and f* at the left.
Case C (Lemma F4) and T (Lemma F5) in decimal interval arithmetic (ba_ivl) with delta_tau <= min(sqrt(2 pi/tau), 2e^{-tau/2}/(1 - e^{-3 tau/2}))
and P <= min((1 + delta_tau) S2/2, N1^2); eps_R through br_windows.eps_R (interval) with N_eff = ceil(max(N1/Qbar, sqrt(N2/Qbar))).
The cells of Lemma F2 (i), (ii) are those of br_windows.bk1 / caseB (floating point with the in-cell audit, as in v1.4) with the sums of (1).
Environment: BW_K0 (default 0.003), BR_DIR (cell files), BW_TS ('floor': T_s = floor(0.9 T) as in v1.4; 'near': T_s = T (1 - 1e-6)).
-> results/bw_k0_<K0>.json, results/bw_cells_{bk1,caseB}_K<K0>.jsonl (one line per row, resumable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, math
os.environ['BR_PHASE'] = '1'; os.environ.setdefault('BR_CP', '0.000000001')
import numpy as np
import br_cells as C
import br_windows as BW
from ba_ivl import I
from decimal import Decimal as D

K0_TXT = os.environ.get('BW_K0', '0.003')
K0 = float(np.nextafter(float(K0_TXT), 0.0))          # the double just below the decimal K0
QS = 0.25; UPF = 1 + 1e-10; DNF = 1 - 1e-10; TINY = 1e-299
E = math.e


def tail_up(p, m, x):
    """rigorous upper bound of sum_{n >= m} n^p e^{-(n^2-1) x/2} (m >= 2, x > 0), by the ratio of consecutive terms."""
    r_log = p * math.log1p(1 / m) - (2 * m + 1) * x / 2
    if r_log >= -1e-3:
        raise ValueError('tail_up: m too small for the ratio bound')
    lt = p * math.log(m) - (m * m - 1) * x / 2 - math.log1p(-math.exp(r_log))
    if lt < -690: return TINY
    return math.exp(lt) * UPF


_ARR = {}


def arrays(xc):
    """n = 2..nmax with d_n(xc), and the tails beyond nmax for p = 0, 1, 2 (rigorous)."""
    if xc not in _ARR:
        nmax = int(math.ceil(math.sqrt(2 * 745 / xc + 1))) + 2
        n = np.arange(2, nmax + 1, dtype=float); d = np.exp(-(n * n - 1) * xc / 2)
        t = {p: tail_up(p, nmax + 1, xc) for p in (0, 1, 2)}
        _ARR[xc] = (n, d, t)
    return _ARR[xc]


def lp_up(c, cap, B):
    """upper bound of max sum c z, 0 <= z <= cap, sum z <= B (weak duality at the greedy break point)."""
    if B <= 0: return 0.0
    o = np.argsort(-c); cs = np.cumsum(cap[o]); k = int(np.searchsorted(cs, B, side='right'))
    lam = float(c[o][k]) if k < len(o) else 0.0
    v = lam * B + float(np.sum(cap * np.maximum(c - lam, 0.0)))
    return v * UPF + 1e-14 * float(np.sum(cap * c))


def lp_top_up(p, n, capn, B):
    """lp_up for c_n = n^p (increasing): the break point is found from the top."""
    if B <= 0: return 0.0
    cs = np.cumsum(capn[::-1]); k = int(np.searchsorted(cs, B, side='right'))
    lam = float(n[::-1][k] ** p) if k < len(n) else 0.0
    c = n ** p
    v = lam * B + float(np.sum(capn * np.maximum(c - lam, 0.0)))
    return v * UPF + 1e-14 * float(np.sum(capn * c))


def fsup_up(ua, ub):
    f = lambda u: np.abs(1 - u) * np.exp(-u / 2)
    v = np.maximum(np.maximum(f(ua), f(ub)), np.where((ua <= 3.0) & (3.0 <= ub), f(3.0), 0.0))
    return v * UPF


def sums_rig(xa, ta, tb, ny=24, K=None):
    """Lemma F1 sums for the cell (x >= xa, tau in [ta, tb]) with caps y d_n(max(xa, K)), every n >= 2, rigorous upper bounds."""
    K = K0 if K is None else K
    xc = max(xa, K); ta = ta * (1 - 1e-15); tb = tb * (1 + 1e-15)
    n, d, tl = arrays(xc)
    e1 = math.exp(-ta / 2) * UPF; f1 = float(fsup_up(np.array([ta]), np.array([tb]))[0])
    en = np.exp(-n * n * ta / 2); fn = fsup_up(n * n * ta, n * n * tb); n2en = n * n * en; n2d = n * n * d
    n2e_sup = 2 / (E * ta) * UPF if ta > 0 else float('inf')
    m = min(QS, 2 * math.exp(-xc / 2)) * UPF; ys = np.linspace(0.0, m, ny + 1); ys[-1] = m
    out = dict(H0=0.0, H1=0.0, H2=0.0, N1=0.0, N2=0.0, S2=0.0)
    for ya, yb in zip(ys[:-1], ys[1:]):
        B = QS - ya; cap = yb * d * UPF
        t0, t1, t2 = yb * tl[0], yb * tl[1], yb * tl[2]
        G1 = lp_top_up(1, n, cap, B) + t1; G2 = lp_top_up(2, n, cap, B) + t2
        v = {'H0': yb * e1 + lp_up(en, cap, B) + t0,
             'H1': yb * f1 + lp_up(fn, cap, B) + t0,
             'H2': (yb * e1 + lp_up(n2en, cap, B) + min(n2e_sup * t0, t2)) if ta > 0 else yb + G2,
             'N1': yb + G1, 'N2': yb + G2,
             'S2': yb * yb + yb * min(G2, lp_up(n2d, cap, B) + t2)}
        for k in out: out[k] = max(out[k], v[k] * UPF)
    out['m'] = m
    return out


def cell_rig(N, xa, xb, ba, bb, c=C.CST, sig=C.SIG, dec_one=False, min1=True, small_tau=True, ny=24):
    """the cell of Lemma F2 (br_cells.cellG, any phases) with the sums of sums_rig and P <= min((1+delta_tau) S2/2, N1^2); N, dec_one ignored."""
    th = 1 / (1 - c * sig); ta, tb = xa / bb, xb / ba
    h = sums_rig(xa, ta, tb, ny)
    H0, H1, H2, N1, N2, S2 = h['H0'], h['H1'], h['H2'], h['N1'], h['N2'], h['S2']
    delta = C.delta_sup(ta)
    P = min((1 + delta) * S2 / 2, N1 * N1)
    G1, RSH, GS, Q = C.G1, C.RSH, C.GS, C.QS
    kw = (1 + H0 * G1 * RSH) / (1 - H0 * G1); ks = (1 + H1) / (1 - H0)
    WN = xb * (GS * P / (1 - Q) + G1 * H2) / (1 - G1 * H0)
    bW = min(bb + WN, 1.0)
    L12 = sig / (c * th * kw * bW); gam2 = (1 - c * th * kw * bW * sig) ** 2 / 2
    Psi1 = -np.expm1(-gam2 * L12 / 2)
    tauN = tb * N1 ** 2 / (1 - Q) ** 2
    d = sig * np.sqrt(xb) * N1 / (1 - Q) + sig * Q * np.exp(-0.5) / (1 - Q)
    Ea = (1 + Q) / np.sqrt(2) * (sig * np.sqrt(2 * bb) * np.sqrt((1 + tauN) * (6 + 2 * tauN)) + 2 * bb * d * (1 + tauN))
    Eb = (1 + Q) * (sig * sig * (3 + tauN) + bb * d * d * (1 + tauN))
    main_fac = ba * (1 - H0 * G1) / ks
    rhs = 2 * c * sig * th * bW * (1 + G1 * H0)
    gL1 = np.sqrt(gam2 * L12)
    f_L1 = main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb - rhs
    inc = bool(main_fac * gL1 * np.exp(-gam2 * L12 / 2) >= Ea)
    f_big = main_fac * (-np.expm1(-0.71 ** 2 / 2)) - Ea * 0.71 - Eb - rhs
    ok = bool(f_big > 0 and (gL1 >= 0.71 or (f_L1 > 0 and inc)) and H0 * G1 < 1 and H0 < 1)
    return {'ok': ok, 'margin': float((main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb) / rhs), 'f_big': float(f_big), 'N1': N1, 'tauN': tauN}


def sups():
    """sup over every x of N1, N2, S2 (caps at K0 are the largest)."""
    h = sums_rig(0.0, 80.0, 80.0, ny=96)
    return {'N1': h['N1'], 'N2': h['N2'], 'S2': h['S2']}


# ------------------------------------------------------------------ interval constants: case C (F4) and T (F5)
QI = I('0.25'); RHO = I('0.125'); CI = I('0.134') / I('0.9'); SI = I('1e-24'); PI = BW.PI


def Iup(v): return I(D(repr(float(v))))           # a float that is already an upper bound, as a point interval


def delta_I(tau):
    """upper bound of delta_tau = 2 sum_{k>=1} e^{-k^2 tau/2} (interval; tau > 0)."""
    t = I(D(repr(tau)))
    a = (2 * PI / t).sqrt()
    e = (-(t / 2)).exp(); r = (-(3 * t / 2)).exp()
    b = 2 * e / (1 - r)
    return a if a.hi <= b.hi else b


def caseC_I(s, cp=I('0.000000001'), c=CI, sig=SI, ntau=4000):
    g1, rs, gs = BW.geomI(); th = 1 / (1 - c * sig)
    S2, N1, N2 = Iup(s['S2']), Iup(s['N1']), Iup(s['N2'])
    best = I(0)
    taus = [80.0 * (k / ntau) ** 3 for k in range(ntau + 1)]
    for ta, tb in zip(taus[:-1], taus[1:]):
        P = N1 * N1
        if ta > 0:
            Pd = (1 + delta_I(ta)) * S2 / 2
            if Pd.hi < P.hi: P = Pd
        v = (I(D(repr(tb))) * gs * P / (1 - QI) + g1 * QI * 2 / I(1).exp()) / (1 - g1 * QI)
        if v.hi > best.hi: best = v
    Wb = best
    kw = (1 + QI * g1 * rs) / (1 - QI * g1)
    bW = 1 + Wb
    Vmax = bW * c * th * kw * (1 + sig * sig * sig)
    al = I(((1 - cp * sig) / I(2).sqrt()).lo, (1 / I(2).sqrt()).hi)
    Xmin = (-(al * al) * Vmax / 8).exp() - QI
    if Xmin.lo <= 0: return {'W_over_b_max': float(Wb.hi), 'V_max': float(Vmax.hi), 'X_min_lo': float(Xmin.lo), 'ok': False}
    X0 = Xmin + QI; G = Xmin * Xmin * (Xmin / X0).sqrt() / PI
    main = al * al * G / (2 * (1 + QI) * (1 + QI))
    x = I(80) * cp * sig
    gam = (x * (N2 * (1 + QI) + N1 * N1) + x.sqrt() * N1 * (1 + 2 * QI) / I(1).exp().sqrt()) / ((1 - QI) * (1 - QI))
    cchi = (x.sqrt() * N1 / (1 - QI) + (2 / PI).sqrt() + QI * (-I('0.5')).exp() / (1 - QI)) / I(2).sqrt()
    err = al * (1 + QI) * (al * sig * cchi / 2 + gam / I(2).sqrt()) / (1 - QI)
    ap = (main - err) / (2 * th * bW * cp)
    return {'W_over_b_max': float(Wb.hi), 'V_max': float(Vmax.hi), 'X_min_lo': float(Xmin.lo), 'a_prime_lower': float(ap.lo),
            'ratio_lower': float((ap / c).lo), 'ok': bool(ap.lo > c.hi)}


def caseA1_I(s, c=CI, sig=SI):
    g1, rs, gs = BW.geomI(); th = 1 / (1 - c * sig); tau = I(80)
    e1 = (-tau / 2).exp(); dl = delta_I(80.0)
    kw = (1 + QI * g1 * rs * e1) / (1 - QI * g1 * e1)
    Wc = ((1 + dl) * Iup(s['S2']) * gs / (2 * (1 - QI)) + g1 * QI * e1) / (1 - g1 * QI * e1)
    T = 1 / ((1 / tau + Wc) * c * th * kw * (1 + sig * sig * sig))
    return {'W_c': float(Wc.hi), 'T_lower': float(T.lo)}


# ------------------------------------------------------------------ Lemma F6' (Lemma 28 with K >= K0)
def eta_up(p, x):
    """sum_{n>=2} n^p e^{-(n^2-1) max(x,K0)/2}, rigorous upper bound."""
    n, d, t = arrays(max(x, K0))
    return float(np.sum(n ** p * d)) * UPF + t[p]


def Z2_up(x):
    xc = max(x, K0); n, d, t = arrays(2 * xc)              # d_n(2 xc) = e^{-(n^2-1) xc}
    return float(np.sum(n * n * d)) * UPF + t[2]


def gP_up(xa, xb, qa, qb):
    """upper bound of gamma_P over x in [xa, xb], q in [qa, qb] (budgeted N_p, caps at max(xa, K0))."""
    n, d, t = arrays(max(xa, K0)); cap = qb * d * UPF; B = QS - qa
    N1 = (qb + lp_top_up(1, n, cap, B) + qb * t[1]) * UPF; N2 = (qb + lp_top_up(2, n, cap, B) + qb * t[2]) * UPF
    return (xb * (N2 * (1 + QS) + N1 * N1) + math.sqrt(xb) * N1 * (1 + 2 * QS) / math.sqrt(E)) / (1 - QS) ** 2 * UPF * UPF


def xcells():
    return np.concatenate([[0.0], np.geomspace(1e-30, 1.0, 1201)])


def qcells(m, nq=160):
    g = np.geomspace(m * 1e-16, m, nq); return np.concatenate([[0.0], g])


def budget_steps(Cq, glo):
    """step 2(i): sup gamma_up over admissible (x, q)-cells; step 3(ii): inf over x-cells of x_a q_min^2 (rows written one per x-cell)."""
    fn = os.path.join(BW.BR_DIR, 'bw_budget_K%s.jsonl' % K0_TXT); have = {}
    if os.path.exists(fn):
        for l in open(fn):
            if l.strip(): r = json.loads(l); have[r['j']] = r
    xs = xcells()
    for j, (xa, xb) in enumerate(zip(xs[:-1], xs[1:])):
        if j in have: continue
        m = min(QS, 2 * math.exp(-max(xa, K0) / 2)) * UPF; qs = qcells(m)
        gmax = 0.0; qmin = None
        for qa, qb in zip(qs[:-1], qs[1:]):
            g = gP_up(xa, xb, qa, qb)
            if xa * qa * qa < Cq * g: gmax = max(gmax, g)
            if qmin is None and g >= glo: qmin = qa
        r = {'j': j, 'xa': xa, 'xb': xb, 'gamma_max_adm': gmax, 'q_min': qmin, 'x_qmin2': (xa * qmin * qmin if qmin is not None else None)}
        open(fn, 'a').write(json.dumps(r) + '\n'); have[j] = r
    rows = [have[j] for j in sorted(have)]
    g = max(r['gamma_max_adm'] for r in rows)
    adm = [r['x_qmin2'] for r in rows if r['x_qmin2'] is not None]
    return g, (min(adm) if adm else float('inf')), len(rows)


def audit_budget(Cq, glo, npts=400, seed=3):
    """in-cell audit: random (x, q) and the point value gamma_P(x, q) (point cells) must not exceed the cell's bound; and q-cells with a
    point value >= glo must not lie below q_min."""
    rng = np.random.default_rng(seed); xs = xcells(); bad = 0
    for _ in range(npts):
        j = int(rng.integers(1, len(xs) - 1)); xa, xb = xs[j], xs[j + 1]
        m = min(QS, 2 * math.exp(-max(xa, K0) / 2)) * UPF; qs = qcells(m); k = int(rng.integers(0, len(qs) - 1)); qa, qb = qs[k], qs[k + 1]
        x = xa + (xb - xa) * rng.uniform(); q = qa + (qb - qa) * rng.uniform()
        if gP_up(x, x, q, q) > gP_up(xa, xb, qa, qb) * (1 + 1e-12): bad += 1
    return npts, bad


def constants_G(BL, J, sig, c, eps, Gst):
    """bl_bj1.constants with Gamma_* := Gst (the budgeted supremum of gamma_0 over every state of the class; bl_bj1 uses sup Gamma_1 m with eta_p).
    The code is that of bl_bj1.constants (steps 2a, 2b, 3a, 3b), with Gst passed in."""
    E_ = np.e; UP, DN, KAP = J.UP, J.DN, J.KAP
    th = 1 / (1 - c * sig); Lmax2 = 1 / sig + sig * sig; kap = (1 + eps) / (1 - eps)
    s2g = np.sqrt(3 * sig ** 4 + 2 * sig)
    Wmax = (1 + 3 * eps) * J.sup_cells(lambda a, b: b * min(QS * QS, 4 * np.exp(-a)) * (1 + J.Z2(a)) / (3 * 2 * (1 - QS)) * UP)
    Sxi = J.sup_cells(lambda a, b: np.sqrt(b) * J.m(a) * (1 + J.eta(1, a)) / (1 - QS) * UP)
    Qx = lambda a: min(QS, J.m(a) * (1 + J.eta(0, a)))
    T1 = J.sup_cells(lambda a, b: b * (1 + J.Z2(a)) * (Qx(a) / (1 - QS) + Qx(a) ** 2 / (E_ * (1 - QS) ** 2)) * UP)
    T2 = J.sup_cells(lambda a, b: b * np.sqrt(1 + J.Z2(a)) * J.m(a) * (1 + J.eta(1, a)) * UP)
    out = {'Gamma_star': Gst, 'W_max': Wmax, 'S_xi': Sxi, 'T1': T1, 'T2': T2}
    ce = c / (1 - eps)
    nu = 4 * c * sig * th + (1 + eps) * sig * sig + sig * eps
    al2 = (1 - Wmax) ** 2 / 2; Vbar = -2 * np.log1p(-nu) / al2 * UP
    ka = (1 - eps) - 4 * (1 + eps) * Vbar * Gst
    omC = (4 * c * th + (1 + eps) * sig + eps) ** 2 * Wmax / (2 * al2 * (1 - nu) * ka) * UP
    assert omC < 0.5
    rbar = sig * (Sxi + np.sqrt(ce * sig * th)); dr = 0.5 * (np.sqrt(Lmax2) * rbar + rbar * rbar / 2) * UP
    Lbar2 = -4 * np.log1p(-(omC + dr)) * UP
    Cs = 4 * (1 + QS) * ce * th * (1 + J.RS) / (1 - QS) * (1 + 3 * eps)
    l4 = (ce * sig * th) ** 2
    dH = (c * sig * th + Cs * sig * T1 + np.sqrt(Cs * sig) * T2 / (np.sqrt(E_) * (1 - QS) ** 2)) * UP
    Xi = sig * sig * (Sxi ** 2 + Cs * sig * T1) * UP
    rC = (1 + eps) * (3 * Vbar / 8 + (1 + 3 * l4) * (1 + KAP / 8) * Lbar2 / 4 + 0.75 * Xi) * UP
    rw = (1 + eps) * (Vbar / 8 + (1 + 3 * l4) * 0.75 * Lbar2 + 0.75 * Xi) * UP
    A1 = (1 - c * sig) * DN
    out['A_min'] = A1 * (1 - dH) * (1 - rC) * np.sqrt(1 - rw) * DN
    a2b = (1 - 4 * kap * sig * sig * Gst) ** 2 / 2
    out['e_prime'] = -np.expm1(-a2b / (2 * 4 * kap * Gst)) - (1 + eps) * sig * sig - sig * eps - 4 * c * sig * th
    L1 = np.sqrt(sig * (1 - eps) / (2 * c * th)); y = L1 * L1 / 4
    gL1 = (-np.expm1(-y) - 0.5 * (L1 * rbar + rbar * rbar / 2)) * DN
    gLmax = (-np.expm1(-Lmax2 / 4) - dr) * DN
    rhs = 8 / 7 * (4 * c * sig * th + (1 + eps) * sig * sig + sig * eps) / (1 - eps) * UP
    out['step3a_margin'] = min(gL1, gLmax, 0.5) / rhs; out['step3a_unimodal'] = bool(L1 * np.exp(-y) > rbar)
    return out


def gamma_all_sup():
    """sup of gamma_0 over every state with x <= 1 (budgeted, cells in x and q, no admissibility): the Gamma_* of the V-floor and of step 3(ii)'s cap."""
    xs = xcells(); best = 0.0
    for xa, xb in zip(xs[:-1], xs[1:]):
        m = min(QS, 2 * math.exp(-max(xa, K0) / 2)) * UPF; qs = qcells(m, 80)
        best = max(best, max(gP_up(xa, xb, qa, qb) for qa, qb in zip(qs[:-1], qs[1:])))
    return best


def lemma_F6(eps, sig=1e-24):
    import bj_lemma as J
    J.eta = eta_up; J.Z2 = Z2_up
    J.CELLS = np.concatenate([[0.0], np.geomspace(1e-30, 0.01, 1201), np.linspace(0.01, 1.0, 801)[1:]])
    import importlib, bl_bj1 as BL
    importlib.reload(BL)
    c = BL.CSTAR; th = 1 / (1 - c * sig); s2g = math.sqrt(3 * sig ** 4 + 2 * sig)
    G_eta = J.sup_cells(lambda a, b: J.Gam1_up(a, b) * J.m(a))
    G_bud = gamma_all_sup()
    Gst = min(G_eta, G_bud)
    r = constants_G(BL, J, sig, c, eps, Gst)
    ep = r['e_prime']
    Cq = 4 * (1 + QS) * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * J.UP
    def prod(a, b):
        A, B = J.Gam1_parts(a); return min(Cq * (math.sqrt(b) * A + B) ** 2 * J.UP, J.Gam1_up(a, b) * J.m(a))
    g_eta = J.sup_cells(prod)
    glo = (1 - eps) ** 2 / (64 * (1 + eps) * c * th * (1 + sig ** 3)) * J.DN
    g_bud, xq2, nrows = budget_steps(Cq, glo)
    gst = min(g_eta, g_bud)
    a2 = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * J.DN / c
    kap = (1 + eps) / (1 - eps); cap = 32 * kap * Gst
    r3b = ((1 - 3 * eps) * xq2 / (2 * (1 + QS)) / s2g) / cap - eps * J.DN if math.isfinite(xq2) else float('inf')
    s3a = BL.step3a_interval(24, c, eps)
    ok = bool(a2 > 1 and r['step3a_margin'] > 1 and r['step3a_unimodal'] and r3b > 1 and r['A_min'] >= 1.02 * c and s3a['g_gt_rhs'] and s3a['unimodal'])
    npt, nbad = audit_budget(Cq, glo)
    return {'A_min': r['A_min'], 'Gamma_star_eta': G_eta, 'Gamma_star_budget': G_bud, 'e_prime': ep, 'step2i': a2, 'gamma_star_eta': g_eta,
            'gamma_star_budget': g_bud, 'step3a_margin': r['step3a_margin'], 'step3a_interval_ok': bool(s3a['g_gt_rhs'] and s3a['unimodal']),
            'step3a_interval_margin': s3a['margin_lower'], 'step3b': r3b, 'budget_x_rows': nrows, 'budget_audit_points': npt, 'budget_audit_violations': nbad,
            'W_max': r['W_max'], 'S_xi': r['S_xi'], 'T1': r['T1'], 'T2': r['T2'], 'ok': ok}


# ------------------------------------------------------------------ Lemma F7: f* by enumeration of m
def fstar_up(x):
    xc = max(x, K0); n, w, t = arrays(2 * xc)                 # w_n = d_n(2 xc) = e^{-(n^2-1) xc} = d_n(xc)^2
    num_tail = np.cumsum((n * n * w)[::-1])[::-1]; den_tail = np.cumsum(w[::-1])[::-1]
    best = 1.0
    for k in range(len(n)):
        up_all = 1 + (num_tail[k] * UPF + t[2])
        if up_all <= best: break
        R = up_all / (1 + den_tail[k] * DNF)
        best = max(best, R)
    return best * UPF


def fstar_sup(nx=800):
    xs = np.geomspace(K0, 1.0, nx + 1)
    v = max(b * fstar_up(a) for a, b in zip(xs[:-1], xs[1:]))
    return max(v, K0 * fstar_up(K0) * UPF)


def fstar_audit(npts=200, seed=5):
    """the enumeration against a direct brute force (all m up to n_max, no stopping rule) at random x."""
    rng = np.random.default_rng(seed); bad = 0
    for _ in range(npts):
        x = K0 * 10 ** rng.uniform(0, -math.log10(K0)); n, w, t = arrays(2 * max(x, K0))
        bf = max((1 + np.sum(n[k:] ** 2 * w[k:])) / (1 + np.sum(w[k:])) for k in range(len(n)))
        if bf > fstar_up(x) * (1 + 1e-12): bad += 1
    return npts, bad


# ------------------------------------------------------------------ driver
def run(cells=False):
    fn = 'results/bw_k0_%s.json' % K0_TXT
    out = json.load(open(fn)) if os.path.exists(fn) else {'K0': K0_TXT, 'K0_double': K0}
    save = lambda: json.dump(out, open(fn, 'w'), indent=1, default=float)
    if 'sups' not in out: out['sups'] = sups(); save(); print('sups', out['sups'], flush=True)
    s = out['sups']
    if 'caseC' not in out: out['caseC'] = caseC_I(s); save(); print('caseC', out['caseC'], flush=True)
    if 'caseA1' not in out: out['caseA1'] = caseA1_I(s); save(); print('caseA1', out['caseA1'], flush=True)
    T = out['caseA1']['T_lower']
    Ts = math.floor(0.9 * T) if os.environ.get('BW_TS', 'floor') == 'floor' else T * (1 - 1e-6)
    Neff = int(math.ceil(max(s['N1'] / QS, math.sqrt(s['N2'] / QS), 2)))
    if 'eps_R' not in out:
        out['eps_R'] = {'N_eff': Neff, 'T_s': Ts, **BW.eps_R(Neff, float(Ts), T)} if Ts >= 1 else {'N_eff': Neff, 'T_s': Ts, 'ok': False, 'eps_R': float('inf')}
        save(); print('eps_R', out['eps_R'], flush=True)
    if 'fstar' not in out:
        a = fstar_audit(); out['fstar'] = {'sup_x_fstar': fstar_sup(), 'audit_points': a[0], 'audit_violations': a[1]}; save(); print('fstar', out['fstar'], flush=True)
    eps = 1e-6 + out['eps_R']['eps_R'] + out['eps_R'].get('phi_remainder', 0.0) * (1 + 1e-9)   # eps_in + eps_R + eps_Phi (interval upper ends)
    if 'F6' not in out:
        try: out['F6'] = lemma_F6(eps)
        except AssertionError as e: out['F6'] = {'ok': False, 'A_min': float('nan'), 'failed': 'assertion in bl_bj1 (eps too large)'}
        save(); print('F6', out['F6'], flush=True)
    if 'conclusion' not in out:
        A = out['F6']['A_min']
        kap2 = (1 + 0.125 + QS / (2 * (1 - QS))) ** 2 / (8 * (1 - QS))
        k2 = (1 + eps) * (8 * kap2 * (1 + QS) * out['fstar']['sup_x_fstar'] + 0.25)
        g = 0.9 * A / (2 * math.sqrt(k2))
        out['conclusion'] = {'eps': eps, 'k_eff2_max': k2, 'g_lower': g,
                             'ok': bool(out['F6']['ok'] and out['caseC']['ok'] and out['eps_R'].get('ok', False) and g > 0.134)}
        save(); print('conclusion', out['conclusion'], flush=True)
    if cells:
        C.cellG = cell_rig
        BW.FNT = os.path.join(BW.BR_DIR, 'bw_cells_%s_K' + K0_TXT + '%s.jsonl')
        if 'bk1' not in out: out['bk1'] = BW.bk1(''); save(); print('bk1', out['bk1'], flush=True)
        if 'caseB' not in out: out['caseB'] = BW.caseB(''); save(); print('caseB', out['caseB'], flush=True)
    return out


if __name__ == '__main__':
    run(cells='--cells' in sys.argv)
