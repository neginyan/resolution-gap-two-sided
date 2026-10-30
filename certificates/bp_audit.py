"""Stage BP §1: audit of every computer-assisted certificate for "worst-corner" evaluations — each quantity that a certificate bounds on a cell
by evaluating monotone factors at the cell's ends is compared with its value at points INSIDE the cell (at least 5 per cell, including points
next to both ends).  A violation is a point value on the wrong side of the cell value (relative tolerance 1e-12).  Numerical audit (it can find
defects; it cannot prove their absence), one JSON line per checked quantity, resumable: results/bp_audit.jsonl.

Sections (usage: python bp_audit.py [section ...]; default all):
  xsup    the x-suprema of the lower chain (bf_cert_p / bl_chain): x gamma_P, sqrt(x) N1/(1 - Qbar), x min(1/16, 4e^{-x}), sqrt(x) gamma_P,
          sqrt(x) N1; monotone harmonic sums eta_p, zeta_p, Qbar, q1; the decreasing tails beyond x = 8; S_i(x) and the S1_P cells
  strip   the strip certificate (ay_certificate, used by bf_ay_bootstrap): x gamma, gamma, x K1(q) q^2 (K1 increasing in q)
  bk1     Lemma BK1 (bk_narrow_lemma.cell): every intermediate quantity, the v1.1 code (H1 at tau_min) and the v1.2 code; general windows (tau0 = 80)
          and strips (x < 1).  The v1.1 rows show that the audit finds the defect corrected in v1.2.
  bj1     Lemma BJ1 (bj_lemma / bl_bj1): every x-cell function, the admissible set and the infimum of step 3(ii); the BN strip version (per-x product)
  theta   bl_theta.eps_table suprema; tau-monotonicity of tau^j nu_k for tau >= 80; bn_strip_x.E_R_sup cells
  sigc    Lemma BJ1: every intermediate quantity at random (sigma, c) against the corner (1e-24, c*)  (Table 8)
  upper   the upper certificate (bf_cert_upper): point values inside cells against the cell bounds; the harmonic interval parameters
  members class membership (bg_members, bh_members_fc): point enclosures inside the cell enclosures
  tail    Lemma 7's bound T(l) on a grid 20x finer than the certificate's
  lcells  the lower chain on l-cells (bl_chain, perturbed): every intermediate quantity at random interior (l, eps) against the cell's corner,
          at the first and the last bootstrap step, and the final L(l) >= L_cell   [the longest section]
Reduced audit: BP_AUDIT_QUICK=1 (fewer cells and points, the same quantities; a few minutes); output file: BP_AUDIT_OUT.
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys, time
import numpy as np

FN = os.environ.get('BP_AUDIT_OUT', 'results/bp_audit.jsonl')
RNG = np.random.default_rng(20261029)
TOL = 1e-12
QUICK = os.environ.get('BP_AUDIT_QUICK', '0') == '1'          # reduced audit (verification/run_all.py default): fewer cells, same quantities


def S(full, quick): return quick if QUICK else full


def done():
    out = {}
    if os.path.exists(FN):
        for line in open(FN):
            if line.strip():
                try: d = json.loads(line); out[d['key']] = d
                except json.JSONDecodeError: pass
    return out


def write(d):
    with open(FN, 'a') as f: f.write(json.dumps(d, default=float) + '\n')


def pts(a, b, n=6):
    u = np.concatenate([[1e-9, 1 - 1e-9, 0.5], RNG.random(max(0, n - 3))])
    return a + (b - a) * u


def audit_cells(key, cells, f_cell, f_point, kind='sup', n=6, rel=TOL):
    """f_cell(a, b) bounds f_point(x) on [a, b] from above (sup) or below (inf)."""
    nv = 0; npts = 0; worst = 0.0; ex = None
    for a, b in cells:
        vc = f_cell(a, b)
        for x in pts(a, b, n):
            vp = f_point(x); npts += 1
            bad = (vp > vc * (1 + rel) + 1e-300) if kind == 'sup' else (vp < vc * (1 - rel) - 1e-300)
            r = (vp / vc - 1) if vc != 0 else 0.0
            if bad:
                nv += 1
                if abs(r) > abs(worst): worst = r; ex = [float(a), float(b), float(x), float(vc), float(vp)]
    return {'key': key, 'cells': len(cells), 'points': npts, 'violations': nv, 'worst_rel': worst, 'example': ex}


def mono_points(key, xs, f, direction):
    """f(x) monotone in the declared direction (+1 non-decreasing, -1 non-increasing) on the sorted grid xs."""
    v = np.array([f(x) for x in xs]); d = np.diff(v) * direction
    bad = np.where(d < -np.abs(v[:-1]) * TOL - 1e-300)[0]
    return {'key': key, 'cells': len(xs) - 1, 'points': len(xs), 'violations': int(len(bad)),
            'example': [float(xs[bad[0]]), float(xs[bad[0] + 1]), float(v[bad[0]]), float(v[bad[0] + 1])] if len(bad) else None}


# ------------------------------------------------------------------------------------------------ xsup
def sec_xsup(have):
    import be_cert_p as P, bf_cert_p as B
    gP, N1, Qb, q1, eta, zeta = P.gamma_P, B.N1, P.Qb, P.q1, P.eta, P.zeta
    QB = P.QB
    grids = [np.linspace(1.0, 21.0, 4001), np.linspace(1.0, 400.0, 4001), np.linspace(30.0, 60.0, 4001)]
    funcs = {
        'xsup:x*gamma_P': (lambda a, b: b * gP(a, b)),
        'xsup:sqrt(x)N1/(1-Qbar)': (lambda a, b: np.sqrt(b) * N1(a) / (1 - Qb(a))),
        'xsup:x*min(1/16,4e^-x)': (lambda a, b: b * min(QB ** 2, 4 * np.exp(-a))),
        'xsup:sqrt(x)*gamma_P (omega-term)': (lambda a, b: np.sqrt(b) * gP(a, b)),
        'xsup:sqrt(x)N1': (lambda a, b: np.sqrt(b) * N1(a)),
        'xsup:gamma_P': (lambda a, b: gP(a, b)),
    }
    for name, f in funcs.items():
        for gi, xs in enumerate(grids):
            key = '%s grid%d' % (name, gi)
            if key in have: continue
            st = S(4, 100); cells = list(zip(xs[:-1][::st], xs[1:][::st]))          # every 4th cell of the certificate's grid
            write(audit_cells(key, cells, f, lambda x, f=f: f(x, x)))
        key = name + ' decreasing for x >= 8'
        if key not in have:
            write(mono_points(key, np.geomspace(8.0, 5000.0, S(4001, 401)), lambda x, f=f: f(x, x), -1))
    for name, f in {'eta_p(x) p=0..2': lambda x: sum(eta(p, x) for p in (0, 1, 2)), 'zeta_p(x) p=0,2': lambda x: zeta(0, x) + zeta(2, x),
                    'Qbar(x)': Qb, 'q1(x)': lambda x: float(q1(x)), 'N1(x)': N1}.items():
        key = 'xsup:%s decreasing' % name
        if key not in have: write(mono_points(key, np.geomspace(0.05, 500.0, S(3001, 301)), f, -1))
    # S_i(x) (Wiener majorant) and the S1_P cells (q0 at a, x at b): S_i must decrease in x (it is used through S_i(1) for x >= 1)
    key = 'xsup:S_i(x) decreasing on [1, 60]'
    if key not in have: write(mono_points(key, np.linspace(1.0, 60.0, S(600, 60)), B.Si_at, -1))
    key = 'xsup:S1_P cells x(S_i - 1)'
    if key not in have:
        xs = np.linspace(1.0, 200.0, 2000)
        def S1cell(a, b):
            from be_plemmas import S_branch_i
            q = float(q1(a)); q0 = np.array([q * P.r_m(a, m) for m in range(1, P.MMAX)]); return b * (S_branch_i(q0)[0] - 1)
        st = S(10, 200); write(audit_cells(key, list(zip(xs[:-1][::st], xs[1:][::st])), S1cell, lambda x: x * (B.Si_at(x) - 1), n=5))
    key = 'xsup:x(S_i(x)-1) decreasing beyond 200 (tail of S1_P)'
    if key not in have: write(mono_points(key, np.linspace(150.0, 400.0, S(200, 20)), lambda x: x * (B.Si_at(x) - 1), -1))


# ------------------------------------------------------------------------------------------------ strip
def sec_strip(have):
    import ay_certificate as A
    from aw_lemmas import K1
    xs = np.linspace(1e-9, 60.0, 4001)
    for name, f in (('strip:x*gamma', A.xgamma_up), ('strip:gamma', A.gamma_up)):
        key = name
        if key in have: continue
        st = S(2, 50); write(audit_cells(key, list(zip(xs[:-1][::st], xs[1:][::st])), lambda a, b, f=f: float(f(np.array([a]), np.array([b]))[0]),
                          lambda x, f=f: float(f(np.array([x]), np.array([x]))[0])))
    key = 'strip:x K1(q) q^2 (S1 cells)'
    if key not in have:
        xs = np.linspace(-2 * np.log(0.25), 60.0, 3001)
        st = S(5, 100); write(audit_cells(key, list(zip(xs[:-1][::st], xs[1:][::st])), lambda a, b: float(A.S1_up(np.array([a]), np.array([b]))[0]),
                          lambda x: float(A.S1_up(np.array([x]), np.array([x]))[0]), n=5))
    key = 'strip:K1(q) increasing in q'
    if key not in have: write(mono_points(key, np.linspace(0.001, 0.25, S(500, 50)), lambda q: K1(q, 0.125, 60), +1))


# ------------------------------------------------------------------------------------------------ bk1
BK_DIR = {'H0': 'U', 'H1': 'U', 'H2': 'U', 'delta': 'U', 'Z2': 'U', 'eta1': 'U', 'q': 'U', 'kw': 'U', 'ks': 'U', 'WN': 'U', 'L12': 'L', 'gam2': 'L',
          'Psi1': 'L', 'Np': 'U', 'tauN': 'U', 'd': 'U', 'Ea': 'U', 'Eb': 'U', 'main_fac': 'L', 'rhs': 'U', 'f_L1': 'L', 'f_big': 'L', 'margin': 'L'}


def bk_cell_full(BK, xa, xb, ba, bb, fixed, c=None, sig=None):
    """bk_narrow_lemma.cell with every intermediate quantity returned (same formulas)."""
    c = BK.CST if c is None else c; sig = BK.SIG if sig is None else sig
    th = 1 / (1 - c * sig); tau_min = xa / bb
    H0, H1, H2, _, _, delta = BK.harm(xa, tau_min)
    if fixed:
        H1 = BK.H1_sup(xa, xa / bb, xb / ba)
    else:
        H1 = BK.harm(xa, tau_min)[1]                                          # v1.1: at tau_min
    _, _, _, Z2, eta1, _ = BK.harm(xa, 0.0)
    q = BK.m(xa); G1, RSH, GS, QS = BK.G1, BK.RSH, BK.GS, BK.QS
    kw = (1 + q * H0 * G1 * RSH) / (1 - q * H0 * G1); ks = (1 + q * H1) / (1 - q * H0)
    WN = xb * ((1 + delta) * q * q * (1 + Z2) * GS / (2 * (1 - QS)) + G1 * q * H2) / (1 - G1 * q * H0)
    L12 = sig / (c * th * kw * (bb + WN)); gam2 = (1 - c * th * kw * (bb + WN) * sig) ** 2 / 2
    Psi1 = -np.expm1(-gam2 * L12 / 2)
    Np = (q * (1 + eta1)) ** 2 / (1 - QS) ** 2; tauN = (xb / ba) * Np
    d = sig * np.sqrt(xb) * q * (1 + eta1) / (1 - QS) + sig * QS * np.exp(-0.5) / (1 - QS)
    Ea = (1 + QS) / np.sqrt(2) * (sig * np.sqrt(2 * bb) * np.sqrt((1 + tauN) * (6 + 2 * tauN)) + 2 * bb * d * (1 + tauN))
    Eb = (1 + QS) * (sig * sig * (3 + tauN) + bb * d * d * (1 + tauN))
    main_fac = ba * (1 - q * H0 * G1) / ks
    rhs = 2 * c * sig * th * (bb + WN) * (1 + G1 * q * H0)
    gL1 = np.sqrt(gam2 * L12)
    f_L1 = main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb - rhs
    f_big = main_fac * (-np.expm1(-0.71 ** 2 / 2)) - Ea * 0.71 - Eb - rhs
    return {'H0': H0, 'H1': H1, 'H2': H2, 'delta': delta, 'Z2': Z2, 'eta1': eta1, 'q': q, 'kw': kw, 'ks': ks, 'WN': WN, 'L12': L12, 'gam2': gam2, 'Psi1': Psi1,
            'Np': Np, 'tauN': tauN, 'd': d, 'Ea': Ea, 'Eb': Eb, 'main_fac': main_fac, 'rhs': rhs, 'f_L1': f_L1, 'f_big': f_big,
            'margin': (main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb) / rhs, 'gL1': gL1,
            'increasing': bool(main_fac * gL1 * np.exp(-gam2 * L12 / 2) >= Ea)}


def audit_quantities(key_prefix, cells, cell_fn, point_fn, dirs, npts=6):
    """cell_fn(cell) -> dict, point_fn(point) -> dict; points from pts2(cell)."""
    stats = {k: [0, 0, 0.0, None] for k in dirs}                               # violations, points, worst rel, example
    for cell, plist in cells:
        vc = cell_fn(cell)
        for p in plist:
            vp = point_fn(p)
            for k, dr in dirs.items():
                a, b_ = vc[k], vp[k]; stats[k][1] += 1
                bad = (b_ > a + abs(a) * TOL + 1e-300) if dr == 'U' else (b_ < a - abs(a) * TOL - 1e-300)
                if bad:
                    stats[k][0] += 1; r = (b_ - a) / (abs(a) + 1e-300)
                    if abs(r) > abs(stats[k][2]): stats[k][2] = r; stats[k][3] = [list(map(float, cell)), list(map(float, p)), float(a), float(b_)]
    return [{'key': '%s:%s' % (key_prefix, k), 'cells': len(cells), 'points': v[1], 'violations': v[0], 'worst_rel': v[2], 'example': v[3], 'direction': dirs[k]}
            for k, v in stats.items()]


def sec_bk1(have):
    import bk_narrow_lemma as BK
    for tag, strip, xr, ncell in (('general_tau80', False, (1.0, 80.0), S(3000, 600)), ('strip_x_lt_1', True, (1e-12, 1.0), S(3000, 600))):
        BK.NN = np.array([1.0]) if strip else np.arange(1, 40, dtype=float)
        xs = np.geomspace(*xr, 241 if not strip else 481)
        ix = RNG.integers(0, len(xs) - 1, ncell); jb = RNG.integers(0, 240, ncell)
        cells = []
        for i, j in zip(ix, jb):
            xa, xb = xs[i], xs[i + 1]; bs = np.geomspace(xa / 80, 1, 241); ba, bb = bs[j], bs[j + 1]
            P = [(x, b) for x, b in zip(pts(xa, xb), pts(ba, bb)[RNG.permutation(6)])]
            cells.append(((xa, xb, ba, bb), P))
        for fixed in (False, True):
            pre = 'bk1(%s, %s)' % (tag, 'v1.2 code' if fixed else 'v1.1 code')
            if any(k.startswith(pre) for k in have): continue
            res = audit_quantities(pre, cells, lambda c, fixed=fixed: bk_cell_full(BK, *c, fixed=fixed),
                                   lambda p, fixed=fixed: bk_cell_full(BK, p[0], p[0], p[1], p[1], fixed=fixed), BK_DIR)
            for r in res: write(r)
    BK.NN = np.arange(1, 40, dtype=float)


# ------------------------------------------------------------------------------------------------ bj1
def sec_bj1(have):
    import importlib
    import bj_lemma as J
    QS, E = J.QS, np.e
    def run(tag, cells):
        Qx = lambda a: min(QS, J.m(a) * (1 + J.eta(0, a)))
        funcs = {
            'Gamma1*m (Gamma*)': (lambda a, b: J.Gam1_up(a, b) * J.m(a), lambda x: J.Gam1(x) * J.m(x)),
            'W_max': (lambda a, b: b * min(QS * QS, 4 * np.exp(-a)) * (1 + J.Z2(a)) / (6 * (1 - QS)) * J.UP,
                      lambda x: x * min(QS * QS, 4 * np.exp(-x)) * (1 + J.Z2(x)) / (6 * (1 - QS))),
            'S_xi': (lambda a, b: np.sqrt(b) * J.m(a) * (1 + J.eta(1, a)) / (1 - QS) * J.UP, lambda x: np.sqrt(x) * J.m(x) * (1 + J.eta(1, x)) / (1 - QS)),
            'T1': (lambda a, b: b * (1 + J.Z2(a)) * (Qx(a) / (1 - QS) + Qx(a) ** 2 / (E * (1 - QS) ** 2)) * J.UP,
                   lambda x: x * (1 + J.Z2(x)) * (Qx(x) / (1 - QS) + Qx(x) ** 2 / (E * (1 - QS) ** 2))),
            'T2': (lambda a, b: b * np.sqrt(1 + J.Z2(a)) * J.m(a) * (1 + J.eta(1, a)) * J.UP, lambda x: x * np.sqrt(1 + J.Z2(x)) * J.m(x) * (1 + J.eta(1, x))),
            'Gamma1 (upper)': (lambda a, b: J.Gam1_up(a, b), J.Gam1),
        }
        for name, (fc, fp) in funcs.items():
            key = 'bj1(%s):%s' % (tag, name)
            if key not in have: write(audit_cells(key, cells, fc, fp))
        key = 'bj1(%s):x/Gamma1^2 (infimum of step 3(ii))' % tag
        if key not in have: write(audit_cells(key, cells, lambda a, b: a / J.Gam1_up(a, b) ** 2 * J.DN, lambda x: x / J.Gam1(x) ** 2, kind='inf'))
        key = 'bj1(%s):Gamma1(x)/x decreasing (R1 = Gamma1(1) is the sup on x >= 1)' % tag
        if key not in have and tag == 'general': write(mono_points(key, np.geomspace(1.0, 200.0, S(3000, 300)), lambda x: J.Gam1(x) / x, -1))
        key = 'bj1(%s):functions decreasing beyond x = 200' % tag
        if key not in have and tag == 'general':
            fs = [funcs[k][1] for k in ('Gamma1*m (Gamma*)', 'W_max', 'S_xi', 'T1', 'T2')]
            write(mono_points(key, np.geomspace(150.0, 5000.0, S(2000, 200)), lambda x: sum(f(x) for f in fs), -1))
    if not any(k.startswith('bj1(general)') for k in have) or True:
        st = S(7, 200); xs = J.CELLS; cells = list(zip(xs[:-1][::st], xs[1:][::st])); run('general', cells)
    # the BN strip version: eta = Z2 = 0, x-cells on [1e-12, 1], and the per-x product of step 2(i)
    eta0, Z20 = J.eta, J.Z2
    J.eta = lambda p, x: 0.0; J.Z2 = lambda x: 0.0
    st = S(5, 200); xs = np.concatenate([np.geomspace(1e-12, 0.01, 4001), np.linspace(0.01, 1.0, 4001)[1:]]); cells = list(zip(xs[:-1][::st], xs[1:][::st]))
    run('strip', cells)
    key = 'bj1(strip):Gamma1 q* product (sqrt x A + B)^2'
    if key not in have:
        A, B = J.Gam1_parts(0.5)
        write(audit_cells(key, cells, lambda a, b: (np.sqrt(b) * A + B) ** 2, lambda x: J.Gam1(x) ** 2 / x))
    J.eta, J.Z2 = eta0, Z20


# ------------------------------------------------------------------------------------------------ theta
def sec_theta(have):
    import bl_theta as T, be_cert_p as P
    key = 'theta:eps_table suprema (N1, gamma_P, c_chi) on [1, 400]'
    if key not in have:
        Q = 0.25; xs = np.linspace(1.0, 400.0, 4001); st = S(3, 100); cells = list(zip(xs[:-1][::st], xs[1:][::st]))
        N1p = lambda x: float(P.q1(x)) * (1 + P.eta(1, x))
        out = []
        for name, fc, fp in (('N1', lambda a, b: N1p(a), N1p), ('gamma_P', lambda a, b: P.gamma_P(a, b), lambda x: P.gamma_P(x, x)),
                             ('c_chi', lambda a, b: (np.sqrt(b) * N1p(a) / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2),
                              lambda x: (np.sqrt(x) * N1p(x) / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2))):
            r = audit_cells(key + ' ' + name, cells, fc, fp); out.append(r)
        r = {'key': key, 'cells': sum(o['cells'] for o in out), 'points': sum(o['points'] for o in out), 'violations': sum(o['violations'] for o in out),
             'parts': out}
        write(r)
    for j in range(4):
        for k in range(5):
            key = 'theta:tau^%d nu_%d decreasing for tau >= 80 (sigma = 1e-24)' % (j, k)
            if key not in have: write(mono_points(key, np.geomspace(80.0, 1e4, S(2000, 200)), lambda t, j=j, k=k: t ** j * T.nus(t, 1e-24 * t)[k], -1))
    key = 'theta:nu_tau = 2 sum e_n (1 + n^2 tau) decreasing for tau >= 80'
    if key not in have:
        NN = np.arange(1, 60, dtype=float)
        write(mono_points(key, np.geomspace(80.0, 1e4, S(2000, 200)), lambda t: 2 * float(np.sum(np.exp(-NN ** 2 * t / 2) * (1 + NN ** 2 * t))), -1))
    key = 'theta:BN E_R cells on [1e-12, 1]'
    if key not in have:
        import bj_lemma as J
        sig, tau, Q, N1 = 1e-24, 80.0, 0.25, 0.25
        def ER(a, b):
            nu = T.nus(tau, sig * tau / np.sqrt(2 * a)); gam = 0.25 * J.Gam1_up(a, b)
            cc = (np.sqrt(b) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)
            return (0.375 * sig * tau * Q * nu[0] / a + (1 + Q) * (1 + Q / 2) * cc * nu[0] + Q * sig * tau ** 2 * nu[2] / (8 * a)
                    + tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2 * a) * (1 - Q)) + sig * tau * nu[0] * (1 + Q) * gam / a
                    + tau * nu[1] * (N1 * cc + (1 + Q) * gam / np.sqrt(2 * a)) + tau * Q * nu[2] * (1 + Q) * cc / 2 + Q * sig * tau ** 3 * nu[4] / (8 * a)
                    + sig * tau * Q * nu[2] / (2 * (1 - nu[0])))
        eta0, Z20 = J.eta, J.Z2; J.eta = lambda p, x: 0.0; J.Z2 = lambda x: 0.0
        xs = np.geomspace(1e-12, 1.0, 4001)
        st = S(3, 100); write(audit_cells(key, list(zip(xs[:-1][::st], xs[1:][::st])), ER, lambda x: ER(x, x)))
        J.eta, J.Z2 = eta0, Z20


# ------------------------------------------------------------------------------------------------ sigc (Table 8)
def sec_sigc(have):
    import bl_bj1 as BL
    if any(k.startswith('sigc:') for k in have): return
    corner = BL.flat(BL.constants(1e-24))
    dirs = {}
    for name, key_, ds, dc, why in BL.MONO:
        dirs[key_] = 'U' if (ds > 0 or dc > 0) else 'L'                       # the corner must be the worst point
    sigs = 10.0 ** (-RNG.uniform(24, 300, S(12, 3))); cs = RNG.uniform(0.01, BL.CSTAR, S(12, 3))
    pairs = [(s, c) for s, c in zip(sigs, cs)] + [(1e-24, c) for c in cs[:S(4, 1)]] + [(s, BL.CSTAR) for s in sigs[:S(4, 1)]]
    stats = {k: [0, 0, 0.0, None] for k in dirs}
    for s, c in pairs:
        p = BL.flat(BL.constants(s, c))
        for k, dr in dirs.items():
            a, b_ = corner[k], p[k]; stats[k][1] += 1
            # quantities whose bad direction is "up" must not exceed the corner value; the others must not fall below it
            bad = (b_ > a * (1 + 1e-9)) if dr == 'U' else (b_ < a * (1 - 1e-9))
            if bad: stats[k][0] += 1; stats[k][3] = [float(s), float(c), float(a), float(b_)]
    for k, v in stats.items():
        write({'key': 'sigc:%s' % k, 'cells': 1, 'points': v[1], 'violations': v[0], 'example': v[3], 'direction': dirs[k]})


# ------------------------------------------------------------------------------------------------ upper
def sec_upper(have):
    key = 'upper:h(k) at interior points against the cell bound (bf_cert_upper)'
    if key in have: return
    from decimal import Decimal as D
    from ba_ivl import I, Dl
    import bf_cert_upper as U
    a = D('3.43'); cells = []
    while a < D('240.0'):
        b = min(a + D('0.02') * (1 + a / 10), D('240.0')); cells.append((a, b)); a = b
    pick = [cells[i] for i in sorted(RNG.choice(len(cells), S(150, 30), replace=False))]
    nv = 0; npts = 0; worst = -1e9; ex = None
    for a, b in pick:
        ub, _ = U.cell_bound(a, b)
        for u in (D('1e-9'), D('0.5'), D(1) - D('1e-9'), D(str(RNG.random())), D(str(RNG.random()))):
            k = a + (b - a) * u; hp = U.h_of(Dl(I(k), I(0)))[0].v; npts += 1
            if hp.lo > ub: nv += 1; ex = [float(a), float(b), float(k), float(ub), float(hp.lo)]
            worst = max(worst, float(hp.hi - ub))
    # float cross-check where float is accurate (k <= 6): bf_upper.U_of with the harmonics summed exactly
    from bf_upper import U_of
    nvf = 0; nf = 0
    for a, b in [c for c in cells if c[1] <= D(6)][::S(5, 40)]:
        ub, _ = U.cell_bound(a, b)
        for u in (0.1, 0.5, 0.9):
            k = float(a) + (float(b) - float(a)) * u; t = 4 * k * k + 11 * np.log(k) - 8 * np.log(2)
            h = U_of(t)[0]['U_main'] * np.sqrt(t) - 1 - 6.0 * np.log(t) / t; nf += 1
            if h > float(ub): nvf += 1
    # the harmonic interval parameters: E_p = sum_{n>=2} n^p (a_n/a_1) e^{-(n^2-1)K/2} <= eta_p(K_min) along S'_C (K = k^2/(1 + rho) >= 10.5)
    nve = 0
    for k in np.linspace(3.43, 240.0, S(400, 40)):
        K = k * k * (1 - 1e-6); tau = 1 / k ** 2; n = np.arange(2, 60, dtype=float)
        an = np.exp(-(n * tau) ** 2 / 2) / np.exp(-tau ** 2 / 2)
        for p in (0, 1, 2):
            if np.sum(n ** p * an * np.exp(-(n * n - 1) * K / 2)) > float(U.EP[p].hi): nve += 1
    write({'key': key, 'cells': len(pick), 'points': npts, 'violations': nv, 'worst_gap_hi_minus_bound': worst, 'example': ex,
           'float_points_k_le_6': nf, 'float_violations': nvf, 'harmonic_parameter_points': S(400, 40) * 3, 'harmonic_parameter_violations': nve})


# ------------------------------------------------------------------------------------------------ members
def sec_members(have):
    key = 'members:point enclosures inside the cell enclosures (bg_members, bh_members_fc)'
    if key in have: return
    from decimal import Decimal as D
    from ba_ivl import I, Dl
    from bg_members import quantities
    from bh_members_fc import deficit
    a = D('3.43'); cells = []
    while a < D('240.0'):
        b = min(a + D('0.02') * (1 + a / 10), D('240.0')); cells.append((a, b)); a = b
    pick = [cells[i] for i in sorted(RNG.choice(len(cells), S(120, 20), replace=False))]
    nv = 0; npts = 0; ex = None
    for a, b in pick:
        qc = quantities(Dl(I(a, b), I(0))); dc = sum(x.hi for x in deficit(Dl(I(a, b), I(0))))
        for u in (D('1e-9'), D('0.5'), D(1) - D('1e-9'), D(str(RNG.random())), D(str(RNG.random()))):
            k = a + (b - a) * u; qp = quantities(Dl(I(k), I(0))); dp = sum(x.hi for x in deficit(Dl(I(k), I(0)))); npts += 1
            ok = qp[0].hi <= qc[0].hi and qp[1].hi <= qc[1].hi and qp[2].lo >= qc[2].lo and dp <= dc
            if not ok: nv += 1; ex = [float(a), float(b), float(k)]
    write({'key': key, 'cells': len(pick), 'points': npts, 'violations': nv, 'example': ex})


# ------------------------------------------------------------------------------------------------ tail
def sec_tail(have):
    key = 'tail:T(l) increasing on a grid 20x finer (bl_chain perturbed and bh v1.0)'
    if key in have: return
    import bl_chain as LC
    from scipy.optimize import brentq
    out = {}
    for name, fn in (('bl_chain', 'results/bl_chain_tail.json'), ('bh_tail', 'results/bh_tail.json')):
        T = json.load(open(fn)); l_end, K_end, Fd = T['l_end'], T['K_end'], T['Fdef_end']; c = 1.41 + 0.25
        meta = json.load(open('results/bh_cert_cells_meta.json')); c = meta['S1_P'] + 0.25
        eps = 1e-6 if name == 'bl_chain' else 0.0
        def Tl(l):
            K = K_end - 4.5 * np.log(l / l_end); X = brentq(lambda X: 4 * X - l - 5 * np.log(X) - K, 1.0, l + 1e3 + abs(K), maxiter=2000)
            return (1 - Fd) * np.sqrt(l / ((1 + eps) * (X + c))) / 2
        g = np.exp(np.linspace(np.log(l_end), np.log(1e300), S(8000, 800))); v = np.array([Tl(l) for l in g])
        out[name] = {'points': len(g), 'violations': int(np.sum(np.diff(v) < -1e-15)), 'min': float(v.min()), 'at_l_end': float(v[0])}
    write({'key': key, 'cells': 2, 'points': sum(o['points'] for o in out.values()), 'violations': sum(o['violations'] for o in out.values()), 'parts': out})


# ------------------------------------------------------------------------------------------------ lcells
LC_DIR = {'U0': 'L', 'x_lo': 'L', 'phi1': 'L', 'beta': 'U', 'nu': 'U', 'V_max': 'U', 'omega': 'L', 'Gam2': 'U', 'Gam3P': 'U', 'Wp_max': 'U',
          'W_p_forced_if_L_gt_1': 'L', 'K': 'U', 'X': 'U', 'L2max': 'U', 'dH': 'U', 'rC': 'U', 'rw': 'U', 'F_int': 'L', 'eta_prime': 'U', 'theta': 'U',
          'absorb_ratio': 'U', 'condA_margin': 'L', 'caseB_m': 'L'}


def sec_lcells(have, n_cells=60, n_pts=5):
    import bl_chain as LC, bh_cert_cells as H, bf_cert_p as B
    S1 = B.S1_P(); Si1 = B.Si_at(1.0); C = H.cells()
    # The chain bounds each supremum over x by cells (x-factor at the right end, decay factors at the left) on a grid laid over the current range.
    # That bound is valid but depends on where the grid falls: two ranges give two different over-estimates (the 'grid wiggle' of App. B).  The
    # point value to compare with is therefore the supremum itself, from below: the largest point value f(x, x) on a grid 4 times finer (stage BQ;
    # BP compared the point's own cell bound, which can exceed the cell's bound by the wiggle although both are valid).
    _cs = LC.cell_sup
    def _point_sup(f, x0, x1, n=4001):
        xs = np.linspace(x0, x1, 4 * n); return max(1e-300, float(max(f(x, x) for x in xs)))
    idx = sorted(set(list(range(0, 421, 7)) + [420]))[:n_cells + 1] if not QUICK else [0, 140, 280, 420]
    n_pts = S(n_pts, 3)
    for i in idx:
        key = 'lcells:cell %d' % i
        if key in have: continue
        la, lb = C[i]
        s_hat = np.exp(-la) if la < 690 else 1e-300
        G = H.G_of(la); F0 = 1.02 * H.G_of(lb)
        # the cell's bootstrap path
        steps = []; F = F0; x_lo = 1.0
        for k in range(60):
            c = LC.chain(s_hat, lb, G, H.ETA, F, x_lo, S1)
            steps.append((F, x_lo, c)); Fn = 0.9 * c['F_int']
            U0n = Fn * Fn / (4 * G * G * (1 + LC.EPS_IN)) - 0.25; x_new = max(1.0, U0n / Si1, U0n - S1)
            if Fn <= F * (1 + 1e-13) and x_new <= x_lo * (1 + 1e-13): break
            F = max(F, Fn); x_lo = max(x_lo, x_new)
        cell_L = LC.cell(la, lb, S1, Si1)['L_cell']
        viol = {}; npts = 0; Lviol = 0; ex = None
        for u in np.concatenate([[1e-9, 1 - 1e-9], RNG.random(n_pts - 2)]):
            l = la + (lb - la) * u; sig = np.exp(-l) if la < 690 else 1e-300; Gl = H.G_of(l); epsr = RNG.uniform(0, 0.1); eta = 1 / (1 - epsr)
            for (F, x_lo, cc) in (steps[0], steps[-1]):
                LC.cell_sup = _point_sup                                          # the point's suprema over x as exactly as possible (stage BQ)
                try: p = LC.chain(sig, l, Gl, eta, F, x_lo, S1)
                finally: LC.cell_sup = _cs
                npts += 1
                p['caseB_m'] = p.get('sigma_caseB_limit', 0) / sig; cc['caseB_m'] = cc['sigma_caseB_limit'] / s_hat
                for k, dr in LC_DIR.items():
                    if k not in cc or k not in p: continue
                    a, b_ = cc[k], p[k]
                    bad = (b_ > a + abs(a) * TOL + 1e-300) if dr == 'U' else (b_ < a - abs(a) * TOL - 1e-300)
                    if bad: viol[k] = viol.get(k, 0) + 1; ex = [k, float(l), float(a), float(b_)]
                if not p['ok']: viol['ok'] = viol.get('ok', 0) + 1
            # the final statement at this l: run the point's own certificate (degenerate cell) and compare L
            Lp = LC.cell(l, l * (1 + 1e-15), S1, Si1)['L_cell']
            if Lp < cell_L * (1 - 1e-12): Lviol += 1
        write({'key': key, 'cells': 1, 'points': npts, 'violations': sum(viol.values()) + Lviol, 'by_quantity': viol, 'L_violations': Lviol, 'example': ex,
               'la': la, 'lb': lb})


SECTIONS = {'xsup': sec_xsup, 'strip': sec_strip, 'bk1': sec_bk1, 'bj1': sec_bj1, 'theta': sec_theta, 'sigc': sec_sigc, 'upper': sec_upper,
            'members': sec_members, 'tail': sec_tail, 'lcells': sec_lcells}

if __name__ == '__main__':
    np.seterr(all='ignore')
    want = sys.argv[1:] or list(SECTIONS)
    for s in want:
        t0 = time.time(); SECTIONS[s](done()); print('section %s done in %.0f s' % (s, time.time() - t0), flush=True)
    rows = list(done().values())
    print('%d checks, %d with violations' % (len(rows), sum(1 for r in rows if r.get('violations'))))
    for r in rows:
        if r.get('violations'): print('VIOLATION:', r['key'], r.get('violations'), r.get('example'))
