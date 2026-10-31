"""Stage BS §2.1: Theorem 11 (upper side on the comb sequence S'_C) for every k >= 240, i.e. sigma <= 1e-100000 (proof with constants in
outward-rounded interval arithmetic), so that U_P sqrt(l) <= 1 + 6.0 ln l/l holds for all sigma <= 1e-24 (with the certificate on [3.43, 240]).

Idea.  On S'_C every length is a monomial in k and E := e^{-k^2} (sigma = 256 k^-11 E^4, ell = 32 k^-6 E^2, L_w = 8 k^-4 E, L_v = 32 k^-8 E^2).
Every quantity of the bound U_P (bf_cert_upper.h_of, transcribed line by line) is carried as an *asymptotic form*
        X = c * k^beta * E^gamma * (1 + R k^-nu),     valid for every k >= K0 = 240,
with c an exact constant in Q(sqrt2) (or an interval when the term is only ever absorbed into a remainder), beta, gamma, nu exact rationals and R an
interval.  Rules (each proved in the docstring of the method): a sum keeps the leading monomial and turns every other term into a relative remainder
-- a term with the same E-power but a smaller power of k gives an exact algebraic remainder (ratio c2/c1 times k^-(beta1 - beta2)); a term with a
larger E-power is super-exponentially small and is bounded by its value at K0 times k^-NU_TINY (k^{p} E^{q}, q > 0, times k^{NU_TINY} decreases for
k >= K0 when p + NU_TINY < 2 q K0^2, asserted); products, inverses, square roots and the 1.5-th power act on (1 + z), z = R k^-nu, with explicit
bounds of (1+z)^-1 - 1, (1+z)^(1/2) - 1, (1+z)^(3/2) - 1 over the range of z; the harmonic parameters E_p, Z_p and the factor exp(k^2 rho/2) are
super-exponentially small relative terms.  Since every leading constant is exact, U_P = (1/(2k)) (1 + R_U k^-2) with an interval R_U; then
    U_P sqrt(t) - 1 <= [R_U + (11 lam - 8 ln2)/8 + R_U (11 lam - 8 ln2)/(8 k^2)] / k^2,   lam = ln k,   t = 4k^2 + 11 lam - 8 ln2,
    6 ln t/t >= (3 lam + 3 ln2)(1 - 11 lam/(4 k^2)) / k^2,
and h < 0 for every k >= K0 reduces to  13 lam/8 + 4 ln2 - R_U - B >= 0  with B >= [R_U 11 lam/8 + (3 lam + 3 ln2) 11 lam/4]/k^2 (lam^j/k^2 decreases
for k >= e, so B is its value at K0); the left side increases with lam.
Note: E_p, Z_p are bounded by their k-dependent bounds (valid for the true sums since K >= 0.999 k^2), not by the fixed box of
bf_cert_upper (K_min = 10.5); with the fixed box the claim would fail near k ~ 1e7.
Checks: the asymptotic forms are evaluated at k = 240, 241.7, 300, 1000 and compared with bf_cert_upper.h_of (interval values must lie inside).
-> results/bs_upper_tail.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
from fractions import Fraction as Fr
from decimal import Decimal as D
from ba_ivl import I

K0 = 240
NU_TINY = Fr(6)                                            # decay exponent given to super-exponentially small relative terms
LN_K0 = I(K0).ln()
LOCALS = {}                                                # the forms of every intermediate of U_form (for bs_upper_tail_stress.py)


class Q2:
    """a + b sqrt2, a, b rationals (exact)."""
    def __init__(self, a, b=0): self.a, self.b = Fr(a), Fr(b)
    def __add__(x, y): return Q2(x.a + y.a, x.b + y.b)
    def __mul__(x, y): return Q2(x.a * y.a + 2 * x.b * y.b, x.a * y.b + x.b * y.a)
    def inv(x):
        d = x.a * x.a - 2 * x.b * x.b; assert d != 0; return Q2(x.a / d, -x.b / d)
    def iv(x): return I(D(x.a.numerator) / D(x.a.denominator)) + I(D(x.b.numerator) / D(x.b.denominator)) * I(2).sqrt()
    def sqrt(x):
        """exact when x is a rational r with r or 2r a rational square; else None."""
        if x.b != 0 or x.a < 0: return None
        for mult, res in ((1, lambda s: Q2(s)), (2, lambda s: Q2(0, s / 2))):          # sqrt(r) = s, or sqrt(r) = s/2 * sqrt2 with s^2 = 2r
            r = x.a * mult; n, d = r.numerator, r.denominator
            rn, rd = int(round(n ** 0.5)), int(round(d ** 0.5))
            if rn * rn == n and rd * rd == d: return res(Fr(rn, rd))
        return None
    def __repr__(s): return '%s + %s sqrt2' % (s.a, s.b)


def hull(a, b): return I(min(a.lo, b.lo), max(a.hi, b.hi))


def k_neg(p):
    """sup over k >= K0 of k^{-p} for p >= 0 (= K0^{-p}); returns the interval [0, K0^{-p}]."""
    p = Fr(p); assert p >= 0
    v = (-(I(D(p.numerator) / D(p.denominator))) * LN_K0).exp()
    return I(D(0), v.hi)


def tiny_sup(p, q, nu):
    """sup over k >= K0 of k^{p + nu} E^q (E = e^{-k^2}, q > 0): attained at K0 when (p + nu) < 2 q K0^2 (the log-derivative is negative)."""
    p, q, nu = Fr(p), Fr(q), Fr(nu); assert q > 0 and p + nu < 2 * q * K0 * K0
    e = (I(D((p + nu).numerator) / D((p + nu).denominator))) * LN_K0 - I(D(q.numerator) / D(q.denominator)) * I(K0 * K0)
    return I(D(0), e.exp().hi)


class AF:
    """X = c k^beta E^gamma (1 + R k^-nu) for all k >= K0; c: Q2 (exact) or I (interval, then absorbed only)."""
    def __init__(self, c, beta=0, gamma=0, R=None, nu=None):
        self.c = c if isinstance(c, (Q2, I)) else Q2(c); self.beta, self.gamma = Fr(beta), Fr(gamma)
        self.R = R if R is not None else I(0); self.nu = Fr(nu) if nu is not None else NU_TINY
    def civ(self): return self.c.iv() if isinstance(self.c, Q2) else self.c

    # ---- relative remainders
    @staticmethod
    def _merge(R1, nu1, R2, nu2):
        """R1 k^-nu1 + R2 k^-nu2 = R k^-nu with nu = min: the other term's k^-(nu_i - nu) lies in [0, K0^-(nu_i - nu)]."""
        nu = min(nu1, nu2)
        a = R1 if nu1 == nu else R1 * k_neg(nu1 - nu)
        b = R2 if nu2 == nu else R2 * k_neg(nu2 - nu)
        return a + b, nu

    def _rel(self, other):
        """other / self as a relative term (R, nu); other must be smaller (lower E-power, or same E-power and lower k-power)."""
        ratio = other.civ() / self.civ()
        db, dg = other.beta - self.beta, other.gamma - self.gamma
        if dg == 0:
            assert db < 0, ('not smaller', db)
            R, nu = ratio, -db
        else:
            assert dg > 0, ('not smaller', dg)
            R, nu = ratio * tiny_sup(db, dg, NU_TINY), NU_TINY
        # other's own remainder: (1 + R_o k^-nu_o) multiplies
        Ro = other.R * k_neg(other.nu) if other.nu > 0 else other.R
        return R * (1 + Ro), nu

    def __add__(x, y):
        y = y if isinstance(y, AF) else AF(y)
        if (x.beta, x.gamma) == (y.beta, y.gamma):
            exact = isinstance(x.c, Q2) and isinstance(y.c, Q2)
            c = x.c + y.c if exact else x.civ() + y.civ()
            cI = c.iv() if exact else c; assert cI.lo > 0
            # (c1 (1 + z1) + c2 (1 + z2))/(c1 + c2) - 1 = (c1 z1 + c2 z2)/(c1 + c2), weights in [0, 1] (both c > 0)
            assert x.civ().lo > 0 and y.civ().lo > 0
            w1 = x.civ() / cI; w2 = y.civ() / cI
            w1 = I(max(D(0), w1.lo), min(D(1), w1.hi)); w2 = I(max(D(0), w2.lo), min(D(1), w2.hi))
            Rx, nx = x.R * w1, x.nu; Ry, ny = y.R * w2, y.nu
            R, nu = AF._merge(Rx, nx, Ry, ny)
            return AF(c, x.beta, x.gamma, R, nu)
        big, small = (x, y) if (x.gamma < y.gamma or (x.gamma == y.gamma and x.beta > y.beta)) else (y, x)
        Rs, ns = big._rel(small)
        R, nu = AF._merge(big.R, big.nu, Rs, ns)
        return AF(big.c, big.beta, big.gamma, R, nu)
    __radd__ = __add__

    def __sub__(x, y):
        """x - y with y smaller than x."""
        y = y if isinstance(y, AF) else AF(y)
        Rs, ns = x._rel(y)
        R, nu = AF._merge(x.R, x.nu, -Rs, ns)
        return AF(x.c, x.beta, x.gamma, R, nu)

    def __rsub__(x, y): return (y if isinstance(y, AF) else AF(y)) - x

    def __mul__(x, y):
        y = y if isinstance(y, AF) else AF(y)
        c = x.c * y.c if isinstance(x.c, Q2) and isinstance(y.c, Q2) else x.civ() * y.civ()
        R, nu = AF._merge(x.R, x.nu, y.R, y.nu)
        cross = x.R * y.R * k_neg(x.nu + y.nu - nu)                              # R1 R2 k^-(nu1+nu2) = (R1 R2 k^-(nu1+nu2-nu)) k^-nu
        return AF(c, x.beta + y.beta, x.gamma + y.gamma, R + hull(cross, I(0)), nu)
    __rmul__ = __mul__

    def zrange(self):
        """the range of z = R k^-nu over k >= K0."""
        return hull(self.R * k_neg(self.nu), I(0)) if self.nu > 0 else self.R

    def inv(x):
        """(1 + z)^-1 = 1 - z/(1 + z):  R' = -R/(1 + z), z in zrange."""
        z = x.zrange(); assert z.lo > -1
        Rn = -(x.R) / (1 + z)
        c = x.c.inv() if isinstance(x.c, Q2) else 1 / x.civ()
        return AF(c, -x.beta, -x.gamma, Rn, x.nu)

    def __truediv__(x, y): return x * (y if isinstance(y, AF) else AF(y)).inv()
    def __rtruediv__(x, y): return (y if isinstance(y, AF) else AF(y)) * x.inv()

    def _pw(x, p):
        """(1 + z)^p - 1 = z g(z), g between p (1 + z)^(p-1) at the ends (mean value); exact constants when possible."""
        z = x.zrange(); assert z.lo > -1
        pI = I(D(p.numerator) / D(p.denominator))
        gl = pI * ((pI - 1) * (1 + z).ln()).exp()                                # p (1 + xi)^(p - 1), xi in z (convex hull)
        Rn = x.R * gl
        if isinstance(x.c, Q2):
            if p == Fr(1, 2): cc = x.c.sqrt()
            elif p == Fr(3, 2):
                s = x.c.sqrt(); cc = (s * x.c) if s is not None else None
            else: cc = None
            if cc is None: cc = (pI * x.c.iv().ln()).exp()
        else: cc = (pI * x.civ().ln()).exp()
        return AF(cc, x.beta * p, x.gamma * p, Rn, x.nu)

    def sqrt(x): return x._pw(Fr(1, 2))
    def p15(x): return x._pw(Fr(3, 2))
    def sq(x): return x * x

    def val(self, k):
        """interval value at a given k (for the consistency checks)."""
        kI = I(k); E = (-(kI * kI)).exp()
        mono = (I(D(self.beta.numerator) / D(self.beta.denominator)) * kI.ln()).exp() * \
               (I(D(self.gamma.numerator) / D(self.gamma.denominator)) * (-(kI * kI))).exp()
        z = self.R * (-(I(D(self.nu.numerator) / D(self.nu.denominator))) * kI.ln()).exp()
        return self.civ() * mono * (1 + z)


def one_plus_tiny(cI, p, q):
    """1 + w with 0 <= w <= cI k^p E^q (super-exponentially small): AF(1) with R in [0, cI sup k^{p+NU} E^q]."""
    return AF(Q2(1), 0, 0, I(D(0), (cI * tiny_sup(p, q, NU_TINY)).hi), NU_TINY)


def U_form():
    """U_P on S'_C (bf_cert_upper.h_of, line by line)."""
    sig = AF(256, -11, 4); ell = AF(32, -6, 2); Lw = AF(8, -4, 1); Lv = AF(32, -8, 2)
    rho = sig.sq() / Lv.sq()
    assert (rho.civ() * tiny_sup(rho.beta, rho.gamma, 0) * (1 + rho.zrange())).hi < D('1e-3')     # K = k^2/(1 + rho) >= 0.999 k^2
    V = Lv.sq() * (1 + rho); L2 = Lw.sq() + sig.sq()
    Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    a1 = AF(Q2(2), 0, 0, I(D('-0.5'), D(0)), 4)                                  # 2 e^{-1/(2k^4)} = 2 (1 + z), z in [-1/(2k^4), 0]
    # qbar_1 = a_1 e^{-K/2}, K = k^2/(1 + rho):  = a_1 E^{1/2} exp(k^2 rho/(2(1 + rho))), k^2 rho = 64 k^-4 E^4 (exp(w) - 1 <= 2w for w <= 1/2)
    qb = a1 * AF(1, 0, Fr(1, 2)) * one_plus_tiny(I(64), -4, 4)
    q0 = qb * (Sb * L2 / (Lw.sq() * (Sb + sig.sq()))).sqrt()
    s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2)
    x = AF(1, 2, 0) / (1 + rho).sq()
    s = s2.sqrt(); L = L2.sqrt(); s2g = sig.sq(); b = s2g / V
    # harmonics: E_p <= sum_{m>=2} m^p e^{-(m^2 - 1) K/2} <= 2 * 2^p e^{-3K/2} (ratio of terms <= 1/2), K >= 0.999 k^2: <= 8 E^{1.4985}
    # Z_p <= sum m^p e^{-(m^2-1) K} <= 8 E^{2.997}
    Ep = one_plus_tiny(I(8), 0, Fr(14985, 10000)); Zp = one_plus_tiny(I(8), 0, Fr(2997, 1000))
    Q = qb * Ep; N1 = qb * Ep; N2 = qb * Ep; Sq0 = q0.sq() * Zp; Sq2 = q0.sq() * Zp
    oQ = 1 - Q; pQ = 1 + Q
    g15 = (1 + 2 * L2 / s2).p15(); geo = s / (s2 + 2 * L2).sqrt()
    u0 = 4 * L2.sq() * Sq0 / (s2.sq() * g15)
    Jww_up = (1 + u0 / (8 * oQ)) / L2
    I2_lo = q0.sq() * geo; I2_up = Sq2 * geo
    den_lo = b + x * I2_lo / (2 * pQ); Jper_up = x * I2_up / (2 * oQ * s2g)
    a2 = (1 - b).sq() / 2; y = a2 * V / 2
    # e^{-y} = 1 - y psi(y), psi(y) = (1 - e^{-y})/y in [1 - y/2, 1]; y = a2 V/2 is super-small (V = 1024 k^-16 E^4 (1 + rho))
    ome = y * one_plus_tiny_neg(y)                                                # 1 - e^{-y} = y psi(y)
    e = 1 - ome
    E1 = AF(I(1).exp())
    xi = sig * x.sqrt() * N1 / oQ; Xi = xi.sq() + s2g * s2g * Q.sq() / (E1 * s2 * oQ.sq())
    r = (3 * (xi.sq() + s2g * s2g / L2 + s2g * Q.sq() * (s2g / s2) / (E1 * oQ.sq()))).sqrt()
    gam = (x * (N2 * pQ + N1.sq()) + x.sqrt() * N1 * (1 + 2 * Q) / E1.sqrt()) / oQ.sq()
    v = L2 * s2 / (s2 + 2 * L2); st2w2 = Jper_up * v
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2.sq()
    st2R2 = st2w2 * (1 + (st2Rw / st2w2).sqrt()).sq();   # = (sqrt st2w2 + sqrt st2Rw)^2, written so the leading constant stays exact
    omC = (L + r).sq() / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + a2 * e + gam * omC
    U = 2 * s2g * Jww_up / (sig * den_lo) + s2g / 2 * Esv2phi / (sig * den_lo)
    LOCALS.clear(); LOCALS.update({kk: vv for kk, vv in locals().items() if isinstance(vv, AF)})
    return U, {'sig': sig, 'L2': L2, 's2': s2, 'q0': q0, 'u0': u0, 'den_lo': den_lo, 'Jww_up': Jww_up, 'Jper_up': Jper_up, 'Esv2phi': Esv2phi, 'Qbar': Q, 'x': x,
               'rho_window': s2 / (2 * L2)}


def one_plus_tiny_neg(y):
    """psi(y) = 1 + w, w in [-y/2, 0], y an AF that is super-small (gamma > 0)."""
    assert y.gamma > 0
    sup = y.civ() * tiny_sup(y.beta, y.gamma, NU_TINY) * (1 + y.zrange())
    return AF(Q2(1), 0, 0, I(-(sup.hi) / 2, D(0)), NU_TINY)


def conclusion(U):
    assert isinstance(U.c, Q2) and (U.c.a, U.c.b) == (Fr(1, 2), 0) and U.beta == -1 and U.gamma == 0, (U.c, U.beta, U.gamma)
    assert U.nu >= 2
    RU = U.R * k_neg(U.nu - 2) if U.nu > 2 else U.R                               # R_U k^-2 >= R k^-nu
    RU = hull(RU, I(0)) if U.nu > 2 else RU
    Rhi = max(RU.hi, D(0))
    lam = LN_K0; ln2 = I(2).ln(); k2 = I(K0 * K0)
    B = (I(Rhi) * 11 * lam / 8 + (3 * lam + 3 * ln2) * 11 * lam / 4) / k2      # lam^j/k^2 decreases for k >= e (j <= 2)
    lhs = 13 * lam / 8 + 4 * ln2 - I(Rhi) - B
    global LHS_LO; LHS_LO = lhs.lo
    return {'R_U_interval': [float(RU.lo), float(RU.hi)], 'nu': float(U.nu), 'B_at_K0': float(B.hi), 'lhs_at_K0': float(lhs.lo), 'h_negative_for_all_k_ge_K0': bool(lhs.lo > 0)}


if __name__ == '__main__':
    U, parts = U_form()
    out = {'U_form': {'c': str(U.c), 'beta': str(U.beta), 'gamma': str(U.gamma), 'nu': str(U.nu), 'R': [float(U.R.lo), float(U.R.hi)]}}
    out['conclusion'] = conclusion(U)
    print('U_P = (1/(2k)) (1 + R k^-%s), R in [%.6g, %.6g]' % (U.nu, float(U.R.lo), float(U.R.hi)))
    print(out['conclusion'])
    # consistency with the interval certificate at single k
    from ba_ivl import Dl
    import bf_cert_upper as BU
    chk = []
    for kk in ('240', '241.7', '300', '1000'):          # (beyond k ~ 1500 the E-factors of h_of underflow the Decimal context)
        _, Uv, t = BU.h_of(Dl(I(D(kk)), I(0)))
        uf = U.val(D(kk)); inside = bool(Uv.v.lo <= uf.hi and Uv.v.hi >= uf.lo)
        chk.append({'k': kk, 'h_of_U': [float(Uv.v.lo), float(Uv.v.hi)], 'form_U': [float(uf.lo), float(uf.hi)], 'overlap': inside,
                    'form_upper_ge_h_of': bool(uf.hi >= Uv.v.lo)})
        print(chk[-1])
    out['consistency'] = chk
    # class membership for k >= K0 (P3) Qbar <= 1/4, (P4) varrho = s^2/(2L^2) <= 1/8, (P5) x >= 1: sup over k >= K0 of c k^beta E^gamma (1 + z)
    def log10_sup(f):
        s = f.civ() * tiny_sup(f.beta, f.gamma, 0) * (1 + f.zrange()) if f.gamma > 0 else None
        return float((s.hi.ln() / D(10).ln()))
    Qb = parts['Qbar']; rw = parts['rho_window']; xx = parts['x']
    assert xx.beta == 2 and xx.gamma == 0
    xlo = (xx.civ() * (1 + xx.zrange())).lo * K0 * K0                      # x >= k^2 inf(c (1 + z)) >= K0^2 inf(...)  (c (1 + z) > 0)
    out['members'] = {'log10_Qbar_sup': log10_sup(Qb), 'log10_varrho_sup': log10_sup(rw), 'x_min': float(xlo),
                      'P3': log10_sup(Qb) < -0.6, 'P4': log10_sup(rw) < -0.91, 'P5': bool(xlo >= 1)}
    print(out['members'])
    # Lemma BL7 (theta_tau remainders) for k >= K0.  tau = 4/(k^3 sigma (1 + sigma k^5/4)) increases with k (bm_bl7.py), ln tau >= ln tau(K0);
    # 1/sigma <= tau k^3/2 <= tau^2/2 (k^3 <= tau), sqrt t <= 3k <= tau, sigma tau <= 4/k^3 <= 4/K0^3:  the bm_bl7 bookkeeping becomes
    #   1.5 eps_U + sqrt(t) A_U <= 24 tau nu + tau [ (0.375 tau + 0.25) nu tau^2/2 + K_E tau^3 nu ]/1.98 <= (25 + K_E) tau^5 nu,
    #   ln nu <= ln 2.0001 - tau/2 + sigma tau.  This must be below the margin lhs/k^2 >= lhs(K0)/tau (k^2 <= tau); with
    #   G(tau) = ln(25 + K_E) + 6 ln tau - tau/2 + ln 2.0001 + 4/K0^3 - ln lhs(K0)  decreasing for tau > 12, G(tau(K0)) < 0 suffices.
    bl = json.load(open('results/bm_bl7.json'))['comb']; KE = D(repr(bl['K_E'])) * D('1.001')                                  # float from bm_bl7.json, widened
    kI = I(K0); lnsig = I(256).ln() - 11 * kI.ln() - 4 * kI.sq(); sig = lnsig.exp()
    ln_tau = I(4).ln() - 3 * kI.ln() - lnsig - (1 + sig * kI.sq().sq() * kI / 4).ln()
    tau = ln_tau.exp()
    G = I(25 + KE).ln() + 6 * ln_tau - tau / 2 + I(D('2.0001')).ln() + I(4) / I(K0 ** 3) - I(LHS_LO).ln()
    out['bl7_tail'] = {'log10_tau_at_K0': float((ln_tau / I(10).ln()).lo), 'G_upper_log10_abs': float(((-G).lo).log10()), 'corrections_below_margin': bool(G.hi < 0)}
    print(out['bl7_tail'])
    out['ok'] = (out['conclusion']['h_negative_for_all_k_ge_K0'] and all(c['overlap'] and c['form_upper_ge_h_of'] for c in chk)
                 and out['members']['P3'] and out['members']['P4'] and out['members']['P5'] and out['bl7_tail']['corrections_below_margin'])
    print('ok', out['ok'])
    json.dump(out, open(os.environ.get('BS_OUT', 'results/bs_upper_tail.json'), 'w'), indent=1)
