"""Stage BA §1: Theorem AZ1's final one-variable inequality, certified cell by cell (computer-assisted proof, as Theorem AY1).

Claim:  h(k) := U(k) sqrt(t(k)) - 1 - 7.4 ln t(k) / t(k) < 0  for every k in [k_lo, k_hi],
  t(k) = 4k^2 + 11 ln k = ln(1/sigma) (the sequence S'; t is increasing in k, so this covers every sigma with 1e-100000 <= sigma <= 1e-24),
  U = the closed-form upper bound of Lemma AZ1 (the same formula as az_upper.U_main / U_main_dec, checked against them below).
Method.
 * On S' every length is an exact monomial exp(alpha k^2 + beta ln k) times an O(1) factor:
   sigma = k^-11 e^{-4k^2}, ell = 2 k^-6 e^{-2k^2}, Lw = 2 k^-4 e^{-k^2}, Lv = 2 k^-8 e^{-2k^2}.  Numbers are carried as
   (alpha, beta, mantissa) with exact rational exponents (class S), so the huge factors cancel exactly and only O(1) mantissas
   (and exponentially small ratios) enter the interval arithmetic.
 * Mantissas are interval dual numbers (ba_ivl.Dl: rigorous enclosures of value and derivative; outward-widened 60-digit decimal).
 * The few transcendental pieces with exponentially small arguments use explicit enclosures: kq = F(u) = ((1-u)^{-1/2} - 1)/u (increasing,
   1/2 + 3u/8 <= F <= 1/2 + 3u/8 + u^2/(1-u), 0 <= F' <= (1-u)^{-2}); (1 - e^{-y})/y in [1 - y/2, 1 - y/2 + y^2/6] (decreasing, derivative
   in [-1/2, 0]) for 0 <= y <= 1e-3.
 * Cells [a, b] cover [k_lo, k_hi] (t(k_lo) <= 24 ln 10, t(k_hi) >= 1e5 ln 10, checked in interval arithmetic).  On each cell the mean-value
   form  h <= h(m) + max|h'([a, b])| (b - a)/2  is evaluated; a cell that does not give h < 0 is bisected.
-> results/ba_cert.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, time
from fractions import Fraction as Fr
from decimal import Decimal as D
from ba_ivl import I, Dl

C74 = D('7.4')


def _Fu(u):
    """F(u) = ((1 - u)^{-1/2} - 1)/u for an interval dual u with 0 < u <= 1/4."""
    assert u.v.lo > 0 and u.v.hi <= D('0.25')
    lo = D('0.5') + 3 * u.v.lo / 8; hi = D('0.5') + 3 * u.v.hi / 8 + u.v.hi ** 2 / (1 - u.v.hi)
    dF = I(D(0), 1 / (1 - u.v.hi) ** 2 * (1 + D('1e-50')))
    return Dl(I(lo * (1 - D('1e-55')), hi * (1 + D('1e-55'))), dF * u.d)


def _psi(y):
    """(1 - e^{-y})/y for an interval dual y with 0 <= y <= 1e-3."""
    assert y.v.lo >= 0 and y.v.hi <= D('1e-3')
    lo = 1 - y.v.hi / 2; hi = 1 - y.v.lo / 2 + y.v.lo ** 2 / 6
    return Dl(I(lo * (1 - D('1e-55')), hi * (1 + D('1e-55'))), I(D('-0.5'), D(0)) * y.d)


class S:
    """exp(a k^2 + b ln k) * m  (a, b exact rationals; m an interval dual)."""
    __slots__ = ('a', 'b', 'm', 'k')

    def __init__(self, a, b, m, k): self.a, self.b, self.m, self.k = Fr(a), Fr(b), Dl.c(m), k

    def _c(self, x): return x if isinstance(x, S) else S(0, 0, x, self.k)

    def _ratio(self, da, db):
        if da == 0 and db == 0: return Dl(I(1), I(0))
        e = Dl(I(0), I(0))
        if da: e = e + Dl(I(D(da.numerator)) / I(D(da.denominator)), I(0)) * self.k.sq()
        if db: e = e + Dl(I(D(db.numerator)) / I(D(db.denominator)), I(0)) * self.k.ln()
        return e.exp()

    def __mul__(x, y): y = x._c(y); return S(x.a + y.a, x.b + y.b, x.m * y.m, x.k)
    __rmul__ = __mul__
    def __truediv__(x, y): y = x._c(y); return S(x.a - y.a, x.b - y.b, x.m / y.m, x.k)
    def __rtruediv__(x, y): return x._c(y) / x
    def __neg__(x): return S(x.a, x.b, -x.m, x.k)

    def __add__(x, y):
        y = x._c(y)
        if (y.a, y.b) > (x.a, x.b): x, y = y, x
        return S(x.a, x.b, x.m + y.m * x._ratio(y.a - x.a, y.b - x.b), x.k)
    __radd__ = __add__
    def __sub__(x, y): return x + (-x._c(y))
    def __rsub__(x, y): return x._c(y) - x

    def sq(x): return S(2 * x.a, 2 * x.b, x.m.sq(), x.k)
    def sqrt(x): return S(x.a / 2, x.b / 2, x.m.sqrt(), x.k)
    def p15(x): return x * x.sqrt()
    def plain(x): return x.m * x._ratio(x.a, x.b)


def h_of(k):
    """h(k) with k an interval dual (value enclosure and derivative seed)."""
    one = lambda m: S(0, 0, m, k)
    sig = S(-4, -11, 1, k); ell = S(-2, -6, 2, k); Lw = S(-1, -4, 2, k); Lv = S(-2, -8, 2, k)
    t = 4 * k.sq() + 11 * k.ln()
    rho = (sig.sq() / Lv.sq()).plain()                         # sigma^2/Lv^2 = k^-6 e^{-4k^2}/4 (tiny)
    V = Lv.sq() * one(1 + rho); L2 = Lw.sq() + sig.sq()
    Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    qb = S(Fr(-1, 2), 0, (k.sq() * rho / (2 * (1 + rho))).exp(), k)    # exp(-k^2 Lv^2/(2V)), c = 1
    q0 = qb * (Sb * L2 / (Lw.sq() * (Sb + sig.sq()))).sqrt(); s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2)
    x = S(0, 2, (1 / (1 + rho)).sq(), k)                         # (k Lv^2/V)^2
    s = s2.sqrt(); L = L2.sqrt(); s2g = sig.sq(); b = s2g / V; E1 = Dl(I(1).exp(), I(0))
    qbp = qb.plain(); u = qbp.sq(); kq = one(_Fu(u))
    g15 = (1 + 2 * L2 / s2).p15()
    B2 = q0.sq() * L2 / (s2.sq() * g15); Iq = q0.sq() * s / (s2 + 2 * L2).sqrt()
    Jww_up = 1 / L2 + kq * B2; Wp_lo = x * Iq / 2; Wp_up = x * Iq / one(1 + (1 - u).sqrt()); den_lo = b + Wp_lo; Jper_up = Wp_up / s2g
    bp = b.plain(); a2 = (1 - bp).sq() / 2; y = a2 * V.plain() / 2; e = (-y).exp(); ome = V * one(a2 / 2 * _psi(y))
    oq = one(1 - qbp)
    xi = sig * x.sqrt() * qb / oq; Xi = xi.sq() + s2g * s2g * qb.sq() / (one(E1) * s2 * oq.sq())
    r = (3 * (xi.sq() + s2g * s2g / L2 + s2g * qb.sq() * (s2g / s2) / (one(E1) * oq.sq()))).sqrt()
    gam = qb * (x * (1 + qb) + x.sqrt()) / oq.sq()
    st2w2 = (x / s2g) * q0.sq() * L2 / g15 / one(1 + (1 - u).sqrt())
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2.sq()
    st2R2 = (st2w2.sqrt() + st2Rw.sqrt()).sq(); omC = (L + r).sq() / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + one(a2 * e) + gam * omC
    Uv = (2 * s2g * Jww_up / (sig * den_lo) + s2g / 2 * Esv2phi / (sig * den_lo)).plain()
    return Uv * t.sqrt() - 1 - C74 * t.ln() / t, Uv, t


def cell_bound(a, b):
    """rigorous upper bound of h on [a, b] (Decimal endpoints); also h(m) (its upper end)."""
    m = (a + b) / 2; rad = (b - a) / 2
    hm = h_of(Dl(I(m), I(0)))[0].v; hd = h_of(Dl(I(a, b), I(1)))[0]
    return min(hm.hi + hd.d.mag() * rad, hd.v.hi), hm.hi


def t_of(k):
    return 4 * I(k).sq() + 11 * I(k).ln()


if __name__ == '__main__':
    import numpy as np
    from az_upper import U_main_dec
    ln10 = I(10).ln(); tlo, thi = 24 * ln10, 100000 * ln10
    k_lo, k_hi = D('3.25'), D('239.9')
    assert t_of(k_lo).hi <= tlo.lo and t_of(k_hi).lo >= thi.hi, (t_of(k_lo), tlo, t_of(k_hi), thi)
    # the formula is the same as az_upper's decimal version
    agree = []
    for kk in ('3.25', '5', '20', '100', '239.9'):
        _, Uv, t = h_of(Dl(I(D(kk)), I(0))); tt = float(t.v.mid()); ud = U_main_dec(tt)[0] / np.sqrt(tt)
        agree.append(abs(float(Uv.v.mid()) / ud - 1))
    print('agreement with az_upper.U_main_dec: max rel %.1e' % max(agree), flush=True)
    t0 = time.time(); a = k_lo; ncell = 0; nbis = 0; worst = (D(-10), None); rec = []; step = D('0.02')
    while a < k_hi:
        b = min(a + step * (1 + a / 10), k_hi); stack = [(a, b)]
        while stack:
            u_, v_ = stack.pop(); ub, hm = cell_bound(u_, v_)
            if ub < 0:
                ncell += 1
                if ub > worst[0]: worst = (ub, float(u_))
            else:
                assert v_ - u_ > D('1e-9'), ('cannot certify', u_, v_, ub)
                w = (u_ + v_) / 2; stack += [(w, v_), (u_, w)]; nbis += 1
        if not rec or float(a) >= rec[-1][0] * 1.2: rec.append((float(a), float(hm), float(ub)))
        a = b
        if ncell % 500 == 0: print('k %.3f  cells %d  bisections %d  worst bound %.3e  %.0fs' % (float(a), ncell, nbis, worst[0], time.time() - t0), flush=True)
    out = {'claim': "h = U sqrt(ln) - 1 - 7.4 lnln/ln < 0 for 1e-100000 <= sigma <= 1e-24 on the sequence S'", 'k_range': [float(k_lo), float(k_hi)],
           't_range_checked': [float(t_of(k_lo).hi), float(t_of(k_hi).lo)], 'cells': ncell, 'bisections': nbis, 'worst_upper_bound_of_h': float(worst[0]),
           'worst_at_k': worst[1], 'samples_k_h_mid_h_upper': rec, 'precision_digits': 60, 'agreement_with_U_main_dec': max(agree),
           'seconds': time.time() - t0, 'all_cells_certified': True}
    json.dump(out, open('results/ba_cert.json', 'w'), indent=1)
    print('certified: %d cells, %d bisections, max upper bound of h = %.4e (at k = %.3f), %.0f s' % (ncell, nbis, worst[0], worst[1], time.time() - t0))
