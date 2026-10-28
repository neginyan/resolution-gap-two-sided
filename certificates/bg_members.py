"""Stage BG: the comb sequence S'_C is in the class of Theorem BF1 on the whole certified range — (P3) Qbar <= 1/4, (P4) rho = s^2/(2L^2) <= 1/8,
(P5) x >= 1 — checked in interval arithmetic on cells covering k0 in [3.43, 240] (the same exact-monomial class S as bf_cert_upper.py; every enclosure
covers the whole cell).  (P1), (P2) hold by construction; (P8) is checked in floating point on the dense grid of bf_upper.py (F_c/(1 - |eps|) >= 1 - 3e-14 for ln(1/sigma) <= 345;
the small terms of Lemma AY2-P are products of the same monomials and decrease along the sequence).  -> results/bg_members.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
from fractions import Fraction as Fr
from decimal import Decimal as D
from ba_ivl import I, Dl
from ba_cert import S
from bf_cert_upper import EP


def quantities(k):
    one = lambda m: S(0, 0, m, k)
    sig = S(-4, -11, 256, k); ell = S(-2, -6, 32, k); Lw = S(-1, -4, 8, k); Lv = S(-2, -8, 32, k)
    rho_v = (sig.sq() / Lv.sq()).plain(); L2 = Lw.sq() + sig.sq(); Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    a1 = 2 * (-(1 / (2 * k.sq().sq()))).exp()
    qb = S(Fr(-1, 2), 0, a1 * (k.sq() * rho_v / (2 * (1 + rho_v))).exp(), k)
    Q = (qb * one(Dl(I(1) + EP[0], I(0)))).plain()
    s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2); varrho = (s2 / (2 * L2)).plain()
    x = (S(0, 2, (1 / (1 + rho_v)).sq(), k)).plain()
    return Q.v, varrho.v, x.v


if __name__ == '__main__':
    a = D('3.43'); kh = D('240.0'); step = D('0.02'); n = 0; Qmax = D(0); rmax = D(0); xmin = D(10 ** 9)
    while a < kh:
        b = min(a + step * (1 + a / 10), kh); Q, r, x = quantities(Dl(I(a, b), I(0)))
        Qmax = max(Qmax, Q.hi); rmax = max(rmax, r.hi); xmin = min(xmin, x.lo); n += 1; a = b
    out = {'cells': n, 'Qbar_max': float(Qmax), 'rho_max': float(rmax), 'x_min': float(xmin), 'P3': bool(Qmax <= D('0.25')), 'P4': bool(rmax <= D('0.125')), 'P5': bool(xmin >= 1)}
    print(out); json.dump(out, open('results/bg_members.json', 'w'), indent=1)
