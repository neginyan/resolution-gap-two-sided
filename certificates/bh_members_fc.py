"""Stage BH: (P8) along the comb sequence S'_C on the whole certified range, in interval arithmetic (the same 1464 cells and exact-monomial class S
as bf_cert_upper.py).  Lemma AY2-P gives F_c/(1 - |eps|) >= (1 - D1)(1 - dH)(1 - rC) sqrt(1 - rw) >= 1 - D, D = D1 + dH + rC + rw,
D1 = J_ww,up/J_vv,lo (>= 1 - A1).  (P8) F_c >= 1.02 G with |eps| <= 0.1 and G <= 0.134 holds if D <= 1 - 1.02 x 0.134/0.9 = 0.848.  Here: max D over the cells.
-> results/bh_members_fc.json.  Python 3.11."""
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
from bf_cert_upper import EP, ZP


def deficit(k):
    one = lambda m: S(0, 0, m, k); par = lambda iv: one(Dl(I(1) + iv, I(0)))
    sig = S(-4, -11, 256, k); ell = S(-2, -6, 32, k); Lw = S(-1, -4, 8, k); Lv = S(-2, -8, 32, k)
    rho = (sig.sq() / Lv.sq()).plain(); V = Lv.sq() * one(1 + rho); L2 = Lw.sq() + sig.sq(); Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    a1 = 2 * (-(1 / (2 * k.sq().sq()))).exp(); qb = S(Fr(-1, 2), 0, a1 * (k.sq() * rho / (2 * (1 + rho))).exp(), k)
    q0 = qb * (Sb * L2 / (Lw.sq() * (Sb + sig.sq()))).sqrt(); s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2); x = S(0, 2, (1 / (1 + rho)).sq(), k)
    s2g = sig.sq(); b = s2g / V; E1 = Dl(I(1).exp(), I(0))
    Q = qb * par(EP[0]); N1 = qb * par(EP[1]); Sq0 = q0.sq() * par(ZP[0]); Sq2 = q0.sq() * par(ZP[2])
    Qp = Q.plain(); oQ = one(1 - Qp); pQ = one(1 + Qp)
    g15 = (1 + 2 * L2 / s2).p15(); geo = s2.sqrt() / (s2 + 2 * L2).sqrt(); u0 = 4 * L2.sq() * Sq0 / (s2.sq() * g15)
    Jww_up = (1 + u0 / (8 * oQ)) / L2; Jww_lo = 1 / L2
    I2_lo = q0.sq() * geo; I2_up = Sq2 * geo
    Jvv_lo = 1 / V + x * I2_lo / (2 * pQ * s2g); Jper_up = x * I2_up / (2 * oQ * s2g)
    D1 = (Jww_up / Jvv_lo).plain()
    sos2 = s2g / s2; l4 = (s2g / L2).sq()
    dH = (s2g / L2 + sos2 * (Q / oQ + Q.sq() / (one(E1) * oQ.sq())) + x.sqrt() * sos2.sqrt() * N1 / (one(E1.sqrt()) * oQ.sq())).plain()
    xi = sig * x.sqrt() * N1 / oQ; Xi = xi.sq() + s2g * s2g * Q.sq() / (one(E1) * s2 * oQ.sq())
    v = L2 * s2 / (s2 + 2 * L2); bp = b.plain()
    rC = (one((1 - bp).sq()) * (3 + V * Jper_up) + 2 * (1 + 3 * l4) * (L2 / V + Jper_up * v)) / (8 * Jvv_lo) + 6 * Xi / 8
    win = 3 * v.sq() * Sq0 * geo / (2 * oQ * s2.sq())
    rw = V / 8 + 2 * (1 + 3 * l4) * (3 + win) / (8 * Jww_lo) + 6 * Xi / 8
    return D1.v, dH.v, rC.plain().v, rw.plain().v


if __name__ == '__main__':
    a = D('3.43'); kh = D('240.0'); step = D('0.02'); n = 0; Dmax = D(0); parts = [D(0)] * 4
    while a < kh:
        b = min(a + step * (1 + a / 10), kh); t = deficit(Dl(I(a, b), I(0))); tot = sum(x.hi for x in t)
        if tot > Dmax: Dmax = tot; parts = [x.hi for x in t]
        n += 1; a = b
    out = {'cells': n, 'D_max': float(Dmax), 'parts_at_max': [float(p) for p in parts], 'threshold': 1 - 1.02 * 0.134 / 0.9, 'P8_on_S_C': bool(Dmax <= D('0.848'))}
    print(out); json.dump(out, open('results/bh_members_fc.json', 'w'), indent=1)
