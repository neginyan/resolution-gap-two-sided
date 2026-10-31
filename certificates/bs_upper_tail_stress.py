"""Stage BS §2.1, numerical check (not part of the proof): direct high-precision evaluation (Decimal, 300 digits, values carried as m * E^g so that
nothing underflows) of every intermediate of bf_cert_upper.h_of on S'_C, compared with the asymptotic forms of bs_upper_tail.U_form, at k from 240 to
1e8, with the harmonic parameters at 0 and at their largest tail bounds.  Written by an independent review agent (stage BS).
-> results/bs_upper_tail_stress.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import sys
from fractions import Fraction as Fr
import decimal
from decimal import Decimal as D
import bs_upper_tail as BS
from bs_upper_tail import AF

CTX = decimal.Context(prec=300, Emin=-999999999, Emax=999999999)


class X:
    def __init__(s, m, g=0): s.m, s.g = (m if isinstance(m, D) else CTX.create_decimal(m)), Fr(g)
    @staticmethod
    def c(v): return v if isinstance(v, X) else X(v)
    def _al(s, o, sign):
        o = X.c(o); o = X(CTX.multiply(sign, o.m), o.g)
        a, b = (s, o) if s.g <= o.g else (o, s)
        dg = b.g - a.g
        f = CTX.exp(CTX.divide(CTX.multiply(-dg.numerator, K2), dg.denominator)) if dg else D(1)
        return X(CTX.add(a.m, CTX.multiply(b.m, f)), a.g)
    def __add__(s, o): return s._al(o, 1)
    __radd__ = __add__
    def __sub__(s, o): return s._al(o, -1)
    def __rsub__(s, o): return X.c(o)._al(s, -1)
    def __mul__(s, o): o = X.c(o); return X(CTX.multiply(s.m, o.m), s.g + o.g)
    __rmul__ = __mul__
    def __truediv__(s, o): o = X.c(o); return X(CTX.divide(s.m, o.m), s.g - o.g)
    def __rtruediv__(s, o): return X.c(o) / s
    def sq(s): return s * s
    def sqrt(s): return X(CTX.sqrt(s.m), s.g / 2)
    def p15(s): return s * s.sqrt()


def kp(b): return CTX.power(KD, CTX.create_decimal(b.numerator) / b.denominator) if b else D(1)


def direct(k, ep_max):
    global K2, KD
    KD = CTX.create_decimal(k); K2 = CTX.multiply(KD, KD)
    M = lambda c, b, g: X(CTX.multiply(CTX.create_decimal(c), kp(Fr(b))), g)
    sig = M(256, -11, 4); ell = M(32, -6, 2); Lw = M(8, -4, 1); Lv = M(32, -8, 2)
    rho = sig.sq() / Lv.sq(); V = Lv.sq() * (1 + rho); L2 = Lw.sq() + sig.sq()
    Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    a1 = X(2 * CTX.exp(-1 / (2 * K2 * K2)))
    w = X(K2) * rho / (2 * (1 + rho)); expw = 1 + w + w.sq() / 2
    qb = a1 * X(1, Fr(1, 2)) * expw
    q0 = qb * (Sb * L2 / (Lw.sq() * (Sb + sig.sq()))).sqrt(); s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2)
    x = X(K2) / (1 + rho).sq()
    s = s2.sqrt(); L = L2.sqrt(); s2g = sig.sq(); b = s2g / V; E1 = X(CTX.exp(1))
    Ep = (1 + X(8, Fr(14985, 10000))) if ep_max else X(1); Zp = (1 + X(8, Fr(2997, 1000))) if ep_max else X(1)
    Q = qb * Ep; N1 = qb * Ep; N2 = qb * Ep; Sq0 = q0.sq() * Zp; Sq2 = q0.sq() * Zp
    oQ = 1 - Q; pQ = 1 + Q
    g15 = (1 + 2 * L2 / s2).p15(); geo = s / (s2 + 2 * L2).sqrt()
    u0 = 4 * L2.sq() * Sq0 / (s2.sq() * g15)
    Jww_up = (1 + u0 / (8 * oQ)) / L2
    I2_lo = q0.sq() * geo; I2_up = Sq2 * geo
    den_lo = b + x * I2_lo / (2 * pQ); Jper_up = x * I2_up / (2 * oQ * s2g)
    a2 = (1 - b).sq() / 2; y = a2 * V / 2
    e = 1 - y + y.sq() / 2; ome = y - y.sq() / 2          # 1 - e^{-y}, y ~ 1e-(1e5): higher terms invisible
    xi = sig * x.sqrt() * N1 / oQ; Xi = xi.sq() + s2g * s2g * Q.sq() / (E1 * s2 * oQ.sq())
    r = (3 * (xi.sq() + s2g * s2g / L2 + s2g * Q.sq() * (s2g / s2) / (E1 * oQ.sq()))).sqrt()
    gam = (x * (N2 * pQ + N1.sq()) + x.sqrt() * N1 * (1 + 2 * Q) / E1.sqrt()) / oQ.sq()
    v = L2 * s2 / (s2 + 2 * L2); st2w2 = Jper_up * v
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2.sq()
    st2R2 = (st2w2.sqrt() + st2Rw.sqrt()).sq()
    omC = (L + r).sq() / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + a2 * e + gam * omC
    U = 2 * s2g * Jww_up / (sig * den_lo) + s2g / 2 * Esv2phi / (sig * den_lo)
    return dict(locals())


def check(af, xv, k):
    """z := xv/(c k^beta E^gamma) - 1; need z k^nu in R (for some c in civ)."""
    if xv.g != af.gamma: return 'GAMMA MISMATCH %s vs %s' % (xv.g, af.gamma)
    ratio = CTX.divide(xv.m, kp(af.beta))
    c = af.civ(); knu = kp(af.nu)
    # z k^nu range over c in [c.lo, c.hi]
    vals = [CTX.multiply(CTX.subtract(CTX.divide(ratio, cc), 1), knu) for cc in (c.lo, c.hi)]
    lo, hi = min(vals), max(vals)
    ok = hi >= af.R.lo and lo <= af.R.hi
    return ok, float(lo), float(hi), float(af.R.lo), float(af.R.hi), str(af.nu)


if __name__ == '__main__':
    U, _ = BS.U_form(); loc = dict(BS.LOCALS)
    bad = 0; nchk = 0; rows = []
    for k in sys.argv[1:] or ['240', '240.0001', '250', '400', '1000', '3000', '10000', '100000', '1000000', '100000000']:
        for epm in (False, True):
            dv = direct(D(k), epm)
            for name, af in loc.items():
                if name not in dv: continue
                res = check(af, dv[name], D(k)); nchk += 1
                if isinstance(res, str) or not res[0]:
                    bad += 1; print('FAIL k=%s epmax=%s %s %s' % (k, epm, name, res))
            r = check(U, dv['U'], D(k)); rows.append({'k': k, 'harmonics_at_max': epm, 'U_zk2': [r[1], r[2]], 'R_U': [r[3], r[4]], 'inside': bool(r[0])})
            print('k=%s epmax=%s U: zk^nu=[%.9g,%.9g] R=[%.9g,%.9g]' % ((k, epm) + r[1:5]), flush=True)
    out = {'checks': nchk, 'failures': bad, 'U_rows': rows, 'ok': bad == 0 and all(r['inside'] for r in rows)}
    print('checks', nchk, 'failures', bad)
    json.dump(out, open(os.environ.get('BS_OUT', 'results/bs_upper_tail_stress.json'), 'w'), indent=1)
