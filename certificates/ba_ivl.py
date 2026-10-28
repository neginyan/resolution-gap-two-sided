"""Stage BA: rigorous interval arithmetic in Python decimal (no external packages) and interval dual numbers (value + derivative enclosure).

Every elementary operation is computed in decimal with `PREC` digits (correctly rounded: +, -, *, /, sqrt, exp, ln are correctly rounded
in Python's decimal module) and the result is widened outward by the relative amount U = 10^-(PREC-3), which exceeds one ulp; a result 0 is
exact (only an exact cancellation or a zero factor produces it).  expm1 / log1p use their Taylor series with an explicit remainder for
|y| < 1e-15 (remainders |y|^4/23 and |y|^4/3) and exp(y) - 1, ln(1 + y) otherwise.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
from decimal import Decimal as D, getcontext

PREC = 60
getcontext().prec = PREC
U = D(10) ** (-(PREC - 3))
TINY = D('1e-15')


def _lo(x): return x - abs(x) * U
def _hi(x): return x + abs(x) * U


class I:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        lo = D(lo) if not isinstance(lo, D) else lo
        hi = lo if hi is None else (D(hi) if not isinstance(hi, D) else hi)
        assert lo <= hi, (lo, hi)
        self.lo, self.hi = lo, hi

    @staticmethod
    def c(x): return x if isinstance(x, I) else I(x)

    def __add__(a, b): 
        if isinstance(b, Dl): return NotImplemented
        b = I.c(b); return I(_lo(a.lo + b.lo), _hi(a.hi + b.hi))
    __radd__ = __add__

    def __neg__(a): return I(-a.hi, -a.lo)
    def __sub__(a, b): 
        if isinstance(b, Dl): return NotImplemented
        b = I.c(b); return I(_lo(a.lo - b.hi), _hi(a.hi - b.lo))
    def __rsub__(a, b): return I.c(b) - a

    def __mul__(a, b):
        if isinstance(b, Dl): return NotImplemented
        b = I.c(b); p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return I(_lo(min(p)), _hi(max(p)))
    __rmul__ = __mul__

    def inv(a):
        assert a.lo > 0 or a.hi < 0, 'division by an interval containing 0'
        return I(_lo(1 / a.hi), _hi(1 / a.lo))

    def __truediv__(a, b):
        if isinstance(b, Dl): return NotImplemented
        return a * I.c(b).inv()
    def __rtruediv__(a, b): return I.c(b) * a.inv()
    def __rsub__(a, b): return I.c(b) - a

    def sq(a):
        if a.lo >= 0: return I(_lo(a.lo * a.lo), _hi(a.hi * a.hi))
        if a.hi <= 0: return I(_lo(a.hi * a.hi), _hi(a.lo * a.lo))
        return I(D(0), _hi(max(a.lo * a.lo, a.hi * a.hi)))

    def sqrt(a): assert a.lo >= 0; return I(max(D(0), _lo(a.lo.sqrt())), _hi(a.hi.sqrt()))
    def exp(a): return I(_lo(a.lo.exp()), _hi(a.hi.exp()))
    def ln(a): assert a.lo > 0; return I(_lo(a.lo.ln()), _hi(a.hi.ln()))
    def p15(a): assert a.lo >= 0; return a * a.sqrt()                    # x^{3/2}, monotone

    def em1(a): return I(_em1(a.lo, -1), _em1(a.hi, +1))                 # expm1 is increasing
    def l1p(a): return I(_l1p(a.lo, -1), _l1p(a.hi, +1))                 # log1p is increasing

    def mag(a): return max(abs(a.lo), abs(a.hi))
    def mid(a): return (a.lo + a.hi) / 2
    def __repr__(a): return 'I[%.6e, %.6e]' % (a.lo, a.hi)


def _em1(y, side):
    if abs(y) < TINY:
        v = y + y * y / 2 + y ** 3 / 6; r = y ** 4 / 23
        return _lo(v - r) if side < 0 else _hi(v + r)
    e = y.exp(); e = _lo(e) if side < 0 else _hi(e)
    return _lo(e - 1) if side < 0 else _hi(e - 1)


def _l1p(y, side):
    if abs(y) < TINY:
        v = y - y * y / 2 + y ** 3 / 3; r = y ** 4 / 3
        return _lo(v - r) if side < 0 else _hi(v + r)
    one = 1 + y; one = _lo(one) if side < 0 else _hi(one)
    l = one.ln(); return _lo(l) if side < 0 else _hi(l)


class Dl:
    """interval dual number: value enclosure v, derivative enclosure d (forward mode)."""
    __slots__ = ('v', 'd')

    def __init__(self, v, d): self.v, self.d = I.c(v), I.c(d)

    @staticmethod
    def c(x): return x if isinstance(x, Dl) else Dl(x, I(0))

    def __add__(a, b): b = Dl.c(b); return Dl(a.v + b.v, a.d + b.d)
    __radd__ = __add__
    def __neg__(a): return Dl(-a.v, -a.d)
    def __sub__(a, b): b = Dl.c(b); return Dl(a.v - b.v, a.d - b.d)
    def __rsub__(a, b): return Dl.c(b) - a
    def __mul__(a, b): b = Dl.c(b); return Dl(a.v * b.v, a.d * b.v + a.v * b.d)
    __rmul__ = __mul__
    def inv(a): iv = a.v.inv(); return Dl(iv, -(a.d * iv.sq()))
    def __truediv__(a, b): return a * Dl.c(b).inv()
    def __rtruediv__(a, b): return Dl.c(b) * a.inv()
    def sq(a): return Dl(a.v.sq(), 2 * a.v * a.d)
    def sqrt(a): r = a.v.sqrt(); return Dl(r, a.d / (2 * r))
    def exp(a): e = a.v.exp(); return Dl(e, e * a.d)
    def ln(a): return Dl(a.v.ln(), a.d / a.v)
    def p15(a): r = a.v.sqrt(); return Dl(a.v * r, D('1.5') * r * a.d)
    def em1(a): return Dl(a.v.em1(), a.v.exp() * a.d)
    def l1p(a): return Dl(a.v.l1p(), a.d / (1 + a.v))
