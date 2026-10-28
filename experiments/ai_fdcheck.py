"""Stage AI: third, crude-but-independent check: s_w and H by 4th-order finite differences of ln p (ln p from the mixture density
formula directly), plain Riemann sums.  For D = 8, z = 0: one line + point and two lines + midpoint.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp
LN2PI = np.log(2 * np.pi)

def lnp(V, W, D, L, two):
    a = [-V ** 2 / 2 - 0.5 * LN2PI, -L - ((V - D) ** 2 + W ** 2) / 2 - LN2PI]
    if two: a.append(-(V - 2 * D) ** 2 / 2 - 0.5 * LN2PI)
    return logsumexp(np.stack(a), axis=0)

def run(D, z, two, h=0.01, e=0.01):
    L = D * (z + D / 2); yv = np.arange(-6, (2 * D + 6) if two else D + abs(z) + 10, h); yw = np.arange(-8, 8, h)
    V, W = np.meshgrid(yv, yw, indexing='ij'); f = lambda dv, dw: lnp(V + dv, W + dw, D, L, two)
    c = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]; c2 = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]; o = [-2, -1, 0, 1, 2]
    sw = sum(ci * f(0, k * e) for ci, k in zip(c, o) if ci) / e
    Hww = -sum(ci * f(0, k * e) for ci, k in zip(c2, o)) / e ** 2
    Hvw = -sum(ci * cj * f(k * e, m * e) for ci, k in zip(c, o) for cj, m in zip(c, o) if ci and cj) / e ** 2
    l0 = f(0, 0); m = l0.max(); P = np.exp(l0 - m)
    J = np.sum(P * sw ** 2); G = np.sum(P * (Hvw ** 2 + Hww ** 2))
    lnM = np.log(2.0 + np.exp(-L)) if two else np.log(1 + np.exp(-L))
    return G / J, lnM - (np.log(J) + m + 2 * np.log(h))

if __name__ == '__main__':
    print('one line + point D=8 z=0: tR %.4f ln %.4f (closed form 5.0246, 32.953)' % run(8.0, 0.0, False))
    print('two lines + point D=8 z=0: tR %.4f ln %.4f (closed form 31.0092, 37.2829)' % run(8.0, 0.0, True))
