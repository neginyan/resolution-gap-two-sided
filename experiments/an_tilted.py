"""Stage AN: the tilted, offset hidden dipole — B' = 1 is NOT the optimal slope of the signed problem (S1).

Leading order (D -> infinity, xi = D (y - y_b), y_b the valley bottom): hidden sources give g = F(xi) sech(xi) with F an exponential sum of
rates in (-1, 1), and  Rq/D^2 -> rho[F] = int sech (F' - F tanh)^2 / int F^2 sech,  ln(M/I)/D^2 -> 1/2.
 * F = xi - xi0 (dipole at the valley centre, zero offset by xi0):  2 rho = 1 exactly for every xi0
   (numerator pi^3/8 + xi0^2 pi/2, denominator pi^3/4 + xi0^2 pi).
 * F = e^{alpha xi}(xi - xi0) (dipole displaced to c = alpha D, zero offset by xi0): 2 rho(alpha, xi0) reaches 1.13.
Exact check: lines at -D, +D (unit weights), sources +1 at c - d and -1 at c + d (d = 1), weights set so that the zero of g is at
y0 = xi0/D and the larger source term is e^{-z} below the lines at y0.  -> results/an_tilted.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from an_signed1d import quantities


def two_rho(a, x0):
    num = quad(lambda u: np.exp(2 * a * u) / np.cosh(u) * (1 + a * (u - x0) - (u - x0) * np.tanh(u)) ** 2, -200, 200, limit=800, points=[x0])[0]
    den = quad(lambda u: np.exp(2 * a * u) * (u - x0) ** 2 / np.cosh(u), -200, 200, limit=800, points=[x0])[0]
    return 2 * num / den


def config(D, alpha, xi0, d=1.0, z=2.0):
    c = alpha * D; y0 = xi0 / D; x1, x2 = c - d, c + d
    # log of the line terms at y0 (two unit lines at -D, +D): ln[phi(y0 + D) + phi(y0 - D)]
    lL = np.logaddexp(-(y0 + D) ** 2 / 2, -(y0 - D) ** 2 / 2)
    # source terms w_i phi(y - x_i) equal at y0, each e^{-z} times the line term there
    lw1 = lL - z + (y0 - x1) ** 2 / 2; lw2 = lL - z + (y0 - x2) ** 2 / 2
    return np.array([-D, D, x1, x2]), np.array([0.0, 0.0, lw1, lw2]), np.array([0.0, 0.0, 1.0, -1.0])


if __name__ == '__main__':
    out = {}
    # identity alpha = 0
    out['alpha0'] = {'%g' % x0: two_rho(0.0, x0) for x0 in (0.0, 0.7, 1.5, 3.0, 5.0)}
    print('alpha = 0: 2 rho =', ', '.join('%.12f' % v for v in out['alpha0'].values()))
    r = minimize(lambda p: -two_rho(p[0], p[1]), [0.2, 3.0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-10})
    out['sup_tilted_dipole'] = {'two_rho': -r.fun, 'alpha': r.x[0], 'xi0': r.x[1]}
    print('sup over (alpha, xi0) of 2 rho = %.5f at alpha = %.4f, xi0 = %.4f' % (-r.fun, r.x[0], r.x[1]))
    a, x0 = r.x
    out['exact'] = {}
    for D in (10.0, 20.0, 40.0, 60.0, 80.0):
        best = None
        for z in (1.0, 2.0, 3.0):
            xs, lw, f = config(D, a, x0, 1.0, z); o = quantities(xs, lw - lw.max(), f)
            rr = {'z': z, 'Rq': o['Rq'], 'lnMI': o['lnMI'], 'ratio': o['Rq'] / o['lnMI'], 'margin': o['Rq'] - o['lnMI']}
            if best is None or rr['margin'] > best['margin']: best = rr
        out['exact']['%g' % D] = best
        print('D %4g: Rq %.2f ln(M/I) %.2f ratio %.4f margin %.2f (z = %g)' % (D, best['Rq'], best['lnMI'], best['ratio'], best['margin'], best['z']), flush=True)
    json.dump(out, open('results/an_tilted.json', 'w'), indent=1, default=float)
