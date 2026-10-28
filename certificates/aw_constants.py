"""Stage AW: the explicit constant of Theorem AW5 — sup over the ripple wavenumber of the two-branch bound of Proposition AW4.
B(L/s) = sup_{x >= x0} min( x (1 + K1(q(x)) q(x)^2),  x K2(q(x), u(x)) u(x) ),  q(x) = e^{-x/2} (>= qbar for every c <= 1), u(x) = sqrt2 q(x)^2 L/s
(branch (ii) only where u <= 1; K1, K2 increase with qbar, so q(x) = e^{-x/2} is the worst case).  Then k_eff^2 <= B(L/s) + 1/4 (+ e^{-k'^2V/2}-small).
-> results/aw_constants.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aw_lemmas import K1, K2

SQ2 = np.sqrt(2.0)


def tables(qmax, rho, nx=6001, xmax=700.0):
    xs = np.linspace(-2 * np.log(qmax), xmax, nx); q = np.exp(-xs / 2)
    k1 = np.array([K1(a, rho, 60) for a in q])
    # K2 depends on u only through the denominator sqrt(1-q) - q sqrt(u)/(4(1-q)^{3/2}); use u = 1 (worst) for a bound valid for all u <= 1
    k2 = np.array([K2(a, rho, 1.0, 60) for a in q])
    return xs, q, k1, k2


def B(Ls, xs, q, k1, k2):
    u = SQ2 * q * q * Ls; br1 = xs * (1 + k1 * q * q); br2 = np.where(u <= 1, xs * k2 * u, np.inf)
    return float(np.max(np.minimum(br1, br2)))


if __name__ == '__main__':
    out = {}
    for qmax, rho in ((0.25, 0.01), (0.25, 0.125), (0.5, 0.01)):
        xs, q, k1, k2 = tables(qmax, rho)
        Ls = np.exp(np.linspace(np.log(10.0), 250 * np.log(10), 600)); Bv = np.array([B(l, xs, q, k1, k2) for l in Ls]); exc = Bv - np.log(Ls)
        big = Ls >= 1e24
        r = {'qmax': qmax, 'rho_max': rho, 'K1_at_qmax': float(K1(qmax, rho)), 'K2_at_qmax': float(K2(qmax, rho, 1.0)), 'sup_x_K1_q2': float(np.max(xs * k1 * q * q)),
             'sup_excess_all': float(exc.max()), 'sup_excess_Ls_ge_1e24': float(exc[big].max()), 'B_floor': float(Bv.min())}
        out['q%.2f_rho%.3f' % (qmax, rho)] = r
        print('qbar <= %.2f, rho <= %.3f: K1(qmax) %.2f, K2(qmax) %.3f, sup x K1 q^2 = %.3f;  B(L/s) - ln(L/s) <= %.3f for all L/s >= 10, <= %.3f for L/s >= 1e24 (floor %.2f)' % (
            qmax, rho, r['K1_at_qmax'], r['K2_at_qmax'], r['sup_x_K1_q2'], r['sup_excess_all'], r['sup_excess_Ls_ge_1e24'], r['B_floor']), flush=True)
    json.dump(out, open('results/aw_constants.json', 'w'), indent=1)
