"""Stage AY: certified constant of Theorem AW5 (with Proposition AX4): sup_x min(x(1 + K1(q)q^2), x 8 kappa2(q) u/(8+u)) + 1/4 - ln(L/s) over
10 <= L/s <= 1e250, bounded cell by cell in x (x-factors increasing, q-factors increasing in q = min(1/4, e^{-x/2}), u = sqrt2 q^2 L/s decreasing in x,
u/(8+u) increasing in u) and in L/s (B increasing in L/s: on [l_j, l_{j+1}], B(L/s) - ln(L/s) <= B(l_{j+1}) - ln(l_j)).  -> results/ay_aw5_cert.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aw_lemmas import K1

SQ2 = np.sqrt(2.0); RHO = 0.125


def k2(q): return (1 + RHO + q / (2 * (1 - q))) ** 2 / (8 * (1 - q))


if __name__ == '__main__':
    xs = np.linspace(-2 * np.log(0.25), 700.0, 70001); a, b = xs[:-1], xs[1:]; qa = np.minimum(0.25, np.exp(-a / 2))
    k1 = np.array([K1(t, RHO, 60) for t in qa[::1]])
    br1 = b * (1 + k1 * qa * qa)
    lgrid = np.exp(np.linspace(np.log(10.0), 250 * np.log(10), 4001))
    Bs = []
    for Ls in lgrid:
        u = SQ2 * qa * qa * Ls; br2 = b * 8 * k2(qa) * u / (8 + u); Bs.append(float(np.max(np.minimum(br1, br2))) + 0.25)
    Bs = np.array(Bs); exc = Bs[1:] - np.log(lgrid[:-1])
    big = lgrid[:-1] >= 1e6
    r = {'certified_sup_excess_Ls_ge_10': float(exc.max()), 'certified_sup_excess_Ls_ge_1e6': float(exc[big].max())}
    print('certified: k_eff^2 - ln(L/s) <= %.3f (10 <= L/s <= 1e250), <= %.3f (L/s >= 1e6)' % (r['certified_sup_excess_Ls_ge_10'], r['certified_sup_excess_Ls_ge_1e6']))
    json.dump(r, open('results/ay_aw5_cert.json', 'w'), indent=1)
