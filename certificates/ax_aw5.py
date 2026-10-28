"""Stage AX: Theorem AW5 with the branch of Proposition AX4 (S <= 8 kappa2 u/(8+u), valid for every u) — the improved constant, and the
explanation of the difference to the main line's independent computation (stage AW used branch (ii) only where u <= 1).  -> results/ax_aw5.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aw_lemmas import K1

SQ2 = np.sqrt(2.0)


def k2(q, rho): return (1 + rho + q / (2 * (1 - q))) ** 2 / (8 * (1 - q))


def Bfun(Ls, rho=0.125, qmax=0.25, nx=6001):
    xs = np.linspace(-2 * np.log(qmax), 700, nx); q = np.exp(-xs / 2)
    k1 = np.array([K1(a, rho, 60) for a in q]); br1 = xs * (1 + k1 * q * q)
    out = []
    for L in Ls:
        u = SQ2 * q * q * L; br2 = xs * 8 * k2(q, rho) * u / (8 + u); out.append(float(np.max(np.minimum(br1, br2))))
    return np.array(out)


if __name__ == '__main__':
    Ls = np.exp(np.linspace(np.log(10.0), 250 * np.log(10), 600)); B = Bfun(Ls) + 0.25; exc = B - np.log(Ls)
    r = {'sup_excess_all_Ls_ge_10': float(exc.max()), 'sup_excess_Ls_ge_1e6': float(exc[Ls >= 1e6].max()), 'sup_excess_Ls_ge_1e24': float(exc[Ls >= 1e24].max()),
         'excess_at_10': float(exc[0])}
    print('Theorem AW5 with Proposition AX4: k_eff^2 - ln(L/s) <= %.3f (L/s >= 10), %.3f (>= 1e6), %.3f (>= 1e24); at L/s = 10: %.3f' % (
        r['sup_excess_all_Ls_ge_10'], r['sup_excess_Ls_ge_1e6'], r['sup_excess_Ls_ge_1e24'], r['excess_at_10']))
    json.dump(r, open('results/ax_aw5.json', 'w'), indent=1)
