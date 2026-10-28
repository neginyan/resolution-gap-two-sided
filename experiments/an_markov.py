"""Stage AN: growth of the leading constants B_n (and of the Lambda constant, integrand sech (2F' - F tanh)^2) = 2 sup { int sech (F' - F tanh)^2 / int F^2 sech : F polynomial of degree <= n } (alpha = 0),
computed stably with orthonormal polynomials for the weight sech (Stieltjes procedure on a fine grid).  -> results/an_markov.json.
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.linalg import eigh

x = np.linspace(-400, 400, 1600001); h = x[1] - x[0]
lw = -np.logaddexp(x, -x) + np.log(2.0)            # log sech, stable
w = np.exp(lw); th = np.tanh(x)
N = 60
P = np.zeros((N + 1, x.size)); dP = np.zeros_like(P)
P[0] = 1 / np.sqrt(np.sum(w) * h); prev = np.zeros_like(x); dprev = np.zeros_like(x); b = 0.0
for k in range(N):                                   # three-term recurrence (weight even: a_k = 0)
    q = x * P[k] - b * prev; dq = P[k] + x * dP[k] - b * dprev
    nb = np.sqrt(np.sum(w * q * q) * h); prev, dprev = P[k], dP[k]; P[k + 1] = q / nb; dP[k + 1] = dq / nb; b = nb
out = {}
for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 50, 60):
    R = dP[:n + 1] - P[:n + 1] * th[None]
    Q = (R * w[None]) @ R.T * h; G = (P[:n + 1] * w[None]) @ P[:n + 1].T * h
    ev = eigh(Q, G, eigvals_only=True)
    R2 = 2 * dP[:n + 1] - P[:n + 1] * th[None]; Q2 = (R2 * w[None]) @ R2.T * h; ev2 = eigh(Q2, G, eigvals_only=True)
    out[str(n)] = {'B_n': float(2 * ev[-1]), 'Lambda_over_ln_n': float(2 * ev2[-1])}       # tR/ln(1/tJ) and Lambda/ln(1/tJ) -> these, D -> inf
    print('n %3d: B_n = %.4f   Lambda/ln -> %.4f   (orthonormality error %.1e)' % (n, 2 * ev[-1], 2 * ev2[-1], np.max(np.abs(G - np.eye(n + 1)))), flush=True)
json.dump(out, open('results/an_markov.json', 'w'), indent=1)
