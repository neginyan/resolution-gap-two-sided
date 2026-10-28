"""Stage AN: hidden multipoles — the leading constant of the signed problem grows with the order of the multipole.

Leading order (D -> infinity, xi = D y, two unit lines at -D, +D): n + 1 hidden sources with signs give g = F(xi) sech(xi), F a combination
of e^{r xi} (r = x_i/D), i.e. in the limit of close sources a tilted polynomial F = e^{alpha xi} p(xi), deg p = n, and
  Rq/D^2 -> rho[F] = int sech (F' - F tanh)^2 / int F^2 sech,   ln(M/I) = D^2/2 + O(n ln D).
So the slope needed in (S1) is >= 2 sup_F rho[F] =: B_n.  Part 1 computes B_n (generalised eigenproblem on the basis e^{alpha xi} xi^k).
Part 2 realises the optimal F by n + 1 sources (finite differences, spacing d) with weights w_i >= 0 and signs f_i = +-1, and evaluates
the exact Rq, ln(M/I) at growing D (log-domain evaluator of an_signed1d.py).  -> results/an_multipole.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
from scipy.linalg import eigh
from math import factorial
from an_signed1d import quantities

X = np.linspace(-150, 150, 600001); HX = X[1] - X[0]; SECH = 1 / np.cosh(X); TH = np.tanh(X); SC = 3.0


def leading(alpha, n):
    E = np.exp(alpha * X); B = [E * (X / SC) ** k for k in range(n + 1)]
    dB = [E * (alpha * (X / SC) ** k + ((k / SC) * (X / SC) ** (k - 1) if k > 0 else 0.0)) for k in range(n + 1)]
    G = np.array([[np.sum(SECH * a * b) * HX for b in B] for a in B])
    Q = np.array([[np.sum(SECH * (da - a * TH) * (db - b * TH)) * HX for db, b in zip(dB, B)] for da, a in zip(dB, B)])
    w, v = eigh(Q, G); return 2 * w[-1], v[:, -1] / SC ** np.arange(n + 1)       # coefficients of xi^k


def best_alpha(n):
    al = np.linspace(0, 0.45, 19); vals = [leading(a, n)[0] for a in al]; a0 = al[int(np.argmax(vals))]
    fine = np.linspace(max(0, a0 - 0.025), a0 + 0.025, 11); vals = [leading(a, n)[0] for a in fine]; a1 = fine[int(np.argmax(vals))]
    B, p = leading(a1, n); return a1, B, p


def realise(D, alpha, p, d, z):
    """sources at x_i = alpha D + d (i - n/2); weights from the moment equations sum_i c_i r_i^k = k! p_k (r_i = x_i/D), k = 0 .. n."""
    n = len(p) - 1; xi = alpha * D + d * (np.arange(n + 1) - n / 2); r = xi / D
    Vm = np.array([[ri ** k for ri in r] for k in range(n + 1)]); mom = np.array([factorial(k) * p[k] for k in range(n + 1)])
    # e^{r xi} = e^{alpha xi} e^{(r - alpha) xi}: match the polynomial part with the shifted rates
    Vs = np.array([[(ri - alpha) ** k for ri in r] for k in range(n + 1)]); c = np.linalg.solve(Vs, mom)
    amp = np.exp(-z) / np.max(np.abs(c))
    lw = np.log(2 * amp * np.abs(c)) + (xi ** 2 - D ** 2) / 2
    xs = np.concatenate([[-D, D], xi]); lws = np.concatenate([[0.0, 0.0], lw]); f = np.concatenate([[0.0, 0.0], np.sign(c)])
    return xs, lws, f


if __name__ == '__main__':
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = {'leading': {}, 'exact': {}}
    for n in range(1, nmax + 1):
        a, B, p = best_alpha(n); out['leading'][str(n)] = {'alpha': float(a), 'B_n': float(B), 'p': p.tolist()}
        print('n %d: B_n = 2 sup rho = %.4f (alpha %.3f)' % (n, B, a), flush=True)
    json.dump(out, open('results/an_multipole.json', 'w'), indent=1)
    for n in (1, 2, 3, 4):
        L = out['leading'][str(n)]; p = np.array(L['p']); res = {}
        for D in (20.0, 40.0, 80.0):
            best = None
            for d in (0.25, 0.5, 1.0):
                for z in (4.0, 8.0, 12.0):
                    xs, lw, f = realise(D, L['alpha'], p, d, z); o = quantities(xs, lw - lw.max(), f)
                    rr = {'d': d, 'z': z, 'Rq': o['Rq'], 'lnMI': o['lnMI'], 'Rq_over_D2': o['Rq'] / D ** 2, 'ratio': o['Rq'] / o['lnMI'], 'margin': o['Rq'] - o['lnMI']}
                    if best is None or rr['ratio'] > best['ratio']: best = rr
            res['%g' % D] = best
            print('  n %d D %g: Rq/D^2 %.4f (-> %.4f) ratio Rq/ln %.4f margin %.1f (d %g z %g)' % (n, D, best['Rq_over_D2'], L['B_n'] / 2, best['ratio'], best['margin'], best['d'], best['z']), flush=True)
        out['exact'][str(n)] = res
        json.dump(out, open('results/an_multipole.json', 'w'), indent=1)
