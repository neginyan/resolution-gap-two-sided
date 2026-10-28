"""Stage AK: the sign-cancellation factor kappa_tau = E_p[(E|sigma_w(X1)| |y)^2] / E_p[s_w^2] on periodic point lattices.
A rectangular lattice is a product state, so its w-structure is the 1-d lattice {j a}: computed here in 1-d (exact sums over j, |j| <= N;
level-t1 posterior per lattice point a Gaussian; 80-node Gauss-Hermite).  t = 1.  Also tR, ln(1/tJ), Delta_tau and kappa_tau/(1 - tau).
(The full 2-d run of ak_run.py on a 121-point patch exceeded the memory budget and was dropped.)  -> results/ak_lattice1d.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.special import logsumexp


def lattice(a, tau, N=40, ny=200, ngh=80, t=1.0):
    t1 = tau * t; t2 = t - t1; P = a * np.arange(-N, N + 1); y = (np.arange(ny) + 0.5) * a / ny
    la = -(y[:, None] - P[None]) ** 2 / (2 * t); lp = logsumexp(la, 1); pi = np.exp(la - lp[:, None])
    m = pi @ P; s = (m - y) / t; v = pi @ P ** 2 - m ** 2; ds = v / t ** 2 - 1 / t          # s' = Var(X|y)/t^2 - 1/t
    gx, gw = np.polynomial.hermite_e.hermegauss(ngh); gw = gw / np.sqrt(2 * np.pi)
    X1 = (P[None, :, None] + (t1 / t) * (y[:, None, None] - P[None, :, None])) + np.sqrt(t1 * t2 / t) * gx[None, None, :]   # (ny, comps, nodes)
    Wn = pi[:, :, None] * gw[None, None, :]
    Xf = X1.reshape(ny, -1); Wf = Wn.reshape(ny, -1)
    sig = np.empty_like(Xf)
    for i in range(ny):                                                                          # level-t1 score at the nodes
        l1 = -(Xf[i][:, None] - P[None]) ** 2 / (2 * t1); q = np.exp(l1 - logsumexp(l1, 1)[:, None]); sig[i] = (q @ P - Xf[i]) / t1
    A = np.sum(Wf * np.abs(sig), 1); s_q = np.sum(Wf * sig, 1)
    lw = lp + np.log(a / ny)                                     # weights over one cell; normalisation by the cell mass
    M = logsumexp(lw); E = lambda f: float(np.exp(logsumexp(lw, b=f) - M))
    J = E(s ** 2); Jabs = E(A ** 2)
    # J(t1): same lattice at blur t1
    la1 = -(y[:, None] - P[None]) ** 2 / (2 * t1); lp1 = logsumexp(la1, 1); pi1 = np.exp(la1 - lp1[:, None]); s1 = (pi1 @ P - y) / t1
    lw1 = lp1 + np.log(a / ny); J1 = float(np.exp(logsumexp(lw1, b=s1 ** 2) - logsumexp(lw1)))
    return {'a': a, 'tau': tau, 'tR': t * E(ds ** 2) / J, 'ln1_tJ': float(-np.log(t * J)), 'kappa': Jabs / J, 'Delta': float(np.log(J1 / J)),
            'tweedie_max_abs': float(np.max(np.abs(s_q - s)) / np.max(np.abs(s))), 'kappa_over_1mtau': Jabs / J / (1 - tau)}


if __name__ == '__main__':
    out = {}
    for a in (2.0, 1.5, 1.2, 1.0, 0.8, 0.7):
        rows = [lattice(a, tau) for tau in (0.5, 0.75, 0.9, 0.95, 0.98)]
        best = min(rows, key=lambda r: r['kappa_over_1mtau'])
        out['%g' % a] = {'rows': rows, 'best_tau': best['tau'], 'min_kappa_over_1mtau': best['kappa_over_1mtau']}
        print('a %.2f tR %.3f ln %.2f | ' % (a, rows[0]['tR'], rows[0]['ln1_tJ']) + ' '.join('tau %.2f: kappa %.3g Delta %.2f' % (r['tau'], r['kappa'], r['Delta']) for r in rows) + ' | min kappa/(1-tau) %.3g at %.2f (tweedie %.1e)' % (best['kappa_over_1mtau'], best['tau'], max(r['tweedie_max_abs'] for r in rows)), flush=True)
    json.dump(out, open('results/ak_lattice1d.json', 'w'), indent=1)
