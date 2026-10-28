"""Stage AM: kappa_e(theta, tau) of the oblique two-level chain on point lattices (direct sums), basis b1 = (a1, 0), b2 = (s, a2), t = 1.
For a lattice the level-1 w-score is periodic; a direction e along which its sign does not change would make kappa_e ~ 1.
-> results/am_lattice.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.special import logsumexp


def lattice_e(a1, s, a2, theta, tau, ngh=30, ny=(24, 24), R=7.0, t=1.0):
    b1, b2 = np.array([a1, 0.0]), np.array([s, a2]); P = []
    n = int(np.ceil(2 * R / min(a1, a2))) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i * b1 + j * b2
            if np.hypot(*(x - (b1 + b2) / 2)) < R + max(a1, a2): P.append(x)
    P = np.array(P); e = np.array([np.cos(theta), np.sin(theta)]); t2 = (1 - tau) * t
    S1 = t * np.eye(2) - t2 * np.outer(e, e); Si = np.linalg.inv(S1)
    u = (np.arange(ny[0]) + 0.5) / ny[0]; v = (np.arange(ny[1]) + 0.5) / ny[1]; U, Vv = np.meshgrid(u, v, indexing='ij')
    Y = (U[..., None] * b1 + Vv[..., None] * b2).reshape(-1, 2)
    la = -((Y[:, None, :] - P[None]) ** 2).sum(-1) / (2 * t); lp = logsumexp(la, 1); pi = np.exp(la - lp[:, None])
    m = pi @ P; s_ = (m - Y) / t; C = np.einsum('nk,ki,kj->nij', pi, P, P) - np.einsum('ni,nj->nij', m, m)
    Hvw = -C[:, 0, 1] / t ** 2; Hww = (1 - C[:, 1, 1] / t) / t
    gx, gw = np.polynomial.hermite_e.hermegauss(ngh); gw = gw / np.sqrt(2 * np.pi)
    Prec = e @ Si @ e + 1 / t2
    lam = ((Y[:, None, :] - P[None]) @ (Si @ e)) / Prec                                   # posterior mean of lambda per component
    LAM = lam[:, :, None] + gx[None, None, :] / np.sqrt(Prec)
    X = Y[:, None, None, :] - LAM[..., None] * e                                          # (ny, comps, nodes, 2)
    Wn = (pi[:, :, None] * gw[None, None, :]).reshape(len(Y), -1); Xf = X.reshape(len(Y), -1, 2)
    sig = np.empty(Xf.shape[:2])
    for i in range(len(Y)):
        d = Xf[i][:, None, :] - P[None]; q = -0.5 * np.einsum('nki,ij,nkj->nk', d, Si, d)
        q = np.exp(q - logsumexp(q, 1)[:, None]); sig[i] = -np.einsum('nk,nki->ni', q, d @ Si.T)[:, 1]
    A = np.sum(Wn * np.abs(sig), 1); sq = np.sum(Wn * sig, 1)
    w = np.exp(lp - lp.max()); w /= w.sum(); E = lambda f: float(np.sum(w * f)); J = E(s_[:, 1] ** 2)
    return {'a1': a1, 's': s, 'a2': a2, 'deg': float(np.degrees(theta)), 'tau': tau, 'kappa_e': E(A ** 2) / J, 'tR_vw': E(Hvw ** 2) / J, 'tR_ww': E(Hww ** 2) / J,
            'ln1_tJ': float(-np.log(J)), 'tweedie_max_rel': float(np.max(np.abs(sq - s_[:, 1])) / np.max(np.abs(s_[:, 1])))}


if __name__ == '__main__':
    out = {}
    for a1, s, a2 in ((1.3, 0.6, 1.1), (2.0, 1.0, 0.9), (1.5, 0.75, 1.2)):
        for tau in (0.25, 0.5, 0.75):
            for deg in range(-60, 61, 10):
                o = lattice_e(a1, s, a2, np.radians(deg), tau); out['%g_%g_%g_%g_%g' % (a1, s, a2, tau, deg)] = o
            best = min((v for v in out.values() if (v['a1'], v['s'], v['a2'], v['tau']) == (a1, s, a2, tau)), key=lambda v: v['kappa_e'])
            print('lattice (%g, %g, %g) tau %.2f: min over theta kappa_e = %.4g at %g deg (theta = 0: %.4g); tweedie %.1e' % (
                a1, s, a2, tau, best['kappa_e'], best['deg'], out['%g_%g_%g_%g_0' % (a1, s, a2, tau)]['kappa_e'], best['tweedie_max_rel']), flush=True)
    json.dump(out, open('results/am_lattice.json', 'w'), indent=1, default=float)
