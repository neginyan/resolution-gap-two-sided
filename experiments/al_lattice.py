"""Stage AL, plan A on point lattices (direct sums; lattice weights are all O(1), so no log-domain bookkeeping is needed):
Proposition AL1's v-only sign-cancellation factor kappa_v = E_p[(E|sigma_w(X1v, y_w)| | y)^2]/J_ww, the split tR = tR_vw + tR_ww, and the
B' needed, on rectangular (product) and oblique lattices with basis b1 = (a1, 0), b2 = (s, a2).  t = 1.  -> results/al_lattice.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.special import logsumexp
from al_aniso import one_d_constants


def lattice(a1, s, a2, tau_v=0.5, ngh=40, ny=(40, 40), R=8.0, t=1.0):
    b1, b2 = np.array([a1, 0.0]), np.array([s, a2]); P = []
    n = int(np.ceil(2 * R / min(a1, a2))) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i * b1 + j * b2
            if np.hypot(*(x - (b1 + b2) / 2)) < R + max(a1, a2): P.append(x)
    P = np.array(P); tv1 = tau_v * t; t2 = t - tv1
    u = (np.arange(ny[0]) + 0.5) / ny[0]; v = (np.arange(ny[1]) + 0.5) / ny[1]; U, Vv = np.meshgrid(u, v, indexing='ij')
    Y = (U[..., None] * b1 + Vv[..., None] * b2).reshape(-1, 2)                                   # one cell
    la = -((Y[:, None, :] - P[None]) ** 2).sum(-1) / (2 * t); lp = logsumexp(la, 1); pi = np.exp(la - lp[:, None])
    m = pi @ P; s_ = (m - Y) / t
    C = np.einsum('nk,ki,kj->nij', pi, P, P) - np.einsum('ni,nj->nij', m, m)
    Hvw = -C[:, 0, 1] / t ** 2; Hww = (1 - C[:, 1, 1] / t) / t
    gx, gw = np.polynomial.hermite_e.hermegauss(ngh); gw = gw / np.sqrt(2 * np.pi)
    X1v = (P[None, :, 0, None] + (tv1 / t) * (Y[:, 0, None, None] - P[None, :, 0, None])) + np.sqrt(tv1 * t2 / t) * gx[None, None, :]
    Wn = pi[:, :, None] * gw[None, None, :]
    Xf = X1v.reshape(len(Y), -1); Wf = Wn.reshape(len(Y), -1); sig = np.empty_like(Xf)
    for i in range(len(Y)):
        l1 = -(Xf[i][:, None] - P[None, :, 0]) ** 2 / (2 * tv1) - (Y[i, 1] - P[None, :, 1]) ** 2 / (2 * t)
        q = np.exp(l1 - logsumexp(l1, 1)[:, None]); sig[i] = (q @ P[:, 1] - Y[i, 1]) / t
    A = np.sum(Wf * np.abs(sig), 1); sq = np.sum(Wf * sig, 1)
    w = np.exp(lp - lp.max()); w /= w.sum(); E = lambda f: float(np.sum(w * f))
    J = E(s_[:, 1] ** 2)
    o = {'a1': a1, 's': s, 'a2': a2, 'tau_v': tau_v, 'n_points': len(P), 'tweedie_max_rel': float(np.max(np.abs(sq - s_[:, 1])) / np.max(np.abs(s_[:, 1]))),
         'tR_vw': t * E(Hvw ** 2) / J, 'tR_ww': t * E(Hww ** 2) / J, 'ln1_tJ': float(-np.log(t * J)), 'kappa_v': E(A ** 2) / J}
    o['tR'] = o['tR_vw'] + o['tR_ww']
    C1 = one_d_constants(); ln1 = o['ln1_tJ']
    o['Bneeded_vw_bound_coeff'] = 16 * o['kappa_v'] / (1 - tau_v) + 16           # coefficient of ln(1/tJ) in the vw part of Prop. AL1
    o['Bprime_AL1'] = C1['B1'] + o['Bneeded_vw_bound_coeff']
    return o


if __name__ == '__main__':
    out = {}
    for a1, s, a2 in ((1.0, 0.0, 1.0), (1.5, 0.0, 0.8), (1.0, 0.5, 1.0), (1.3, 0.6, 1.1), (1.0, 0.3, 0.8), (2.0, 1.0, 0.9), (1.5, 0.75, 1.2)):
        for tau_v in (0.25, 0.5, 0.75):
            o = lattice(a1, s, a2, tau_v); k = '%g_%g_%g_%g' % (a1, s, a2, tau_v); out[k] = o
            print(k, ' '.join('%s %.4g' % (q, o[q]) for q in ('tR', 'tR_vw', 'tR_ww', 'ln1_tJ', 'kappa_v', 'Bprime_AL1', 'tweedie_max_rel')), flush=True)
    json.dump(out, open('results/al_lattice.json', 'w'), indent=1, default=float)
