"""Stage AJ: independent check of the two-level identities on a 3-point state (no line, no log-domain code shared with aj_dv):
  s_w(y) = E[sigma_w(X1) | y]   and   t2 d_j s_w(y) = Cov(X1_j, sigma_w(X1) | y),   sigma = grad ln rho_1,  rho_1 = rho * N(0, t1 I),
with s_w and its gradient from the closed-form mixture at blur t (central differences), the posterior expectations by 2-d Gauss-Hermite
(80 x 80 nodes per component).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np

P = np.array([[0.0, 0.0], [2.5, 1.0], [1.0, -2.0]]); Wt = np.array([0.5, 0.3, 0.2])


def score(y, s):                       # score of sum_c W_c N(P_c, s I) at points y (n, 2)
    d = y[:, None, :] - P[None]; la = np.log(Wt)[None] - (d ** 2).sum(-1) / (2 * s); pi = np.exp(la - la.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    return -(pi[..., None] * d).sum(1) / s


def check(t=1.0, tau=0.5, n=12, seed=1):
    rng = np.random.default_rng(seed); t1 = tau * t; t2 = t - t1; Y = rng.normal(0, 1.5, (n, 2)) + np.array([1.0, 0.0])
    gx, gw = np.polynomial.hermite_e.hermegauss(80); gw = gw / np.sqrt(2 * np.pi); U1, U2 = np.meshgrid(gx, gx, indexing='ij'); GW = np.outer(gw, gw).ravel()
    U = np.stack([U1.ravel(), U2.ravel()], 1); err_s, err_c = 0.0, 0.0
    for y in Y:
        d = y[None] - P; la = np.log(Wt) - (d ** 2).sum(1) / (2 * t); pi = np.exp(la - la.max()); pi /= pi.sum()
        X, Wn = [], []
        for c in range(3):
            m = P[c] + (t1 / t) * (y - P[c]); X.append(m[None] + np.sqrt(t1 * t2 / t) * U); Wn.append(pi[c] * GW)
        X = np.concatenate(X); Wn = np.concatenate(Wn); sig = score(X, t1)[:, 1]
        s_q = np.sum(Wn * sig); s_c = score(y[None], t)[0, 1]
        mX = (Wn[:, None] * X).sum(0); cov = ((Wn * (sig - s_q))[:, None] * (X - mX)).sum(0)
        h = 1e-5; grad = np.array([(score((y + h * e)[None], t)[0, 1] - score((y - h * e)[None], t)[0, 1]) / (2 * h) for e in np.eye(2)])
        err_s = max(err_s, abs(s_q - s_c) / (abs(s_c) + 1e-12)); err_c = max(err_c, np.max(np.abs(t2 * grad - cov)) / (np.max(np.abs(cov)) + 1e-12))
    return err_s, err_c


if __name__ == '__main__':
    print('Tweedie (s_w = E[sigma_w | y]) max rel %.2e; Hessian = Cov(X1, sigma_w)/t2 max rel %.2e' % check())
