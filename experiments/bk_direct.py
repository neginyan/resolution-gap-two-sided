"""Stage BK §0 (numerical tool, not a proof): the two branches (a), (a') of a periodic window by direct 2-D quadrature, valid for every
tau = k'^2 V (no separation of the ripple period from the v-envelope is assumed).

p(v, w) = N(v; V) Gamma(w) h(k' v, w),  h = 1 + sum_n q0_n e^{-w^2/2 s^2} cos(n t)   (Lemma 2, exact),  b = sigma^2/V,  x = (k' sigma)^2,  tau = x/b.
s = grad ln p;  m = y + sigma^2 s (Tweedie);  m_z = (m_v + m_w)/sqrt2;  phi = 1 - cos m_z.
J_vv = E s_v^2,  J_ww = E s_w^2,  den = sigma^2 (J_vv + J_ww),  (a) = 2 A_w/(sigma den), A_w = sigma^2 E[s_w^2 (1 - phi/4)],  (a') = N_phi/(sigma den),
N_phi = (sigma^2/2) E[s_v^2 phi].  Also the exact W = sigma^2 J_vv - b and the period-average sandwich of Table 2 for comparison.
Quadrature: v on +-11 sqrt(V) with >= 48 points per ripple period; w on a union of uniform grids resolving s and L (L <= 12).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np

E_ = np.e


def trap(x):
    w = np.zeros_like(x); d = np.diff(x); w[:-1] += d / 2; w[1:] += d / 2; return w


def grids(sig, b, x, L2, s2):
    V = sig * sig / b; kp = np.sqrt(x) / sig; sv = np.sqrt(V); L = np.sqrt(L2); s = np.sqrt(s2)
    per = 2 * np.pi / kp; nper = 22 * sv / per
    nv = int(max(601, 32 * nper + 601))
    if nv > 40001: raise ValueError("tau too large for the direct grid (separated regime)")
    v = np.linspace(-11 * sv, 11 * sv, nv)
    wl = [np.linspace(-9 * L, 9 * L, 701), np.linspace(-9 * s, 9 * s, 351)]
    w = np.unique(np.concatenate(wl)); return v, w, V, kp


def branches(sig, b, x, q0, L2, s2):
    v, w, V, kp = grids(sig, b, x, L2, s2); q0 = np.asarray(q0, float); n = np.arange(1, len(q0) + 1)
    WV, WW = trap(v), trap(w)
    env = np.exp(-w ** 2 / (2 * s2))                                    # window envelope
    t = kp * v
    C = np.cos(np.outer(n, t)); S = np.sin(np.outer(n, t))              # (N, nv)
    qe = np.outer(q0, env)                                              # (N, nw)
    h = 1 + np.einsum('kv,kw->vw', C, qe)
    h_t = -np.einsum('kv,kw->vw', n[:, None] * S, qe)
    h_w = np.einsum('kv,kw->vw', C, qe * (-w / s2)[None, :])
    Nv = np.exp(-v ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V); Gm = np.exp(-w ** 2 / (2 * L2)) / np.sqrt(2 * np.pi * L2)
    p = Nv[:, None] * Gm[None, :] * h
    sv_ = -v[:, None] / V + kp * h_t / h; sw_ = -w[None, :] / L2 + h_w / h
    mv = v[:, None] + sig * sig * sv_; mw = w[None, :] + sig * sig * sw_
    phi = 2 * np.sin((mv + mw) / (2 * np.sqrt(2))) ** 2                # 2 sin^2(m_z/2): no cancellation for small m_z
    dA = np.outer(WV, WW); P = p * dA; mass = P.sum()
    Jvv = (P * sv_ ** 2).sum() / mass; Jww = (P * sw_ ** 2).sum() / mass
    den = sig * sig * (Jvv + Jww)
    Aw = sig * sig * (P * sw_ ** 2 * (1 - phi / 4)).sum() / mass; Nphi = sig * sig / 2 * (P * sv_ ** 2 * phi).sum() / mass
    a_w = 2 * Aw / (sig * den); a_n = Nphi / (sig * den)
    W = sig * sig * Jvv - b
    Q = float(np.sum(np.abs(q0))); g_s = np.sqrt(s2) / np.sqrt(s2 + 2 * L2); I2 = float(np.sum(n * n * q0 ** 2)) * g_s
    return {'a_w': float(a_w), 'a_N': float(a_n), 'sum': float(a_w + a_n), 'W_exact': float(W), 'W_up_period': x * I2 / (2 * (1 - Q)),
            'W_lo_period': x * I2 / (2 * (1 + Q)), 'sig2Jww': float(sig * sig * Jww), 'den': float(den), 'mass': float(mass), 'tau': x / b, 'Phi': float((P * phi).sum() / mass)}


if __name__ == '__main__':
    # sanity: separated regime agrees with the period sandwich; narrow regime does not
    for b in (1e-3, 0.01, 0.05, 0.2, 0.5):
        r = branches(1e-24, b, 1.0, [0.25], 1e-3, 1e-4)
        print('b %.0e tau %.0e: W_exact %.5f  period sandwich [%.5f, %.5f]  (a) %.3e (a\') %.3e  mass %.6f' % (
            b, r['tau'], r['W_exact'], r['W_lo_period'], r['W_up_period'], r['a_w'], r['a_N'], r['mass']))


def ax3_check(sig, b, x, q0, L2, s2):
    """exact N_phi against Lemma AX3-P's lower bound (1/2)[(1 - e - sigma^2) W + (1 - C)(b - sigma^2 gamma_0)], with the exact W and
    C = E_p[cos(R/sqrt2)], R = sqrt2 m_z - v (1 - b) (the periodic part), gamma_0 the q0-version of gamma_P (Q = sum |q0|)."""
    v, w, V, kp = grids(sig, b, x, L2, s2); q0 = np.asarray(q0, float); n = np.arange(1, len(q0) + 1)
    WV, WW = trap(v), trap(w); env = np.exp(-w ** 2 / (2 * s2)); t = kp * v
    C_ = np.cos(np.outer(n, t)); S_ = np.sin(np.outer(n, t)); qe = np.outer(q0, env)
    h = 1 + np.einsum('kv,kw->vw', C_, qe); h_t = -np.einsum('kv,kw->vw', n[:, None] * S_, qe); h_w = np.einsum('kv,kw->vw', C_, qe * (-w / s2)[None, :])
    Nv = np.exp(-v ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V); Gm = np.exp(-w ** 2 / (2 * L2)) / np.sqrt(2 * np.pi * L2)
    p = Nv[:, None] * Gm[None, :] * h; P = p * np.outer(WV, WW); Z = P.sum()
    sv_ = -v[:, None] / V + kp * h_t / h; sw_ = -w[None, :] / L2 + h_w / h
    mz = ((v[:, None] + sig * sig * sv_) + (w[None, :] + sig * sig * sw_)) / np.sqrt(2)
    phi = 2 * np.sin(mz / 2) ** 2; Nphi = sig * sig / 2 * (P * sv_ ** 2 * phi).sum() / Z
    R = np.sqrt(2) * mz - v[:, None] * (1 - b); C = (P * np.cos(R / np.sqrt(2))).sum() / Z
    omC = (P * 2 * np.sin(R / (2 * np.sqrt(2))) ** 2).sum() / Z           # 1 - C without cancellation
    W = sig * sig * (P * sv_ ** 2).sum() / Z - b
    a2 = (1 - b) ** 2 / 2; ome = -np.expm1(-a2 * V / 2)
    Q = float(np.sum(np.abs(q0))); N1, N2 = float(np.sum(n * np.abs(q0))), float(np.sum(n * n * np.abs(q0)))
    g0 = (x * (N2 * (1 + Q) + N1 ** 2) + np.sqrt(x) * N1 * (1 + 2 * Q) / np.sqrt(E_)) / (1 - Q) ** 2
    rhs = 0.5 * ((ome - sig * sig) * W + omC * (b - sig * sig * g0))
    return {'tau': x / b, 'Nphi': float(Nphi), 'AX3_rhs': float(rhs), 'ratio': float(Nphi / rhs) if rhs > 0 else None, 'one_minus_C': float(omC), 'W': float(W)}
