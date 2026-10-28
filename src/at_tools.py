"""Stage AT tools: the window inequality (Theorem AT2) evaluated on exact point fields.

Theorem AT2 (proof; see paperF/sections_F_material.md §7.14).  For every state (decay at infinity),
  g >= (a/den)(1 - eps) |C_w| sqrt((1 - Phi/4)(1 - delta_w)) / (2 k_eff),
  C_w   = E[s_v^2 cos(m_z/2) (1 - sigma^2 (H_ww + H_vw))] / J_vv                (the v-information seen where the pendulum is still straight)
  k_eff^2 = sigma^2 E[(s_v s_w/2 - H_vw)^2] / J_ww = sigma^2 int (d_v d_w sqrt p)^2 / int (d_w sqrt p)^2   (v-fineness of the w-gradient)
Proof chain: f = s_v sqrt p = 2 d_v sqrt p; Heisenberg in w with psi = 2 sin(m_z/2) (psi^2 = 2 phi exactly, d_w psi = cos(m_z/2) (1 - s2 (H_ww + H_vw))/sqrt 2):
  (C_w J_vv)^2 <= 16 E[s_v^2 phi] int (d_w f)^2;  d_w f = 2 d_v d_w sqrt p (mixed derivatives commute);  int (d_w f)^2 = k_eff^2 J_ww / sigma^2;
  then paper E's P_w = 4 sigma^2 E[s_w^2 (1 - phi/4)] E[s_v^2 phi] / a^2 >= C_w^2 (1 - Phi/4)(1 - delta_w) / (4 k_eff^2).
point_fields2 = paper E's v3_tools.point_fields with H_vw exported (source copied verbatim, one extra key; the release is not modified).
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, inspect
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
import v3_tools
from v3_tools import _pre
from u_cgm import trap_weights

_src = inspect.getsource(v3_tools.point_fields).replace('def point_fields(', 'def point_fields2(')
_old = "extra.update({'B_alt': Bx, 'B_abs': np.abs(Bx), 'sww_w': sw * sw * (1 - phi / 4)})"
assert _src.count(_old) == 1
_src = _src.replace(_old, _old + "\n    extra.update({'Hvw': Hvw, 'Cw_int': sv * sv * np.cos(mz / 2) * (1 - s2 * (Hww + Hvw)), 'X_int': (sv * sw / 2 - Hvw) ** 2, 'mz': mz, 'sv': sv})")
_ns = dict(v3_tools.__dict__); exec(compile(_src, 'point_fields2', 'exec'), _ns); point_fields2 = _ns['point_fields2']

KT = ('svv', 'sww', 'Nibp', 'N1', 'phi_svv', 'sww_w', 'ibp_rest', 'phi', 'Cw_int', 'X_int')


def _acc(pre, sig, X, dA, acc):
    p, q = point_fields2(pre, sig, X); q = dict(q); q['phi'] = 2 * np.sin(np.sqrt(q['mz2']) / 2) ** 2; pw = p * dA
    for k in KT: acc[k] += float(np.sum(pw * q[k]))
    return float(pw.sum())


def sums_grid(T, sig, vn, wn, max_entries=2.5e6):
    """plain trapezoid grid (mixtures, two clusters, class C)."""
    pre = _pre(T, sig); wv, ww = trap_weights(vn), trap_weights(wn); acc = {k: 0.0 for k in KT}; m = 0.0
    rows = max(1, int(max_entries // max(1, len(pre) * len(wn))))
    for a in range(0, len(vn), rows):
        V, W = np.meshgrid(vn[a:a + rows], wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(wv[a:a + rows], ww).ravel()
        m += _acc(pre, sig, X, dA, acc)
    return {k: v / m for k, v in acc.items()}


def sums_envelope(T, sig, kb, Lv, wn, nv=64, n0=31, ext=6.5, max_entries=2.5e6):
    """true V-envelope with one-period averages (as paper E's w_full; strips)."""
    pre = _pre(T, sig); per = 2 * np.pi * sig / kb; ww = trap_weights(wn); v0s = np.linspace(-ext * Lv, ext * Lv, n0); w0 = trap_weights(v0s)
    acc = {k: 0.0 for k in KT}; m = 0.0; npts = max(256, int(max_entries // max(1, len(pre))))
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2
        V, W = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = cw * np.outer(np.full(nv, 1.0 / nv), ww).ravel()
        for i0 in range(0, len(X), npts): m += _acc(pre, sig, X[i0:i0 + npts], dA[i0:i0 + npts], acc)
    return {k: v / m for k, v in acc.items()}


def analyse(E, sig):
    s2 = sig * sig; Jvv, Jww = E['svv'], E['sww']; a = s2 * Jvv; den = s2 * (Jvv + Jww)
    g = (2 * s2 * Jww + E['Nibp']) / (sig * den)
    Pw = 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2
    Bt = s2 / 2 * E['ibp_rest']; R3 = E['Nibp'] - E['N1']; eps = -(Bt + R3) / (sig * a * np.sqrt(Pw))
    Phi = E['phi']; dw = 1 - E['sww_w'] / (Jww * (1 - Phi / 4)); dc = 1 - E['phi_svv'] / (Jvv * Phi)
    Cw = E['Cw_int'] / Jvv; keff = np.sqrt(s2 * E['X_int'] / Jww)
    fac = (a / den) * (1 - eps)
    b_thm2 = fac * np.sqrt(Pw); b_at2 = fac * abs(Cw) * np.sqrt((1 - Phi / 4) * (1 - dw)) / (2 * keff)
    # the inequality inside AT2 (Heisenberg in w), as a ratio >= 1
    heis = 16 * E['phi_svv'] * E['X_int'] / (Cw * Jvv) ** 2          # int (d_w f)^2 = E[(s_v s_w/2 - H_vw)^2] (f = s_v sqrt p); >= 1 by Heisenberg
    return {'g': g, 'sig2Jvv': a, 'sig2Jww': s2 * Jww, 'a_over_den': a / den, 'eps': eps, 'P_w': Pw, 'Phi': Phi, 'delta_w': dw, 'delta_c': dc,
            'C_w': Cw, 'k_eff': keff, 'bound_ThmW': b_thm2, 'bound_AT2': b_at2, 'AT2_over_ThmW': b_at2 / b_thm2, 'AT2_over_g': b_at2 / g,
            'heisenberg_ratio': heis, 'one_over_2keff': 1 / (2 * keff)}
