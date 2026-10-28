"""Stage AW: Lemma AW6 — an exact identity for N_phi on the strips (the background v-information at the bent pendulum), checked on exact
point fields.  With p = N_V(v) P(v, w), P periodic in v (Lemma AW1), s_v = -v/V + s~, m_z = (v' + R)/sqrt2, v' = v(1 - sigma^2/V),
R = sigma^2 s~ + m_w (periodic), a = (1 - sigma^2/V)/sqrt2, e = e^{-a^2 V/2}, C = <cos(R/sqrt2)>, G = d_v s~ + d_v s_w (periodic):
  E[s_v^2 phi] = E[s~^2 phi] + (1 - e C)/V + a^2 e C + sqrt2 sigma^2 a e <cos(R/sqrt2) G>      (+ e^{-k'^2 V/2}-small)
(integration by parts in v against the Gaussian envelope; Gaussian averages of slow functions times periodic ones factorise).
Consequences (proof): E[s~^2 phi] >= (1 - e) J_vv,per... (the window v-information seen at the v-offsets) and the background term (1 - eC)/V ~ Lw^2/(4 Lv^2).
-> results/aw_nphi.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from v3_tools import _pre
from u_cgm import trap_weights
from at_tools import point_fields2
from aq_hybrid import strip_params

KEYS = ('svv', 'phi_svv', 'st2_phi', 'cosR', 'omC', 'cosR_G', 'st2', 'one')


def run(sig, k0, lam, M, c, eta, nper=50, coarse=400, n0=31, nv=64, ext=6.5):
    Lv_sig = eta * lam / np.sqrt(sig); bg = [(1.0, 0.0, lam * M)]; win = [(c, 0.0, lam, 1)]
    T = terms_of(bg, win, k0, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse); Lv = Lv_sig * sig; V = Lv ** 2 + sig ** 2; s2 = sig * sig
    pre = _pre(T, sig); per = 2 * np.pi * sig / k0; ww = trap_weights(wn); v0s = np.linspace(-ext * np.sqrt(V), ext * np.sqrt(V), n0); w0 = trap_weights(v0s)
    acc = {k: 0.0 for k in KEYS}; m = 0.0; npts = max(256, int(2.5e6 // max(1, len(pre)))); a = (1 - s2 / V) / np.sqrt(2)
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2
        V_, W_ = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V_.ravel(), W_.ravel()], 1); dA = cw * np.outer(np.full(nv, 1.0 / nv), ww).ravel()
        for i0 in range(0, len(X), npts):
            x = X[i0:i0 + npts]; p, q = point_fields2(pre, sig, x); pw = p * dA[i0:i0 + npts]; m += pw.sum()
            v = x[:, 0]
            mz = q['mz']; phi = 2 * np.sin(mz / 2) ** 2                                      # cancellation-free 1 - cos
            sv = q['sv']; stl = sv + v / V; R = np.sqrt(2) * mz - a * np.sqrt(2) * v
            G = -q['aHvv'] / s2 + 1 / V - q['Hvw']
            vals = {'svv': sv * sv, 'phi_svv': sv * sv * phi, 'st2_phi': stl * stl * phi, 'cosR': np.cos(R / np.sqrt(2)), 'omC': 2 * np.sin(R / (2 * np.sqrt(2))) ** 2, 'cosR_G': np.cos(R / np.sqrt(2)) * G, 'st2': stl * stl, 'one': np.ones_like(v)}
            for k in KEYS: acc[k] += float(np.sum(pw * vals[k]))
    E = {k: v / m for k, v in acc.items()}; e = np.exp(-a * a * V / 2); C = E['cosR']; ome = -np.expm1(-a * a * V / 2)
    bgt = (ome + e * E['omC']) / V                                                     # (1 - e C)/V without cancellation
    rhs = E['st2_phi'] + bgt + a * a * e * C + np.sqrt(2) * s2 * a * e * E['cosR_G']
    return {'lhs_E_sv2_phi': E['phi_svv'], 'rhs': rhs, 'rel': abs(rhs / E['phi_svv'] - 1), 'background_term': bgt, 'window_term': E['st2_phi'],
            'G_term': np.sqrt(2) * s2 * a * e * E['cosR_G'], 'one_minus_C': E['omC'], 'Lw2_over_4': (lam * M) ** 2 * sig / 4, 'one_minus_e': ome, 'J_vv_per': E['st2'],
            'Jvv_split_rel': abs((E['st2'] + 1 / V) / E['svv'] - 1), 'window_lower_bound_ok': bool(E['st2_phi'] >= ome * E['st2'] * (1 - 1e-9))}


if __name__ == '__main__':
    out = {}
    cases = {}
    for s in (1e-12, 1e-24):
        P = strip_params(s); cases['opt_%g' % s] = (s, P['k0'], P['lam'], P['M'], P['c'], P['Lv_sig'] * np.sqrt(s) / P['lam'])
    cases['wideLv_1e-12'] = (1e-12, 2.0, 1.3, 300.0, 1.0, 2.0)
    for name, args in cases.items():
        r = run(*args); out[name] = r
        print('%-14s E[s_v^2 phi] %.6e  identity rhs %.6e  rel %.1e | background %.3e (1-C %.4e vs Lw^2/4 %.4e)  window %.3e  G-term %.1e | J_vv = J_vv,per + 1/V: rel %.1e | E[s~^2 phi] >= (1-e) J_vv,per: %s' % (
            name, r['lhs_E_sv2_phi'], r['rhs'], r['rel'], r['background_term'], r['one_minus_C'], r['Lw2_over_4'], r['window_term'], r['G_term'], r['Jvv_split_rel'], r['window_lower_bound_ok']), flush=True)
    json.dump(out, open('results/aw_nphi.json', 'w'), indent=1, default=float)
