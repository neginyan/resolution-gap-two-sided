"""Stage AS: slice-wise Cramer-Rao (Theorem AS1) and the kappa factors on the strips of paper E.

Theorem AS1 (proof).  For every state, write p(w|v) for the slice of the blurred density p = rho * phi_t at fixed v and V(v) = Var_p(w|v).
  J_ww = E_v[ I(p(.|v)) ]  (d_w ln p = d_w ln p(w|v))   >=   E_v[ 1/V(v) ]   (Cramer-Rao on every slice)
       >= 1/E_v[V(v)]  (Jensen)   >=   1/Var_p(w) = 1/(Var_w(rho) + sigma^2)   (law of total variance).
  Hence with kappa_loc^2 := 4 Phi (1 - Phi/4)(1 - delta_c)(1 - delta_w) E_v[1/V],
     g >= (a/den)(1 - eps) kappa_loc / sqrt(sigma^2 J_vv),   kappa <= kappa_loc <= kappa_true,
  where kappa_true^2 := 4 Phi (1 - Phi/4)(1 - delta_c)(1 - delta_w) J_ww = sigma^2 J_vv P_w   (the exact factor of paper E's Theorem 2).
Here (as in stage AR) Phi = E[1 - cos m_z], delta_c = 1 - E[s_v^2 phi]/(J_vv Phi), delta_w = 1 - E[s_w^2 (1 - phi/4)]/(J_ww (1 - Phi/4)).

Numerics: the strip optima (sigma = 1e-6, 1e-12, 1e-24) with paper E's evaluator (point-chunked w_full, exact point fields), accumulating
also the slice moments of p on every v node.  -> results/as_slices.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from v3_tools import _pre, point_fields
from u_cgm import trap_weights
from aq_hybrid import strip_params

KK = ('svv', 'sww', 'Nibp', 'N1', 'phi_svv', 'sww_w', 'ibp_rest', 'phi')


BFAC = (0.5, 0.7, 1.0, 1.4, 2.0)


def basis(X, sig, k0, bs):
    """h_0 = w;  h = {cos, sin}(k0 v/sigma) w e^{-w^2/2b^2} (window-scale, ripple-locked).  Returns h (nb x n) and d_w h."""
    v, w = X[:, 0], X[:, 1]; c, s_ = np.cos(k0 * v / sig), np.sin(k0 * v / sig); H = [w]; D = [np.ones_like(w)]
    for b in bs:
        e = np.exp(-w * w / (2 * b * b)); f = w * e; df = (1 - w * w / (b * b)) * e
        H += [c * f, s_ * f]; D += [c * df, s_ * df]
    return np.array(H), np.array(D)


def strip_slices(sig, nper=50, coarse=400, n0=31, nv=64, ext=6.5, max_entries=2.5e6):
    P = strip_params(sig); k0 = P['k0']
    bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'], 0.0, P['lam'], 1)]
    T = terms_of(bg, win, k0, sig, P['Lv_sig']); wn = w_nodes(bg, win, sig, nper, coarse)
    Lv = np.sqrt((P['Lv_sig'] * sig) ** 2 + sig ** 2)
    pre = _pre(T, sig); per = 2 * np.pi * sig / k0; ww = trap_weights(wn)
    v0s = np.linspace(-ext * Lv, ext * Lv, n0); w0 = trap_weights(v0s)
    tot = {k: 0.0 for k in KK}; mass = 0.0; invV = 0.0; EV = 0.0; Vs = []
    ell = P['lam'] * np.sqrt(sig); bs = ell * np.array(BFAC); nb = 1 + 2 * len(bs)
    Ge = np.zeros(nb); GG = np.zeros((nb, nb))                               # two-scale test functions (Cramer-Rao with a basis)
    npts = max(256, int(max_entries // max(1, len(pre))))
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2
        V, W = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(np.full(nv, 1.0 / nv), ww).ravel()
        pr = np.empty(len(X)); acc = {k: 0.0 for k in KK}
        for i0 in range(0, len(X), npts):
            p, q = point_fields(pre, sig, X[i0:i0 + npts]); q = dict(q); q['phi'] = 2 * np.sin(np.sqrt(q['mz2']) / 2) ** 2
            pw = p * dA[i0:i0 + npts]; pr[i0:i0 + npts] = p
            for k in acc: acc[k] += float(np.sum(pw * q[k]))
            h, dh = basis(X[i0:i0 + npts], sig, k0, bs)
            Ge += cw * (dh @ pw); GG += cw * ((h * pw) @ h.T)
        pr = pr.reshape(nv, len(wn))
        m0 = pr @ ww; m1 = pr @ (ww * wn); m2 = pr @ (ww * wn * wn)            # slice moments (per unit v)
        Vv = m2 / m0 - (m1 / m0) ** 2
        rowm = m0 / nv
        mass += cw * rowm.sum(); invV += cw * np.sum(rowm / Vv); EV += cw * np.sum(rowm * Vv); Vs.append([float(Vv.min()), float(Vv.max())])
        for k in tot: tot[k] += cw * acc[k]
    Ge /= mass; GG /= mass; J2s = float(Ge @ np.linalg.solve(GG, Ge)); Jw_only = Ge[0] ** 2 / GG[0, 0]
    E = {k: v / mass for k, v in tot.items()}; E['E_invV'] = invV / mass; E['E_V'] = EV / mass; E['V_range'] = [min(a for a, b in Vs), max(b for a, b in Vs)]
    s2 = sig * sig; Jvv, Jww = E['svv'], E['sww']; a = s2 * Jvv; den = s2 * (Jvv + Jww)
    g = (2 * s2 * Jww + E['Nibp']) / (sig * den)
    Pw = 4 * s2 * E['sww_w'] * E['phi_svv'] / a ** 2
    Bt = s2 / 2 * E['ibp_rest']; R3 = E['Nibp'] - E['N1']; eps = -(Bt + R3) / (sig * a * np.sqrt(Pw))
    Phi = E['phi']; dc = 1 - E['phi_svv'] / (Jvv * Phi); dw = 1 - E['sww_w'] / (Jww * (1 - Phi / 4)); F = 4 * Phi * (1 - Phi / 4) * (1 - dc) * (1 - dw)
    Lw2 = (P['lam'] * P['M']) ** 2 * sig; Varw = Lw2                         # Var_w(rho) of the background; window part checked below
    out = {'sigma': sig, 'g': g, 'sig2Jvv': a, 'a_over_den': a / den, 'eps': eps, 'P_w': Pw, 'Phi': Phi, 'delta_c': dc, 'delta_w': dw,
           'Jww': Jww, 'E_invV': E['E_invV'], 'E_V': E['E_V'], 'Var_p_w': Varw + s2, 'V_range': E['V_range'],
           'kappa_true': float(np.sqrt(F * Jww)), 'kappa_loc': float(np.sqrt(F * E['E_invV'])), 'kappa': float(np.sqrt(F / (Varw + s2))),
           'kappa_true_check': float(np.sqrt(Pw * a)), 'Phi_over_Lw2_4': Phi / (Lw2 / 4), 'one_minus_dc': 1 - dc,
           'CR_loc': Jww / E['E_invV'], 'CR_glob': Jww * (Varw + s2), 'ell2_over_Lw2': 1 / P['M'] ** 2, 'M': P['M'], 'k0': k0, 'c': P['c'],
           'bound_ThmW': (a / den) * np.sqrt(Pw) * (1 - eps)}
    out.update({'J_2s': J2s, 'J_w_only': Jw_only, 'CR_2s': Jww / J2s, 'kappa_2s': float(np.sqrt(F * J2s)), 'bfac': list(BFAC)})
    out['bound_2s'] = (a / den) * (1 - eps) * out['kappa_2s'] / np.sqrt(a)
    out['bound_AS1'] = (a / den) * (1 - eps) * out['kappa_loc'] / np.sqrt(a); out['bound_AR2'] = (a / den) * (1 - eps) * out['kappa'] / np.sqrt(a)
    return out


if __name__ == '__main__':
    fn = 'results/as_slices.json'; R = json.load(open(fn)) if os.path.exists(fn) else {}
    for s in [float(x) for x in sys.argv[1:]] or (1e-6, 1e-12, 1e-24):
        key = '%g' % s
        if key in R: continue
        t0 = time.time(); r = strip_slices(s); r['seconds'] = time.time() - t0; R[key] = r
        print(key, ' '.join('%s %.4g' % (k, r[k]) for k in ('g', 'bound_ThmW', 'bound_AS1', 'bound_AR2', 'kappa_true', 'kappa_loc', 'kappa',
                                                              'CR_loc', 'CR_glob', 'CR_2s', 'bound_2s', 'Phi_over_Lw2_4', 'one_minus_dc', 'delta_w', 'ell2_over_Lw2')), '%.0f s' % r['seconds'], flush=True)
        json.dump(R, open(fn, 'w'), indent=1, default=float)
