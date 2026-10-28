"""Stage BD: case 3 (two w-scales) — its structure, the visibility inequality, and counterexample candidates with non-cos windows.

Lemma BD1 (proof, every state): 1 - cos t >= (2/pi^2) d(t)^2, d(t) = distance of t to 2 pi Z.  With pi_v := s_v^2 p/J_vv (the v-information weight),
   E_{pi_v}[d(m_z)^2] <= (pi^2/2) theta = (pi^2/8) Q W2.
So in case 3 (Q < c^2/ln(1/sigma), theta < 2 sigma c/sqrt(ln(1/sigma))) the v-information sits, in the posterior mean m_z, within a mean-square distance
(pi^2/8) Q W2 << W2/ln(1/sigma) of the straight points: a window narrow against the w-spread sqrt(W2).
Visibility inequality (VI; the strip form is Theorem AW5: k_eff^2 <= ln(L/s) + 1.68):  k_eff^2 <= (1/2) ln(1/Q) + C  (Q ~ s^2/(2L^2) on strips).
Numerics here:
 'scan'  : k_eff^2 against (1/2) ln(1/Q) over window shapes (cos = strip; comb of thin lines; two harmonics k, 2k; amplitude varying inside the window
           = narrow + wide window at one k), scale ratios M = L/s and wavenumbers k (sigma = 1e-12);
 'opt'   : the true gap minimised (Nelder-Mead, fast settings) in the comb / harmonic / amplitude families, sigma = 1e-6, 1e-12; case, Lambda/ln at the minimum.
usage: python bd_case3.py scan|opt SIGMA [SHAPE] -> results/bd_<part>_<shape>_<sigma>.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
import at_tools
from w_tools import terms_of, w_nodes
from w_full import full_sums
from at_tools import sums_envelope, analyse
from ba_tools import strip_opt
from bc_cases import W2_of

# s_v^2 d(m_z)^2 accumulated alongside paper E's sums (Lemma BD1 check)
_pf2 = at_tools.point_fields2
def _pf2x(pre, sig, X):
    p, q = _pf2(pre, sig, X); q = dict(q); mz = np.sqrt(q['mz2']); d = np.abs(mz - 2 * np.pi * np.round(mz / (2 * np.pi)))
    q['svd2'] = q['svv'] * d * d; return p, q
at_tools.point_fields2 = _pf2x; at_tools.KT = at_tools.KT + ('svd2',)


def windows(shape, k, lam, c, par):
    """returns (kb, win list).  par: comb -> tooth width st (sigma units); harm -> c2/c1; amp -> (lam2/lam1, c2/c1)."""
    if shape == 'cos': return k, [(c, 0.0, lam, 1)]
    if shape == 'comb':
        st = par; mmax = max(1, int(np.ceil(np.sqrt(2 * np.log(1e6)) / (k * st))))
        return k, [(c * 2 * np.exp(-(m * k * st) ** 2 / 2), 0.0, lam, m) for m in range(1, mmax + 1)]
    if shape == 'harm':
        r = par; c1 = c / (1 + r); return k, [(c1, 0.0, lam, 1), (c1 * r, 0.0, lam, 2)]
    if shape == 'amp':
        rl, rc = par; c1 = c / (1 + rc); return k, [(c1, 0.0, lam, 1), (c1 * rc, 0.0, lam * rl, 1)]
    raise ValueError(shape)


def row(sig, bg, win, kb, Lv_sig, label, nv=32, n0=21, nper=40, coarse=300):
    mmax = max(w[3] for w in win); T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    E = sums_envelope(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv * mmax, n0=n0); r = analyse(E, sig)
    s = np.sqrt(sig); W2 = W2_of(sig, bg[0][2] * s, Lv_sig * sig); A, B = r['sig2Jww'], r['sig2Jvv']
    theta = (1 - r['delta_c']) * r['Phi']; Q = 4 * theta / W2; ln = np.log(1 / sig); keff2 = r['k_eff'] ** 2
    Ed2 = E['svd2'] / E['svv']
    case3 = bool(Q * ln < 0.1 and (1 - abs(r['eps'])) * theta / (2 * sig) < r['bound_AT2'] and A / (A + B) < 2 * sig)
    return {'label': label, 'sigma': sig, 'g': float(r['g']), 'theta': float(theta), 'Q': float(Q), 'W2': W2, 'k_eff2': float(keff2), 'half_ln_1_over_Q': 0.5 * np.log(1 / Q),
            'VI_excess': float(keff2 - 0.5 * np.log(1 / Q)), 'Lambda_over_ln': float(4 * keff2 / ln), 'AT2': float(r['bound_AT2']), 'AT2_sqrt_ln': float(r['bound_AT2'] * np.sqrt(ln)),
            'E_piv_d2': float(Ed2), 'BD1_bound': float(np.pi ** 2 / 2 * theta), 'BD1_ratio': float(Ed2 / (np.pi ** 2 / 2 * theta)), 'A_frac': float(A / (A + B)), 'eps': float(r['eps']),
            'C_w': float(r['C_w']), 'case3': case3}


def scan(sig, shape):
    fn = 'results/bd_scan_%s_%g.json' % (shape, sig); R = json.load(open(fn)) if os.path.exists(fn) else {}
    s = np.sqrt(sig); lam = 1.5; c = 0.999; eta = 0.2
    par = {'cos': None, 'comb': 0.4, 'harm': 1.0, 'amp': (4.0, 1.0)}[shape]
    for M in (1e2, 1e4, 1e5):
        if M * lam * s >= 2: continue
        for k in (1.5, 2.5, 3.5, 4.5):
            key = 'M%g_k%g' % (M, k)
            if key in R: continue
            t0 = time.time(); kb, win = windows(shape, k, lam, c, par)
            q = row(sig, [(1.0, 0.0, lam * M)], win, kb, eta * lam / s, '%s M = %g k = %g' % (shape, M, k)); q.update({'M': M, 'k': k, 'shape': shape, 'seconds': time.time() - t0})
            R[key] = q; json.dump(R, open(fn, 'w'), indent=1, default=float)
            print('%-5s M %-6g k %-4g: k_eff^2 %.3f  (1/2)ln(1/Q) %.3f  ln M %.3f  excess %+.3f | Lambda/ln %.3f  case3 %s  g %.4g  BD1 ratio %.3f  %.0fs' % (
                shape, M, k, q['k_eff2'], q['half_ln_1_over_Q'], np.log(M), q['VI_excess'], q['Lambda_over_ln'], q['case3'], q['g'], q['BD1_ratio'], q['seconds']), flush=True)


def opt(sig, shape, maxfev=100):
    fn = 'results/bd_opt_%s_%g.json' % (shape, sig); P = strip_opt(sig); s = np.sqrt(sig)
    def st(x):
        k, lam, M, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4]))
        par = {'comb': np.exp(x[5]), 'harm': np.exp(x[5]), 'amp': (1 + np.exp(x[5]), np.exp(x[6]) if len(x) > 6 else 1.0)}[shape]
        kb, win = windows(shape, k, lam, c, par); return kb, win, [(1.0, 0.0, lam * M)], eta * lam / s, par
    def obj(x, full=False):
        try:
            kb, win, bg, Lv_sig, par = st(x)
            if not (0.3 < kb < 12 and bg[0][2] * s < 2 and Lv_sig > 30 / kb and len(win) <= 12): return 1e300
            T = terms_of(bg, win, kb, sig, Lv_sig); mmax = max(w[3] for w in win)
            wn = w_nodes(bg, win, sig, 25 if full else 12, 200 if full else 100)
            o = full_sums(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=(32 if full else 16) * mmax, n0=15 if full else 11, ext=6.5)
            g = float(o['gap_over_sigma']); return g if np.isfinite(g) and o['y_over_sigma'] <= 0.5 else 1e300
        except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
            return 1e300
    eta0 = P['Lv_sig'] * sig / (P['lam'] * s); base = [np.log(P['k0']), np.log(P['lam']), np.log(P['M']), np.log(eta0), np.log(P['c'] / (1 - P['c']))]
    x0 = np.array(base + {'comb': [np.log(0.5 / P['k0'])], 'harm': [np.log(0.3)], 'amp': [np.log(3.0), np.log(0.3)]}[shape])
    t0 = time.time(); g0 = obj(x0); r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-6})
    gf = obj(r.x, True); kb, win, bg, Lv_sig, par = st(r.x)
    q = row(sig, bg, win, kb, Lv_sig, 'optimum of the %s family' % shape)
    out = {'shape': shape, 'sigma': sig, 'g_start': g0, 'g_min_fast': float(r.fun), 'g_min_full': gf, 'strip_g': P['g'], 'x': list(map(float, r.x)), 'par': par,
           'windows': [list(map(float, w)) for w in win], 'kb': kb, 'case_row': q, 'seconds': time.time() - t0}
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    print('%s sigma %g: g %.5g -> %.6g (full %.6g; strip %.6g)  par %s | case3 %s Lambda/ln %.3f VI excess %+.3f  %.0fs' % (shape, sig, g0, r.fun, gf, P['g'], par, q['case3'],
          q['Lambda_over_ln'], q['VI_excess'], out['seconds']), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    part, sig = sys.argv[1], float(sys.argv[2])
    if part == 'scan': scan(sig, sys.argv[3])
    else: opt(sig, sys.argv[3])
