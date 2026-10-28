"""Stage BA §2, candidate 3 (route 3 of the map): hidden multipole rows inside the strip with the window's w-modulation switched off.

State: rho = N(v; 0, Lv^2) N(w; 0, Lw^2) [ 1 + c cos(k0 v/sigma) + beta sum_i a_i C(v - x_i) (1 + f_i cos(kw w/sigma)) ],
  i.e. the ripple fills the whole background (no window edges: the window's own w-information is switched off), and signed rows
  (stage AQ's H-A rows: dipole n = 1 / quadrupole n = 2, spacing d sigma, at the ripple trough v = pi sigma/k0, Gaussian teeth 0.8 sigma)
  carry a w-modulation at kw (8 periods per Lw, resolvable at every sigma).  Hiding: small d (the blurred w-modulations of the rows
  cancel) and c -> 1 (deep valleys).  Controls: the same state without rows (flat ripple), and the window switched on (stage AQ).
The control is minimised over (k0, Lw, Lv, c) (Nelder-Mead, fast settings, re-evaluated at full settings); the rows (n = 1, 2; d = 0.6, 0.2;
beta = 0.01, 0.1) are then added at that minimum and at 3x its Lw; branches (a), (a') and Lambda at the minimum and with beta = 0.1 rows.
usage: python ba_hidden.py SIGMA [MAXFEV] -> results/ba_hidden_<sigma>.json.  Python 3.11."""
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
from w_tools import terms_of, w_nodes
from w_full import full_sums
from at_tools import sums_envelope, analyse
from aq_hybrid import rows_terms, multipole
from ba_tools import strip_opt

FAST = dict(nper=12, coarse=100, n0=11, nv=16); FULL = dict(nper=25, coarse=200, n0=15, nv=32)


def save(fn, val): json.dump(val, open(fn + '.tmp', 'w'), indent=1, default=float); os.replace(fn + '.tmp', fn)


def build(sig, k0, Lw, Lv_sig, c, rows):
    s = np.sqrt(sig); lam_big = 30 * Lw / s                 # window width 30 Lw: flat over the background (Abr = 1 - 1/1800)
    P = {'Lv_sig': Lv_sig, 'lam': lam_big, 'M': Lw / (lam_big * s)}
    bg = [(1.0, 0.0, Lw / s)]; win = [(c, 0.0, lam_big, 1)]
    T = terms_of(bg, win, k0, sig, Lv_sig); wn = w_nodes(bg, win, sig, 12, 200); mrows = 1
    if rows is not None:
        n, d, beta = rows; kw = 2 * np.pi * 8 * sig / Lw; xs, a, f = multipole(n, np.pi / k0, d); a = a / a.sum()
        mrows = int(np.ceil(5.5 / (k0 * 0.8))) + 1
        T = T + rows_terms(sig, P, k0, xs, a, f, beta, kw)
        wn = np.unique(np.concatenate([wn, np.arange(-6.5 * Lw, 6.5 * Lw, 2 * np.pi * sig / kw / 10)]))
    return T, wn, mrows


def g_of(sig, k0, Lw, Lv_sig, c, rows, st=FAST):
    T, wn, mr = build(sig, k0, Lw, Lv_sig, c, rows)
    o = full_sums(T, sig, k0, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=st['nv'] * mr, n0=st['n0'], ext=6.5)
    return float(o['gap_over_sigma']), o


def br_of(sig, k0, Lw, Lv_sig, c, rows):
    T, wn, mr = build(sig, k0, Lw, Lv_sig, c, rows)
    r = analyse(sums_envelope(T, sig, k0, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=32 * mr, n0=21), sig)
    den = r['sig2Jvv'] + r['sig2Jww']
    aw = 2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (sig * den); aN = r['sig2Jvv'] * r['Phi'] * (1 - r['delta_c']) / (2 * sig * den)
    return {'g': float(r['g']), 'a_w': float(aw), 'a_N': float(aN), 'Lambda': float(4 * r['k_eff'] ** 2), 'eps': float(r['eps'])}


def run(sig, maxfev=60):
    """(1) the flat-ripple control (no rows) is minimised over (k0, Lw, Lv, c); (2) the rows are added at that minimum and at 3x Lw
    (each row evaluation is ~100x the cost of the control: the w-modulation must be resolved across the background)."""
    fn = 'results/ba_hidden_%g.json' % sig; R = json.load(open(fn)) if os.path.exists(fn) else {}
    P = strip_opt(sig); s = np.sqrt(sig); R['strip_g'] = P['g']; ln = np.log(1 / sig)
    if 'flat' not in R:
        def obj(x):
            try:
                k0, Lw, Lv_sig = np.exp(x[:3]); c = 1 / (1 + np.exp(-x[3]))
                if not (0.3 < k0 < 12 and 2 * sig < Lw < 2.0 and Lv_sig > 30 / k0): return 1e9
                g, o = g_of(sig, k0, Lw, Lv_sig, c, None)
                return g if np.isfinite(g) else 1e9
            except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
                return 1e9
        best = None; t0 = time.time()
        for Lw0 in (2 * s, 20 * s):                 # the AP optimum of the lines is Lw = 2 sqrt(sigma)
            x0 = np.array([np.log(P['k0']), np.log(Lw0), np.log(P['Lv_sig']), np.log(0.999 / 0.001)])
            r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev * 3, 'xatol': 1e-3, 'fatol': 1e-6})
            if best is None or r.fun < best.fun: best = r
        k0, Lw, Lv_sig = np.exp(best.x[:3]); c = 1 / (1 + np.exp(-best.x[3]))
        g_full, _ = g_of(sig, k0, Lw, Lv_sig, c, None, FULL); b = br_of(sig, k0, Lw, Lv_sig, c, None)
        R['flat'] = {'rows': None, 'g_min_fast': float(best.fun), 'g_min': g_full, 'k0': k0, 'Lw': Lw, 'Lw_over_sqrt_sigma': Lw / s, 'Lv_sig': Lv_sig, 'c': c,
                     'branches': b, 'seconds': time.time() - t0}
        save(fn, R)
        print('hidden sigma %g flat (no rows) g_min %.6g (strip %.6g)  (a) %.4g (a\') %.4g Lambda %.4g (Lambda/ln %.3f)  Lw/sqrt(sig) %.3g c %.4f' % (
            sig, g_full, P['g'], b['a_w'], b['a_N'], b['Lambda'], b['Lambda'] / ln, Lw / s, c), flush=True)
    F = R['flat']
    for n in (1, 2):
        for d in (0.6, 0.2):
            for beta in ((1e-2, 0.1) if sig >= 1e-9 else (0.1,)):          # smaller sigma: the beta = 0.1 rows at the control's Lw only (cost)
                for lf in ((1.0, 3.0) if sig >= 1e-9 else (1.0,)):
                    key = 'n%d_d%g_b%g_Lw%g' % (n, d, beta, lf)
                    if key in R: continue
                    t0 = time.time(); rows = (n, d, beta); Lw = F['Lw'] * lf
                    g, o = g_of(sig, F['k0'], Lw, F['Lv_sig'], F['c'], rows)
                    rec = {'rows': rows, 'Lw_factor': lf, 'g_fast': g, 'seconds': time.time() - t0}
                    if beta == 0.1 and lf == 1.0: rec['branches'] = br_of(sig, F['k0'], Lw, F['Lv_sig'], F['c'], rows)
                    R[key] = rec; save(fn, R)
                    b = rec.get('branches')
                    print('hidden sigma %g %-22s g %.6g (flat %.6g, strip %.6g)%s  %.0fs' % (sig, key, g, F['g_min'], P['g'],
                          '' if b is None else '  (a) %.4g (a\') %.4g Lambda %.4g (Lambda/ln %.3f)' % (b['a_w'], b['a_N'], b['Lambda'], b['Lambda'] / ln), time.time() - t0), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    run(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 60)
