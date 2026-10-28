"""Stage BC §1: the two-window family at sigma = 1e-12, started from (i) stage BA's optimum and (ii) the strip optimum embedded in the family
(window 1 = the strip's window, k_1 = 0.8 k0 = the strip's k0; window 2 at 1.2 k0 on top of it (offset 0.01 window widths: a window off the straight point by 3 widths already costs g = 3.01 by Lemma BB1) with a tiny amplitude).
The family contains the strip as the limit c_2 -> 0, so its infimum is <= the strip g_min; the question is whether it goes BELOW.
 (1) c_2 scan from the embedded strip (full settings): does a second window lower g at first order?
 (2) Nelder-Mead (300 evaluations, fast settings, re-evaluated at full settings) from both starts.
-> results/bc_two.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, time, sys
import numpy as np
from scipy.optimize import minimize
from ba_tools import gap, strip_opt
from ba_window import two_state, FAST

SIG = 1e-12; BIG = 1e300


def obj(x, full=False):
    try:
        kb, bg, win, Lv_sig, info = two_state(x, SIG); Lb = bg[0][2]; s = np.sqrt(SIG)
        if not (0.05 < kb < 12 and Lb * s < 2.0 and Lv_sig > 30 / kb and all(0.1 < w[2] < 1e3 for w in win)): return BIG
        o = gap(SIG, bg, win, kb, Lv_sig, **({} if full else FAST)); g = float(o['gap_over_sigma'])
        return g if np.isfinite(g) and o['y_over_sigma'] <= 0.5 else BIG
    except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
        return BIG


if __name__ == '__main__':
    np.seterr(all='ignore'); out = {}
    P = strip_opt(SIG); lg = lambda c: np.log(c / (1 - c))
    eta = P['Lv_sig'] * np.sqrt(SIG) / P['lam']
    emb = lambda c2: np.array([np.log(P['k0'] / 0.8), np.log(P['lam']), np.log(P['lam']), np.log(0.01), np.log(P["M"]), np.log(eta), lg(P['c']), lg(c2)])
    out['strip_g'] = P['g']; scan = []
    for c2 in (1e-12, 1e-4, 1e-3, 1e-2, 0.05, 0.2):
        g = obj(emb(c2), full=True); scan.append((c2, g)); print('embedded strip + window 2 (c2 = %g): g = %.8f  (strip %.8f)' % (c2, g, P['g']), flush=True)
    out['c2_scan'] = scan; json.dump(out, open('results/bc_two.json', 'w'), indent=1)
    BA = json.load(open('results/ba_window_two_1e-12.json')); xba = np.array(min(BA['runs'], key=lambda q: q['g_min'])['x'])
    out['runs'] = []
    for name, x0 in (('from stage-BA optimum', xba), ('from the embedded strip (c2 = 1e-3)', emb(1e-3))):
        t0 = time.time(); r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': 300, 'xatol': 1e-4, 'fatol': 1e-8})
        gf = obj(r.x, full=True); kb, bg, win, Lv_sig, info = two_state(r.x, SIG)
        rec = {'start': name, 'g_start_full': obj(x0, full=True), 'g_min_fast': float(r.fun), 'g_min_full': gf, 'x': list(map(float, r.x)), 'nfev': r.nfev,
               'windows': [list(map(float, w)) for w in win], 'kb': kb, 'seconds': time.time() - t0}
        out['runs'].append(rec); json.dump(out, open('results/bc_two.json', 'w'), indent=1)
        print('%s: g %.6f -> %.6f (full %.6f), strip %.6f, windows %s  %.0fs' % (name, rec['g_start_full'], r.fun, gf, P['g'],
              [(round(w[0], 4), round(w[3] * kb, 3)) for w in win], rec['seconds']), flush=True)
