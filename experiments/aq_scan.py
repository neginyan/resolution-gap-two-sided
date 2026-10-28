"""Stage AQ, hybrid H-A: the strip optimum at sigma plus signed, w-modulated rows (dipole n = 1, quadrupole n = 2) at the ripple trough,
relative mass beta.  TRUE gap with paper E's evaluator (aq_hybrid.gap_of -> exact point fields, true V-envelope).
The w-modulation has wavenumber kw/sigma (kw = 1, the multipole of stages AN-AP) where the window can be resolved (ell/sigma <= 2e4);
otherwise kw is lowered to 8 periods per window width (the sigma-scale case is covered by the marginal-ratio argument, AQ1).
usage: python aq_scan.py SIGMA -> results/aq_hA_<SIGMA>.json (restartable: finished cases are skipped).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import sys, os, json, time
import numpy as np
from aq_hybrid import strip_params, gap_of, rows_terms, multipole

sig = float(sys.argv[1]); fn = 'results/aq_hA_%g.json' % sig
out = json.load(open(fn)) if os.path.exists(fn) else {}
P = strip_params(sig); ell_over_sig = P['lam'] / np.sqrt(sig)
kw = 1.0 if ell_over_sig <= 2e4 else 2 * np.pi * 8 / ell_over_sig
if 'strip' not in out:
    t0 = time.time(); o = gap_of(sig, P); out['strip'] = {'g': o['gap_over_sigma'], 'g_stored': P['g_stored'], 'params': P, 'kw': kw, 'seconds': time.time() - t0}
    json.dump(out, open(fn, 'w'), indent=1, default=float)
print('sigma %g strip g = %.5f (stored %.5f); kw = %.3g' % (sig, out['strip']['g'], P['g_stored'], kw), flush=True)
betas = (1e-5, 1e-3, 1e-1)                 # (sigma = 1e-6, beta = 0.1 added in stage AR; restartable: done cases skipped)
for n in (1, 2):
    for beta in betas:
        key = 'n%d_beta%g' % (n, beta)
        if key in out: continue
        xs, a, f = multipole(n, np.pi / P['k0'], 0.6); a = a / a.sum()
        t0 = time.time(); o = gap_of(sig, P, rows_terms(sig, P, P['k0'], xs, a, f, beta, kw), kw=kw)
        out[key] = {'n': n, 'beta': beta, 'kw': kw, 'g': o['gap_over_sigma'], 'sig2Jww': sig ** 2 * o['J_ww'], 'sig2Jvv': sig ** 2 * o['J_vv'], 'N': o['N_ibp'],
                    'y_over_sigma': o['y_over_sigma'], 'seconds': time.time() - t0}
        json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('  %s: g = %.5f (strip %.5f, change %+.2e), %.0f s' % (key, out[key]['g'], out['strip']['g'], out[key]['g'] - out['strip']['g'], out[key]['seconds']), flush=True)
