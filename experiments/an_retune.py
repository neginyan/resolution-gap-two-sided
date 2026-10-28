"""Stage AN: does the worst point configuration of the (S1) search start a family with growing margin?  Re-optimise the margin
Rq - ln(M/I) (Nelder-Mead) from the rescaled worst configuration (x -> lam x, log-weights -> lam^2 log-weights), lam = 1, 1.5, 2, 3.
usage: python an_retune.py SRC LAM -> results/an_retune_<SRC>_<LAM>.json (one process per file).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
from scipy.optimize import minimize
import an_signed1d
from an_signed1d import quantities, objective

src, lam = sys.argv[1], float(sys.argv[2])
an_signed1d.XMAX, an_signed1d.LWMAX = 40.0 * lam + 10, 400.0 * lam ** 2 + 50
r = json.load(open('results/an_signed1d_search_%s.json' % src)); K = r['K']
xs = np.array(r['xs']) * lam; lw = np.array(r['lw']) * lam ** 2; f = np.clip(np.array(r['f']), -0.999999, 0.999999)
nz = int(np.sum(np.abs(f) < 1e-12)); order = np.argsort(np.abs(f) > 1e-12, kind='stable')          # f = 0 masses first (pinned)
xs, lw, f = xs[order], lw[order], f[order]
z0 = np.concatenate([xs, lw, np.arctanh(f)])
o0 = quantities(xs, lw - lw.max(), np.tanh(z0[2 * K:]) * (np.arange(K) >= nz))
best = minimize(objective, z0, args=(K, 1.0, 'margin', nz), method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-5, 'fatol': 1e-7})
for _ in range(2):
    best = minimize(objective, best.x, args=(K, 1.0, 'margin', nz), method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-5, 'fatol': 1e-7})
z = best.x; xs, lw, fz = z[:K], z[K:2 * K], np.tanh(z[2 * K:]); fz[:nz] = 0
o = quantities(xs, lw - lw.max(), fz)
out = {'src': src, 'lam': lam, 'start_margin': o0['Rq'] - o0['lnMI'], 'margin': o['Rq'] - o['lnMI'], 'Rq': o['Rq'], 'lnMI': o['lnMI'],
       'xs': xs.tolist(), 'lw': (lw - lw.max()).tolist(), 'f': fz.tolist(), 'K': K}
print(json.dumps({k: out[k] for k in ('src', 'lam', 'start_margin', 'margin', 'Rq', 'lnMI')}), flush=True)
json.dump(out, open('results/an_retune_%s_%g.json' % (src, lam), 'w'), indent=1)
