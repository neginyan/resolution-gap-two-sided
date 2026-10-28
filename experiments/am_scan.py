"""Stage AM: kappa_e(theta, tau) of the oblique two-level chain on the staggered double row (two w-lines + rows at v = D -+ 1 offset by a/2)
and the B' needed by Proposition AM1, minimised over the direction and tau.  -> results/am_scan_<D>_<a>.json (restartable; one file per case).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np
from am_oblique import chain
from al_staggered import config

if __name__ == '__main__':
    tag = '_'.join(sys.argv[1:3]) if len(sys.argv) > 2 else 'all'
    fn = 'results/am_scan_%s.json' % tag; out = json.load(open(fn)) if os.path.exists(fn) else {}
    cases = [(8.0, 2.0), (14.0, 1.5), (20.0, 1.2)]
    if len(sys.argv) > 1: cases = [tuple(map(float, sys.argv[1:3]))]
    for D, a in cases:
        lines, pts = config(D, a, 0.1); yv = np.arange(-8.0, 2 * D + 8.0, min(0.04, 0.25 / D)); yw = (np.arange(10) + 0.5) * a / 10
        full = D <= 8
        for tau in ((0.1, 0.25, 0.5, 0.75, 0.9) if full else (0.5, 0.75, 0.9)):
            for deg in ((0, 5, 10, 15, 20, 25, 30, 40, 50) if full else (0, 5, 10, 20)):
                k = '%g_%g_%g_%g' % (D, a, tau, deg)
                if k in out: continue
                o = chain(lines, pts, yv, yw, theta=np.radians(deg), tau=tau, ngh=30); o.update({'D': D, 'a': a, 'deg': deg})
                out[k] = o; json.dump(out, open(fn, 'w'), indent=1, default=float)
                print('%-18s kappa_e %10.4g  tR_e %8.3f  B_needed %10.4g  slope %10.4g  id %.1e' % (k, o['kappa_e'], o['tR_e'], o['Bneeded_AM1'], o['slope_AM1'], o['identity_c2e_rel_L2']), flush=True)
