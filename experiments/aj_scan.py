"""Stage AJ: (Delta_1/2, tR, ln(1/tJ)) over the w-line + point family (z from -D/2 to D) and the two-line + midpoint family:
where do states sit relative to the dichotomy of Theorem AJ2 [ Delta <= Delta0  (AJ1)  or  tR <= K Delta + c  (increment route) ]?
-> results/aj_scan.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from aj_increment import delta_lp, delta_tl

if __name__ == '__main__':
    fn = 'results/aj_scan.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    jobs = [('lp', D, zr) for D in (6.0, 8.0, 12.0, 16.0, 24.0) for zr in (-0.25, -0.1, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)]
    jobs += [('tl', D, z) for D in (8.0, 14.0, 20.0) for z in (-0.4, -0.2, 0.0, 0.1, 0.3, 0.6, 1.0, 2.0)]
    for kind, D, zz in jobs:
        z = zz * D if kind == 'lp' else zz
        k = '%s_%g_%g' % (kind, D, z)
        if k in out: continue
        d, r = (delta_lp(D, z, Wmax=8.0, nw=801) if kind == 'lp' else delta_tl(D, z))
        out[k] = {'kind': kind, 'D': D, 'z': z, 'Delta': d, 'tR': r['tR'], 'ln1_tJ': r['ln1_tJ'], 'tR_over_Delta': r['tR'] / max(d, 1e-12)}
        json.dump(out, open(fn, 'w'), indent=1)
        print('%-14s Delta %8.3f  tR %8.3f  ln %8.2f  tR/Delta %7.3f' % (k, d, r['tR'], r['ln1_tJ'], r['tR'] / max(d, 1e-12)), flush=True)
