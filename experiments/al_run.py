"""Stage AL, plan A: Proposition AL1 (anisotropic chain) on the families: one w-line + point, two w-lines + midpoint, two w-lines + dotted
midpoint row (the gap family of stage AK), rectangular (product) and oblique point lattices.  -> results/al_run.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from al_aniso import chain


def case(kind, D, z, a=None):
    hv = min(0.02, 0.25 / max(D, 1.0))
    if kind == 'lp':
        return [(0.0, 0.0)], [(D, 0.0, D * (z + D / 2))], np.arange(-8.0, D + max(z, 0) + 10.0, hv), np.arange(-7.0, 7.05, 0.1)
    if kind == 'tl':
        return [(0.0, 0.0), (2 * D, 0.0)], [(D, 0.0, D * (z + D / 2))], np.arange(-8.0, 2 * D + 8.0, hv), np.arange(-7.0, 7.05, 0.1)
    if kind == 'dot':
        Lp = D * (z + D / 2); N = int(np.ceil(12.0 / a))
        return [(0.0, 0.0), (2 * D, 0.0)], [(D, j * a, Lp - np.log(a)) for j in range(-N, N + 1)], np.arange(-8.0, 2 * D + 8.0, hv), (np.arange(20) + 0.5) * a / 20
    if kind == 'lat':                        # basis b1 = (D, 0), b2 = (z, a): rectangular if z = 0 (product state); y over one cell
        b1, b2 = np.array([D, 0.0]), np.array([z, a]); pts = []
        for i in range(-12, 13):
            for j in range(-12, 13):
                x = i * b1 + j * b2
                if abs(x[0]) < 7 + D and abs(x[1]) < 7 + a: pts.append((x[0], x[1], 0.0))
        return [], pts, (np.arange(40) + 0.5) * D / 40, (np.arange(20) + 0.5) * a / 20


JOBS = [('lp', 8.0, 0.0, None), ('lp', 16.0, 0.0, None), ('lp', 12.0, 6.0, None), ('lp', 8.0, 8.0, None),
        ('tl', 8.0, 0.1, None), ('tl', 14.0, 0.1, None), ('tl', 20.0, 0.1, None), ('tl', 26.0, 0.09, None),
        ('dot', 8.0, 0.1, 2.0), ('dot', 14.0, 0.1, 1.5), ('dot', 20.0, 0.1, 1.2), ('dot', 26.0, 0.1, 1.0),
        ('lat', 1.0, 0.0, 1.0), ('lat', 1.0, 0.5, 1.0), ('lat', 1.3, 0.6, 1.1), ('lat', 0.8, 0.4, 0.9)]

if __name__ == '__main__':
    fn = 'results/al_run.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for kind, D, z, a in JOBS:
        for tau_v in (0.5, 0.25, 0.75):
            k = '%s_%g_%g_%s_%g' % (kind, D, z, a, tau_v)
            if k in out: continue
            lines, pts, yv, yw = case(kind, D, z, a)
            o = chain(lines, pts, yv, yw, tau_v=tau_v); o.update({'kind': kind, 'D': D, 'z': z, 'a': a})
            out[k] = o; json.dump(out, open(fn, 'w'), indent=1, default=float)
            print('%-24s tR %8.3f (vw %8.3f ww %6.3f) ln %8.2f kappa_v %.4f  B_needed %.1f (vw only %.1f)  id %.1e' % (
                k, o['tR'], o['tR_vw'], o['tR_ww'], o['ln1_tJ'], o['kappa_v'], o['Bneeded_AL1'], o['Bneeded_vw_only'], o['identity_c2v_rel_L2']), flush=True)
