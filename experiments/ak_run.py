"""Stage AK: Proposition AK1's chain (single blur up to the sign-cancellation factor kappa) on the three families:
one w-line + point, two w-lines + midpoint, periodic point lattices (central cell).  -> results/ak_run_<kind>.json (restartable; one file per family so runs can go in parallel).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np
from ak_abs import chain


def case(kind, D, z, tau=0.5):
    if kind == 'lp':
        L = D * (z + D / 2); hv = min(0.02, 0.25 / D)
        return [(0.0, 0.0)], [(D, 0.0, L)], np.arange(-8.0, D + max(z, 0) + 10.0, hv), np.arange(-7.0, 7.05, 0.1)
    if kind == 'tl':
        L = D * (z + D / 2); hv = min(0.02, 0.25 / D)
        return [(0.0, 0.0), (2 * D, 0.0)], [(D, 0.0, L)], np.arange(-8.0, 2 * D + 8.0, hv), np.arange(-7.0, 7.05, 0.1)
    if kind == 'lat':                        # square/rect lattice: D = spacing a_v, z = spacing a_w; y over the central cell
        av, aw = D, z; nv, nw = int(np.ceil(5.0 / av)), int(np.ceil(5.0 / aw))
        pts = [(i * av, j * aw, 0.0) for i in range(-nv, nv + 1) for j in range(-nw, nw + 1)]
        return [], pts, np.arange(0, av, av / 20) + av / 40, np.arange(0, aw, aw / 20) + aw / 40


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fn = 'results/ak_run_%s.json' % tag; out = json.load(open(fn)) if os.path.exists(fn) else {}
    jobs = [('lp', 8.0, 0.0), ('lp', 12.0, 0.0), ('lp', 16.0, 0.0), ('lp', 8.0, 4.0), ('lp', 12.0, 6.0),
            ('tl', 8.0, 0.1), ('tl', 14.0, 0.1), ('tl', 20.0, 0.1), ('tl', 26.0, 0.09),
            ('lat', 1.0, 1.0), ('lat', 0.8, 0.8), ('lat', 1.5, 0.6)]
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] == sys.argv[1]]
    for kind, D, z in jobs:
        for tau in (0.5,):
            k = '%s_%g_%g_%g' % (kind, D, z, tau)
            if k in out: continue
            lines, pts, yv, yw = case(kind, D, z)
            o = chain(lines, pts, yv, yw, tau=tau, ngh=24 if kind != 'lat' else 16)
            o.update({'kind': kind, 'D': D, 'z': z, 'Bneeded_AK1': o['bound_AK1'] / o['ln1_tJ'], 'tR_over_ln': o['tR'] / o['ln1_tJ']})
            out[k] = o; json.dump(out, open(fn, 'w'), indent=1)
            print(k, ' '.join('%s %.4g' % (q, o[q]) for q in ('tR', 'ln1_tJ', 'kappa', 'step_split', 'step_CS', 'step_DV_jensen', 'step_final', 'Bneeded_AK1', 'identity_c2_rel_L2', 'tweedie_rel_L2')), flush=True)
