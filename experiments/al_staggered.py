"""Stage AL, plan A, next probe: two w-lines + TWO dotted midpoint rows at v = D -+ d, staggered by half a period in w (sign cancellation
of the w-score along v, i.e. the oblique-lattice mechanism, combined with the hidden mass).  Measures kappa_v(tau_v), Psi_v =
min kappa_v/(1 - tau_v), Delta_1/2, tR, ln(1/tJ).  -> results/al_staggered.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from al_aniso import chain
from ak_dotted import lnJ


def config(D, a, z, d=1.0):
    Lp = D * (z + D / 2); La = Lp - np.log(a) + np.log(2.0); N = int(np.ceil(12.0 / a))       # the two rows share the linear density e^{-L'}
    pts = [(D - d, j * a, La) for j in range(-N, N + 1)] + [(D + d, (j + 0.5) * a, La) for j in range(-N, N + 1)]
    return [(0.0, 0.0), (2 * D, 0.0)], pts


if __name__ == '__main__':
    fn = 'results/al_staggered.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for D, a in ((8.0, 2.0), (14.0, 1.5), (20.0, 1.2)):
        k = '%g_%g' % (D, a)
        if k in out: continue
        lines, pts = config(D, a, 0.1); hv = min(0.02, 0.25 / D)
        yv = np.arange(-8.0, 2 * D + 8.0, hv); yw = (np.arange(20) + 0.5) * a / 20
        lJ1, _ = lnJ(lines, pts, yv, yw, 1.0); lJh, _ = lnJ(lines, pts, yv, yw, 0.5)
        r = {'D': D, 'a': a, 'Delta_half': lJh - lJ1, 'taus': {}}
        for tau_v in (0.5, 0.75, 0.9, 0.95):
            o = chain(lines, pts, yv, yw, tau_v=tau_v); r['taus']['%g' % tau_v] = o
            print(k, tau_v, 'kappa_v %.4g tR %.3f (vw %.3f ww %.3f) ln %.2f' % (o['kappa_v'], o['tR'], o['tR_vw'], o['tR_ww'], o['ln1_tJ']), flush=True)
        o5 = r['taus']['0.5']; r.update({'tR': o5['tR'], 'tR_vw': o5['tR_vw'], 'ln1_tJ': o5['ln1_tJ']})
        r['Psi_v'] = min(v['kappa_v'] / (1 - float(tt)) for tt, v in r['taus'].items()); r['tR_over_Delta'] = r['tR'] / r['Delta_half']
        r['Bneeded_AL1_best'] = min(v['Bneeded_AL1'] for v in r['taus'].values())
        out[k] = r; json.dump(out, open(fn, 'w'), indent=1, default=float)
        print(k, 'Delta %.3f tR %.3f ln %.2f Psi_v %.4g tR/Delta %.3f B_needed(AL1, best tau_v) %.4g' % (r['Delta_half'], r['tR'], r['ln1_tJ'], r['Psi_v'], r['tR_over_Delta'], r['Bneeded_AL1_best']), flush=True)
