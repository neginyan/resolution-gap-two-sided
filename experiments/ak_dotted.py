"""Stage AK: two w-lines (v = 0, 2D) + a dotted midpoint row (points (D, j a), linear density e^{-L'}, L' = D(z + D/2)): the combination of the
hidden-mass mechanism (two-line family) and w-sign cancellation (lattice).  Measures kappa_tau (several tau), Psi = min_tau kappa_tau/(1-tau),
Delta_1/2, tR, ln(1/tJ) on one w-period (rows long enough that the ends are > 10 sqrt(t) away).  Tests whether the dichotomy
[Psi <= Psi0 (Proposition AK1)] or [tR <= K Delta (heat flow)] can fail with T2 still true.  -> results/ak_dotted.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np
from scipy.special import logsumexp
from ak_abs import chain, closed_form


def config(D, a, z):
    Lp = D * (z + D / 2); La = Lp - np.log(a); N = int(np.ceil(12.0 / a))
    return [(0.0, 0.0), (2 * D, 0.0)], [(D, j * a, La) for j in range(-N, N + 1)], Lp


def lnJ(lines, pts, yv, yw, t):
    hv = np.gradient(yv); hw = yw[1] - yw[0]; acc, accM = [], []
    for w_ in yw:
        lp, lsw, *_ = closed_form(lines, pts, yv, np.full_like(yv, w_), t); acc.append(logsumexp(lp + 2 * lsw + np.log(hv * hw))); accM.append(logsumexp(lp + np.log(hv * hw)))
    return float(logsumexp(acc)), float(logsumexp(accM))


if __name__ == '__main__':
    fn = 'results/ak_dotted.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for D, a, z in ((8.0, 2.0, 0.1), (14.0, 1.5, 0.1), (20.0, 1.2, 0.1)):
        k = '%g_%g_%g' % (D, a, z)
        if k in out: continue
        lines, pts, Lp = config(D, a, z); hv = min(0.02, 0.25 / D)
        yv = np.arange(-8.0, 2 * D + 8.0, hv); yw = (np.arange(10) + 0.5) * a / 10
        lJ1, lM1 = lnJ(lines, pts, yv, yw, 1.0); lJh, lMh = lnJ(lines, pts, yv, yw, 0.5)
        r = {'D': D, 'a': a, 'z': z, 'Delta_half': lJh - lJ1, 'taus': {}}
        for tau in (0.5, 0.75, 0.9, 0.95):
            o = chain(lines, pts, yv, yw, tau=tau, ngh=16); r['taus']['%g' % tau] = o
            print(k, tau, 'kappa %.4g tR %.3f ln %.2f final/tR %.4g' % (o['kappa'], o['tR'], o['ln1_tJ'], o['step_final']), flush=True)
        r['tR'] = r['taus']['0.5']['tR']; r['ln1_tJ'] = r['taus']['0.5']['ln1_tJ']
        r['Psi'] = min(v['kappa'] / (1 - float(tt)) for tt, v in r['taus'].items()); r['tR_over_Delta'] = r['tR'] / r['Delta_half']
        r['Bneeded_AK1_best'] = min(v['bound_AK1'] / v['ln1_tJ'] for v in r['taus'].values())
        out[k] = r; json.dump(out, open(fn, 'w'), indent=1, default=float)
        print(k, 'Delta %.3f tR %.3f ln %.2f Psi %.3g tR/Delta %.3f B_needed(AK1, best tau) %.3g' % (r['Delta_half'], r['tR'], r['ln1_tJ'], r['Psi'], r['tR_over_Delta'], r['Bneeded_AK1_best']), flush=True)
