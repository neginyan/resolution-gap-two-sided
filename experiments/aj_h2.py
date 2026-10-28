"""Stage AJ, side route: under (H2) Var X_z <= pi^2 the w-line has physical length O(pi): in units of sqrt(t) = sigma its Gaussian envelope has
V = (pi/sigma)^2.  Largest curvature t^2 R' (and t^2 R' - A tR, A = 1, 2) over the w-line + point family (stage AH evaluator ah_counter.row,
probability measures) as a function of sigma, to see whether kappa(sigma) = polylog(1/sigma) in t^2 R' <= A tR + kappa(sigma).
-> results/aj_h2.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from ah_counter import row

if __name__ == '__main__':
    fn = 'results/aj_h2.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for s in (1e-2, 1e-4, 1e-8, 1e-16, 1e-32, 1e-64):
        k = '%g' % s
        if k in out: continue
        V = (np.pi / s) ** 2; Dmax = 2 * np.sqrt(np.log(np.pi / s)) + 3; rows = []
        for D in np.arange(2.0, Dmax + 0.01, 1.0):
            for zr in (-0.25, 0.0, 0.25, 0.5):
                r = row(float(D), zr * D, V=V); rows.append({q: float(v) for q, v in r.items()})
        best = {q: max(rows, key=lambda r: r[q]) for q in ('t2R1', 'kappa_H', 'tR')}
        out[k] = {'sigma': s, 'ln1_sigma': float(np.log(1 / s)), 'max_t2R1': best['t2R1']['t2R1'], 'at_t2R1': [best['t2R1']['D'], best['t2R1']['z']],
                  'max_kappa_H': best['kappa_H']['kappa_H'], 'max_tR': best['tR']['tR'],
                  'max_t2R1_minus_1tR': max(r['t2R1'] - r['tR'] for r in rows), 'max_t2R1_minus_2tR': max(r['t2R1'] - 2 * r['tR'] for r in rows), 'rows': rows}
        json.dump(out, open(fn, 'w'), indent=1)
        o = out[k]; print('sigma %g ln(1/s) %.1f  max t2R1 %.3f at D,z %s  max kappa_H %.2f  max tR %.2f  max(t2R1 - tR) %.3f  max(t2R1 - 2tR) %.3f' % (
            s, o['ln1_sigma'], o['max_t2R1'], o['at_t2R1'], o['max_kappa_H'], o['max_tR'], o['max_t2R1_minus_1tR'], o['max_t2R1_minus_2tR']), flush=True)
