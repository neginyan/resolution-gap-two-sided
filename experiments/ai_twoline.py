"""Stage AI: the two-line + midpoint family (lines v = 0 and v = 2D + delta, densities 1 and e^{lr}; point at v = D, weight e^{-L},
L = D(z + D/2)), the most dangerous family found for (T2) with B' = 1: tR/ln(1/tJ) -> 1 as D grows, tR - ln(1/tJ) stays negative.
(1) scan D = 8 ... 32 at the best z (bounded scalar search in z), (2) Nelder-Mead in (z, lr, delta) at D = 14 and 20 for max(tR - ln),
(3) D = 26, 32, 40 at the best z of (1) with the separate fine-grid closed form ai_twoline_exact (the h = 0.04 mixture grid under-resolves
the valley, width 1/D, for D >= 26).  -> results/ai_twoline.json, results/ai_twoline_exact_large.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from ai_mixlog import targets


def ev(D, z, lr=0.0, delta=0.0):
    h = 0.05 if D < 20 else 0.04
    r = targets([(0.0, 0.0), (2 * D + delta, lr)], [(D, 0.0, D * (z + D / 2))], h=h)
    r['T2_margin'] = r['tR'] - r['ln1_tJ']; r['tR/ln'] = r['tR'] / r['ln1_tJ']; r['T1_margin'] = r['Lambda'] - 4 * r['ln1_tJ']; return r


if __name__ == '__main__':
    fn = 'results/ai_twoline.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for D in (8.0, 14.0, 20.0, 26.0, 32.0):
        k = 'scan_D%g' % D
        if k in out: continue
        res = minimize_scalar(lambda z: -ev(D, z)['T2_margin'], bounds=(-0.5, 1.0), method='bounded', options={'xatol': 0.01})
        r = ev(D, res.x); r['z'] = float(res.x); out[k] = r; json.dump(out, open(fn, 'w'), indent=1)
        print(k, 'z %.3f tR %.2f ln %.2f tR/ln %.4f T2 margin %.2f T1 margin %.1f' % (r['z'], r['tR'], r['ln1_tJ'], r['tR/ln'], r['T2_margin'], r['T1_margin']), flush=True)
    for D in (14.0, 20.0):
        k = 'nm_D%g' % D
        if k in out: continue
        z0 = out['scan_D%g' % D]['z']
        f = lambda x: -ev(D, x[0], x[1], x[2])['T2_margin']
        res = minimize(f, [z0, 0.0, 0.0], method='Nelder-Mead', options={'maxfev': 70, 'initial_simplex': [[z0, 0, 0], [z0 + 0.2, 0, 0], [z0, 2.0, 0], [z0, 0, 2.0]]})
        r = ev(D, *res.x); r['x'] = list(map(float, res.x)); out[k] = r; json.dump(out, open(fn, 'w'), indent=1)
        print(k, 'x', np.round(res.x, 3), 'tR %.2f ln %.2f tR/ln %.4f T2 margin %.2f' % (r['tR'], r['ln1_tJ'], r['tR/ln'], r['T2_margin']), flush=True)
    fn2 = 'results/ai_twoline_exact_large.json'; out2 = json.load(open(fn2)) if os.path.exists(fn2) else {}
    from ai_twoline_exact import evaluate
    for D, z in ((26.0, 0.09), (32.0, 0.08), (40.0, 0.07)):
        k = '%g_%g' % (D, z)
        if k in out2: continue
        r = evaluate(D, z); r['tR/ln'] = r['tR'] / r['ln1_tJ']; r['T2_margin'] = r['tR'] - r['ln1_tJ']; out2[k] = r
        json.dump(out2, open(fn2, 'w'), indent=1); print('exact', k, r, flush=True)
