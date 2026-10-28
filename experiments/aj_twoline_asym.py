"""Stage AJ, Proposition AJ3 (two w-lines + midpoint, t = 1): asymptotics derived from the closed form,
  tR = D^2/2 + 1 - c1 D^2 e^{-Dz} (1 + o(1)) + o(1),          c1 = (2/3)^{3/2} / (2 pi)^{3/2} = 0.03456,
  ln(1/tJ) = D^2/2 + 2 D z + ln D + C0 + o(1),               C0 = (3/2) ln 2pi - ln(pi^{3/2}/4) + ln 2 = 3.1192,
  max_z [tR - ln(1/tJ)] = 4.003 - 5 ln D + o(1)  at  D z* = ln(c1 D^2/2)       (-1 - 2 ln(c1/2) - C0 = 4.003).
Checked against the separate closed-form evaluator ai_twoline_exact at z = z*(D).  -> results/aj_twoline_asym.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
from ai_twoline_exact import evaluate

c1 = (2 / 3) ** 1.5 / (2 * np.pi) ** 1.5; C0 = 1.5 * np.log(2 * np.pi) - np.log(np.pi ** 1.5 / 4) + np.log(2); K0 = -1 - 2 * np.log(c1 / 2) - C0

if __name__ == '__main__':
    fn = 'results/aj_twoline_asym.json'; out = json.load(open(fn)) if os.path.exists(fn) else {'c1': c1, 'C0': C0, 'K0': K0, 'rows': {}}
    for D in (10.0, 20.0, 30.0, 40.0, 60.0):
        k = '%g' % D
        if k in out['rows']: continue
        z = np.log(c1 * D * D / 2) / D; r = evaluate(D, z)
        pred = {'tR': D * D / 2 + 1 - c1 * D * D * np.exp(-D * z), 'ln1_tJ': D * D / 2 + 2 * D * z + np.log(D) + C0, 'margin': K0 - 5 * np.log(D)}
        out['rows'][k] = {'D': D, 'z_star': z, **r, 'margin': r['tR'] - r['ln1_tJ'], 'tR/ln': r['tR'] / r['ln1_tJ'], 'pred': pred}
        json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('D %3.0f z* %.4f  tR %9.3f (pred %9.3f)  ln %9.3f (pred %9.3f)  margin %7.3f (pred %7.3f)  tR/ln %.4f' % (D, z, r['tR'], pred['tR'], r['ln1_tJ'], pred['ln1_tJ'], r['tR'] - r['ln1_tJ'], pred['margin'], r['tR'] / r['ln1_tJ']), flush=True)
