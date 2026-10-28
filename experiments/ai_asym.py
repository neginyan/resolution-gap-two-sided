"""Stage AI: large-D asymptotics of tR/ln(1/tJ) and Lambda/ln(1/tJ) on the w-line + point family, z = alpha D (independent evaluator
ai_lp_exact, fully log-domain).  Asymptotic prediction (stage-AI note): ln(1/tJ) ~ (D+z)^2/2 (z <= D), 2Dz (z >= D);
tR ~ D^2 (1/(1-a) + c)/(1/a + 1/(1-a) + c) for a < 1, D^2 + O(1) for a > 1;  sup_a lim tR/ln = 1/2 (approached at a -> 1).
-> results/ai_asym.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
from ai_lp_exact import evaluate

if __name__ == '__main__':
    fn = 'results/ai_asym.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for D in (8.0, 16.0, 24.0, 32.0, 48.0):
        for a in (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5):
            k = '%g_%g' % (D, a)
            if k in out: continue
            r = evaluate(D, a * D, Wmax=8.0, nw=801); r['tR/ln'] = r['tR'] / r['ln1_tJ']; r['Lambda/ln'] = r['Lambda'] / r['ln1_tJ']
            out[k] = r; json.dump(out, open(fn, 'w'), indent=1)
            print(k, 'tR %.2f ln %.1f tR/ln %.4f Lambda/ln %.4f' % (r['tR'], r['ln1_tJ'], r['tR/ln'], r['Lambda/ln']), flush=True)
