"""Stage BF: the rigorous (upward) reading of stage AY's F-iteration, for the strips (ay_certificate.py) and the periodic windows (bf_cert_p.py).

Theorem AY1's certificate iterated F downward from F = 1 (a fixed point: "if F_c >= F then F_c >= (1 - |eps|) F_int(F) >= F").  Read as a proof, each step is an
implication from an assumed lower bound, so the iteration has to start from an assumed F_0 and may only go up.  Here: the smallest F_0 on a grid from which the
upward bootstrap closes at the full F_int, for the strips (no x >= 1 assumption; x_low from U0) at 1e-24, 1e-96, 1e-1024, eps = 0.01.  For the periodic windows
with (S6) x >= 1 the start F_0 = 1.02 G suffices (bf_cert_p.py).  -> results/bf_ay_bootstrap.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from ay_certificate import certificate

if __name__ == '__main__':
    out = {}
    for e in (24, 96, 1024):
        L = e * np.log(10); rows = []; first = None
        for f0 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            F = f0; r = None
            for k in range(40):
                r = certificate(L, eps=0.01, F=F, iterate=False); Fn = 0.99 * r['F_int']
                if not r['ok'] or Fn <= F * (1 + 1e-12): break
                F = Fn
            closed = bool(r['ok'] and F > 0.98)
            rows.append({'F0': f0, 'F_final': F, 'closed': closed})
            if closed and first is None: first = f0
        full = certificate(L, eps=0.01)
        out['1e-%d' % e] = {'rows': rows, 'smallest_F0': first, 'G': full['G'], 'AY1_fixed_point_F': full['F_used'], 'AY1_g_sqrt_ln': full['g_bound_sqrt_ln']}
        print('strips 1e-%-5d: upward bootstrap closes from F_0 = %s (G = %.4f); fixed point of stage AY F = %.6f' % (e, first, full['G'], full['F_used']), flush=True)
    json.dump(out, open('results/bf_ay_bootstrap.json', 'w'), indent=1, default=float)
