"""Stage AI: structured attempts to break (T2) tR <= A' + B' ln(1/(tJ_ww)) (B' = 1) with more lines / more points, at the regime where
one line + one point is most dangerous (z ~ D, asymptotic tR/ln -> 1/2).  All with ai_mixlog (log domain, conservative ln).  t = 1.
 (a) one line + K points along w (spacing s): does a dotted row raise tR/ln?
 (b) two lines (v = 0 and v = 2D, densities 1 and e^{lr}) + one point at v = D (valleys on both sides)
 (c) one line + two points on the v axis (D and D2 > D), equal or different offsets
 (d) one line + two points at +-w0 (a pair facing the line)
-> results/ai_break.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, itertools
import numpy as np
from ai_mixlog import targets


def lz(D, a): return D * (a * D + D / 2)


def cases():
    for D, a, K, s in itertools.product((8.0, 14.0), (0.5, 1.0), (3, 5), (0.5, 1.5, 4.0)):
        yield 'dotted_D%g_a%g_K%d_s%g' % (D, a, K, s), [(0.0, 0.0)], [(D, (k - K // 2) * s, lz(D, a)) for k in range(K)]
    for D, a, lr in itertools.product((8.0, 14.0), (0.0, 0.5, 1.0), (0.0, -20.0)):
        yield 'twolines_D%g_a%g_lr%g' % (D, a, lr), [(0.0, 0.0), (2 * D, lr)], [(D, 0.0, lz(D, a))]
    for D, a, f, a2 in itertools.product((8.0, 14.0), (0.5, 1.0), (1.5, 2.0), (0.5, 1.0)):
        D2 = f * D; yield 'twopts_D%g_a%g_D2%g_a2%g' % (D, a, D2, a2), [(0.0, 0.0)], [(D, 0.0, lz(D, a)), (D2, 0.0, lz(D2, a2))]
    for D, a, w0 in itertools.product((8.0, 14.0), (0.5, 1.0), (1.0, 3.0, 6.0)):
        yield 'pair_D%g_a%g_w%g' % (D, a, w0), [(0.0, 0.0)], [(D, w0, lz(D, a)), (D, -w0, lz(D, a))]


if __name__ == '__main__':
    fn = 'results/ai_break.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for name, L, P in cases():
        if name in out: continue
        try: r = targets(L, P, h=0.05)
        except MemoryError: r = targets(L, P, h=0.08)
        r['tR/ln'] = r['tR'] / r['ln1_tJ']; r['T2_margin'] = r['tR'] - r['ln1_tJ']; r['Lambda/ln'] = r['Lambda'] / r['ln1_tJ']
        out[name] = r; json.dump(out, open(fn, 'w'), indent=1)
        print('%-32s tR %9.2f ln %9.1f tR/ln %.4f  Lambda/ln %.4f  T2 margin %.1f' % (name, r['tR'], r['ln1_tJ'], r['tR/ln'], r['Lambda/ln'], r['T2_margin']), flush=True)
