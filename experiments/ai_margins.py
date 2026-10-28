"""Stage AI: margins of the static targets on all evaluated states and on the line + point families.
  T2 margin: A'_needed = tR - B' ln(1/(tJ_ww)) (B' = 1; also the smallest B' with A' = 7 fixed),  T1 margin: Lambda - 4 ln(1/(tJ_ww)) (= 4 F).
Stored states: stage-AC heat-flow rows (lattices, points, mixtures, strips; 58 states x 13 blurs), stage-AD lattice ladders (tR, J only),
stage-AF strip optima (t = sigma^2/4, /2, 1).  Families (ai_mix, conservative unnormalised ln): one line + one point (D, z),
two lines + one point, one line + K points (dotted), two lines + two points; Nelder-Mead maximisation of each margin.
-> results/ai_margins.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, json, time
import numpy as np
from scipy.optimize import minimize
from ai_mixlog import targets          # fully log-domain (ai_mix underflows when the valley is far, z >~ D)


def stored():
    rows = []
    for mode in ('lattice', 'points', 'mix', 'strip'):
        for k, v in json.load(open('results/ac_heat_%s.json' % mode)).items():
            for key, r in v['rows'].items():
                l = np.log(1 / (r['t'] * r['J'])); rows.append({'src': 'AC:%s:%s' % (mode, k), 'm': key, 'tR': r['t'] * r['R'], 'Lambda': r['Lambda'], 'ln': float(l)})
    for k, v in json.load(open('results/ad_qlc_lattice.json')).items():
        for r in v['rows']:
            if r: rows.append({'src': 'AD-ladder:' + k, 'm': r['t'], 'tR': r['tR'], 'Lambda': None, 'ln': float(np.log(1 / (r['t'] * r['J'])))})
    for k, v in json.load(open('results/af_strip.json')).items():
        rows.append({'src': 'AF-strip:' + k, 'm': v['m'], 'tR': v['tR'], 'Lambda': None, 'ln': float(np.log(1 / v['tJ']))})
    return rows


def _z(D, x):                                     # keep the valley offset in [-D/2, 2D] (beyond 2D: tR is flat, ln grows) and D <= 30
    return float(np.clip(x, -0.5 * D, 2.0 * D))


def fam(kind, x):
    x = list(x); x[0] = float(np.clip(x[0], -28.0, 28.0)); x[1] = x[1] if kind == 'two_lines_point' else _z(abs(x[0]) + 2, x[1])
    if kind == 'two_lines_point': x[2] = _z(abs(x[0]) + 2, x[2])
    if kind == 'line_point':
        D, z = abs(x[0]) + 2, x[1]; return [(0.0, 0.0)], [(D, 0.0, D * (z + D / 2))]
    if kind == 'two_lines_point':
        D, D2, z, lr = abs(x[0]) + 2, abs(x[1]) + 2, x[2], x[3]; return [(0.0, 0.0), (D + D2, lr)], [(D, 0.0, D * (z + D / 2))]
    if kind == 'line_dotted':
        D, z, s, K = abs(x[0]) + 2, x[1], abs(x[2]) + 0.5, 5; L = D * (z + D / 2); return [(0.0, 0.0)], [(D, (k - K // 2) * s, L) for k in range(K)]
    D, z, dw, D2 = abs(x[0]) + 2, x[1], x[2], abs(x[3]) + 2; L = D * (z + D / 2)          # two lines + two points (one near each line)
    return [(0.0, 0.0), (2 * D + D2, 0.0)], [(D, 0.0, L), (D + D2, dw, L)]


X0 = {'line_point': [6.0, 0.0], 'two_lines_point': [6.0, 6.0, 0.0, 0.0], 'line_dotted': [6.0, 0.0, 2.0], 'two_lines_two_points': [6.0, 0.0, 1.0, 4.0]}

if __name__ == '__main__':
    fn = 'results/ai_margins.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    if 'stored' not in out:
        S = stored(); out['stored_n'] = len(S)
        out['stored'] = {'T2_Aneeded_B1': max((r['tR'] - r['ln'], r['src'], r['m']) for r in S),
                         'T1_Lambda_minus_4ln': max((r['Lambda'] - 4 * r['ln'], r['src'], r['m']) for r in S if r['Lambda'] is not None),
                         'T2_Bneeded_A7': max(((r['tR'] - 7) / r['ln'], r['src'], r['m']) for r in S if r['ln'] > 0.5),
                         'T1_Bneeded_A2': max(((r['Lambda'] - 2) / r['ln'], r['src'], r['m']) for r in S if r['Lambda'] is not None and r['ln'] > 0.5),
                         'max_tR': max((r['tR'], r['src']) for r in S)}
        out['stored_scatter'] = [[r['ln'], r['tR'], r['Lambda']] for r in S]
        json.dump(out, open(fn, 'w'), indent=1, default=float); print('stored:', json.dumps(out['stored'], default=float), flush=True)
    for kind in X0:
        for what in ('T2', 'T1'):
            key = '%s_%s' % (kind, what)
            if key in out: continue
            t0 = time.time()
            def obj(x):
                try:
                    r = targets(*fam(kind, x)); v = r['tR'] - r['ln1_tJ'] if what == 'T2' else r['Lambda'] - 4 * r['ln1_tJ']
                    return -v if np.isfinite(v) else 1e3
                except Exception: return 1e3
            best = None
            for x0 in (X0[kind], list(np.array(X0[kind]) * 1.5), list(np.array(X0[kind]) * 0.7)):
                r = minimize(obj, np.array(x0, float), method='Nelder-Mead', options={'maxfev': 120, 'xatol': 1e-4, 'fatol': 1e-6})
                if best is None or r.fun < best.fun: best = r
            R = targets(*fam(kind, best.x))
            out[key] = {'x': list(map(float, best.x)), 'margin': float(-best.fun), **{k: float(v) for k, v in R.items()}, 'seconds': time.time() - t0}
            json.dump(out, open(fn, 'w'), indent=1, default=float)
            print('%-26s max margin %.3f  (tR %.3f, Lambda %.3f, ln %.3f)  x %s  (%.0fs)' % (key, out[key]['margin'], R['tR'], R['Lambda'], R['ln1_tJ'], np.round(best.x, 3), out[key]['seconds']), flush=True)
    # the D, z map of the one-line family
    if 'line_point_map' not in out:
        rows = []
        for D in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0):
            for zr in (-0.25, 0.0, 0.25, 0.5, 1.0):
                r = targets(*fam('line_point', [D - 2, zr * D])); rows.append({'D': D, 'z': zr * D, **{k: float(v) for k, v in r.items()}})
        out['line_point_map'] = rows; json.dump(out, open(fn, 'w'), indent=1, default=float)
        print('line_point map: max T2 margin %.3f, max T1 margin %.3f' % (max(r['tR'] - r['ln1_tJ'] for r in rows), max(r['Lambda'] - 4 * r['ln1_tJ'] for r in rows)))
