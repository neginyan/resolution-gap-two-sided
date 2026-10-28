"""Stage BB §1: global checks of the stage-BA minimisations — random multi-starts (N starts; each is the best of 6 admissible random draws over broad
parameter boxes, followed by a Nelder-Mead, fast evaluator settings; the overall best re-evaluated at full settings) for the families of the three candidates:
  2pi   : a window at z = 2 pi inside one (H1)-admissible background (ba_2pi, frame shifted by -2 sqrt2 pi, exact by Lemma BA1);
  two   : two windows with k_1 = 0.8 k0, k_2 = 1.2 k0 (ba_window.two_state);
  chirp : three-window staircase 3k_b, 4k_b, 5k_b (ba_window.chirp_state);
  flat  : the flat ripple without window edges (ba_hidden control; the rows only raise g, stage BA), and 'rows': random hidden-row
          configurations (n, d, beta) added at the flat minimum.
usage: python bb_multistart.py FAMILY SIGMA [NSTART] [MAXFEV] -> results/bb_ms_<family>_<sigma>.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
from ba_tools import gap, strip_opt
from ba_window import two_state, chirp_state, FAST
from ba_hidden import g_of, FULL as HFULL

W2 = 2 * np.sqrt(2) * np.pi; BIG = 1e300; NPRE = 6


def save(fn, val): json.dump(val, open(fn + '.tmp', 'w'), indent=1, default=float); os.replace(fn + '.tmp', fn)


def family(name, sig):
    s = np.sqrt(sig)
    if name in ('two', 'chirp'):
        st = (lambda x: two_state(x, sig)) if name == 'two' else (lambda x: chirp_state(x, sig))

        def obj(x, full=False):
            try:
                kb, bg, win, Lv_sig, info = st(x); Lb = bg[0][2]
                if not (0.05 < kb < 12 and Lb * s < 2.0 and Lv_sig > 30 / kb and all(0.1 < w[2] < 1e3 for w in win)): return BIG
                o = gap(sig, bg, win, kb, Lv_sig, **({} if full else FAST)); g = float(o['gap_over_sigma'])
                return g if np.isfinite(g) and o['y_over_sigma'] <= 0.5 else BIG
            except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
                return BIG

        def sample(rng):
            lam = np.exp(rng.uniform(np.log(0.3), np.log(5))); Mmax = min(1e5, 1.9 / (lam * s))
            M = np.exp(rng.uniform(np.log(3), np.log(Mmax))); k0 = rng.uniform(0.7, 3.5); eta = np.exp(rng.uniform(np.log(0.02), np.log(1.5)))
            if name == 'two':
                return np.array([np.log(k0), np.log(lam), np.log(lam * np.exp(rng.uniform(-1, 1))), rng.uniform(np.log(0.1), np.log(5)), np.log(M), np.log(eta),
                                 rng.uniform(-1, 6), rng.uniform(-1, 6)])
            return np.array([np.log(k0), np.log(lam), rng.uniform(np.log(0.2), np.log(3)), np.log(M), np.log(eta), rng.uniform(-1, 6)])
        return obj, sample

    if name == '2pi':
        def state(x):
            k0, lam, Lw, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4])); Lv = eta * lam * s
            cb = np.sqrt(max(2 * np.pi ** 2 - Lw ** 2 - Lv ** 2, 0.0)) * np.tanh(x[5])
            return k0, lam, Lw, Lv / sig, c, cb

        def obj(x, full=False):
            try:
                k0, lam, Lw, Lv_sig, c, cb = state(x)
                if not (0.3 < k0 < 12 and 0.1 < lam < 1e3 and lam * s < Lw < np.sqrt(2) * np.pi and Lv_sig > 30 / k0): return BIG
                o = gap(sig, [(1.0, (cb - W2) / s, Lw / s)], [(c, 0.0, lam, 1)], k0, Lv_sig); g = float(o['gap_over_sigma'])
                return g if np.isfinite(g) and o['y_over_sigma'] <= 0.5 else BIG
            except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
                return BIG

        def sample(rng):
            return np.array([np.log(rng.uniform(0.4, 3)), rng.uniform(np.log(0.3), np.log(5)), rng.uniform(np.log(0.3), np.log(4.4)),
                             rng.uniform(np.log(0.02), np.log(1.5)), rng.uniform(-1, 6), rng.uniform(-1, 3)])
        return obj, sample

    if name == 'flat':
        def obj(x, full=False):
            try:
                k0, Lw, Lv_sig = np.exp(x[:3]); c = 1 / (1 + np.exp(-x[3]))
                if not (0.3 < k0 < 12 and 2 * sig < Lw < 2.0 and Lv_sig > 30 / k0): return BIG
                g, o = g_of(sig, k0, Lw, Lv_sig, c, None, HFULL if full else {'nper': 12, 'coarse': 100, 'n0': 11, 'nv': 16})
                return g if np.isfinite(g) else BIG
            except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
                return BIG

        def sample(rng):
            return np.array([np.log(rng.uniform(0.5, 4)), rng.uniform(np.log(1.5 * s), np.log(50 * s)), rng.uniform(np.log(50), np.log(10 / s)), rng.uniform(0, 8)])
        return obj, sample
    raise ValueError(name)


def run(name, sig, nstart, maxfev):
    fn = 'results/bb_ms_%s_%g.json' % (name, sig); R = json.load(open(fn)) if os.path.exists(fn) else {'runs': []}
    R['strip_g'] = strip_opt(sig)['g']; R['nstart'] = nstart; R['maxfev'] = maxfev
    obj, sample = family(name, sig); rng = np.random.default_rng(12345 + int(-np.log10(sig)))
    for i in range(nstart):
        cand = []; tries = 0
        while len(cand) < NPRE and tries < 200:   # NPRE admissible random draws; the best one starts the local search
            x0 = sample(rng); tries += 1
            if i >= len(R['runs']):
                g0 = obj(x0)
                if g0 < BIG: cand.append((g0, x0))
            else: cand.append((0.0, x0))
        if i < len(R['runs']): continue
        if not cand: continue
        g0, x0 = min(cand, key=lambda c: c[0])
        t0 = time.time()
        r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-6})
        R['runs'].append({'i': i, 'g_start': g0, 'g_min_fast': float(r.fun), 'x': list(map(float, r.x)), 'nfev': r.nfev, 'seconds': time.time() - t0}); save(fn, R)
        print('%s sigma %g start %d: g %.4g -> %.6g (strip %.6g)  %.0fs' % (name, sig, i, g0, r.fun, R['strip_g'], time.time() - t0), flush=True)
    ok = [q for q in R['runs'] if q['g_min_fast'] < BIG]
    if not ok: save(fn, R); print('%s sigma %g: no admissible start' % (name, sig)); return
    best = min(ok, key=lambda q: q['g_min_fast']); R['best_fast'] = best['g_min_fast']; R['best_full'] = obj(np.array(best['x']), full=True)
    R['n_valid'] = len(ok); R['sorted_minima_fast'] = sorted(q['g_min_fast'] for q in ok)
    if name == 'flat':      # random hidden-row configurations at the flat minimum
        k0, Lw, Lv_sig = np.exp(best['x'][:3]); c = 1 / (1 + np.exp(-best['x'][3])); R.setdefault('rows', [])
        for j in range(len(R['rows']), 6):
            n = int(rng.integers(1, 3)); d = float(rng.uniform(0.1, 1.0)); beta = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.3))))
            t0 = time.time(); g, _ = g_of(sig, k0, Lw, Lv_sig, c, (n, d, beta))
            R['rows'].append({'n': n, 'd': d, 'beta': beta, 'g': g, 'seconds': time.time() - t0}); save(fn, R)
            print('rows sigma %g n %d d %.2f beta %.3g: g %.6g (flat %.6g)' % (sig, n, d, beta, g, R['best_full']), flush=True)
    save(fn, R)
    print('%s sigma %g: best of %d valid starts %.6g (full %.6g), strip g_min %.6g; minima %s' % (name, sig, len(ok), R['best_fast'], R['best_full'], R['strip_g'],
          ', '.join('%.4g' % v for v in R['sorted_minima_fast'][:8])), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    run(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24, int(sys.argv[4]) if len(sys.argv) > 4 else 40)
