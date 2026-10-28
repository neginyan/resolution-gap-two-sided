"""Stage BA §2, candidate 2 (route 1 of the map): a two-window strip and a chirped strip, true g minimised with paper E's evaluator.

Two-window strip: one background N(w; 0, Lw^2), two ripple windows at w = -+d/2 with their own widths ell_1, ell_2 and wavenumbers
  k_1 = m_1 k_b, k_2 = m_2 k_b ((m_1, m_2) = (2, 3): the windows at different w carry different k0).
Chirped strip: a staircase of 2R + 1 overlapping windows at w_r = r Delta (r = -R..R) with wavenumbers (m_0 + r) k_b, i.e. the local
  wavenumber changes linearly across the window, k(w) ~ k0 (1 + w/(m_0 Delta)) (the evaluator needs commensurate wavenumbers; a continuous
  chirp is not v-periodic).
Positivity: sum_r c_r e^{-(w - w_r)^2/2 ell_r^2} <= 1 for every w (checked on a grid) => rho >= 0.
Free parameters: k0 (central), the window widths, spacing, amplitudes, M (Lw = M ell), eta (Lv = eta ell); constraints as paper E's strip
  optimiser (Lw < 2, y/sigma <= 0.5, Lv >= 30/k0 sigma).
usage: python ba_window.py two|chirp SIGMA [MAXFEV] -> results/ba_window_<kind>_<sigma>.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
from ba_tools import gap, branches, strip_opt


FAST = dict(nper=12, coarse=100, n0=11, nv=16)       # optimisation; the minima are re-evaluated at the default (full) settings


def save(fn, val): json.dump(val, open(fn + '.tmp', 'w'), indent=1, default=float); os.replace(fn + '.tmp', fn)


def positive(win, Lw_s):
    w = np.linspace(-8 * Lw_s, 8 * Lw_s, 4001)
    return float(max(sum(c * np.exp(-(w - wr) ** 2 / (2 * er ** 2)) for c, wr, er, _ in win)))


def two_state(x, sig):
    k0, lam1, lam2, dd, M, eta = np.exp(x[:6]); c1, c2 = 1 / (1 + np.exp(-x[6])), 1 / (1 + np.exp(-x[7])); m1, m2 = 2, 3
    kb = k0 / 2.5; lam = np.sqrt(lam1 * lam2)                       # central wavenumber k0 between k_1 = 0.8 k0 and k_2 = 1.2 k0
    win = [(c1, -dd * lam / 2, lam1, m1), (c2, dd * lam / 2, lam2, m2)]; Lb = M * lam
    pmax = positive(win, Lb)
    if pmax > 1: win = [(c / pmax, wr, er, m) for c, wr, er, m in win]
    return kb, [(1.0, 0.0, Lb)], win, eta * lam / np.sqrt(sig), {'k1': m1 * kb, 'k2': m2 * kb, 'lam1': lam1, 'lam2': lam2, 'd_over_lam': dd, 'M': M, 'eta': eta}


def chirp_state(x, sig, m0=4, R=1):
    k0, lam, dd, M, eta = np.exp(x[:5]); c = 1 / (1 + np.exp(-x[5])); kb = k0 / m0
    win = [(c, r * dd * lam, lam, m0 + r) for r in range(-R, R + 1)]; Lb = M * lam
    pmax = positive(win, Lb); win = [(cc / pmax, wr, er, m) for cc, wr, er, m in win] if pmax > 1 else win
    return kb, [(1.0, 0.0, Lb)], win, eta * lam / np.sqrt(sig), {'k0': k0, 'lam': lam, 'Delta_over_lam': dd, 'M': M, 'eta': eta, 'm0': m0, 'R': R,
                                                                  'chirp_dk_over_k_per_ell': 1 / (m0 * dd)}


def run(kind, sig, maxfev):
    P = strip_opt(sig); s = np.sqrt(sig); log = []; fn = 'results/ba_window_%s_%g.json' % (kind, sig)
    st = (lambda x: two_state(x, sig)) if kind == 'two' else (lambda x: chirp_state(x, sig))

    def obj(x):
        try:
            kb, bg, win, Lv_sig, info = st(x); Lb = bg[0][2]
            if not (0.05 < kb < 12 and Lb * s < 2.0 and Lv_sig > 30 / kb and all(0.1 < w[2] < 1e3 for w in win)): return 60.0
            o = gap(sig, bg, win, kb, Lv_sig, **FAST); g = float(o['gap_over_sigma'])
            if not np.isfinite(g) or o['y_over_sigma'] > 0.5: return 40.0
            log.append(g); return g
        except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
            return 70.0

    eta0 = P['Lv_sig'] * sig / (P['lam'] * s); lc = np.log(P['c'] / (1 - P['c']))
    if kind == 'two':
        starts = [np.array([np.log(P['k0']), np.log(P['lam']), np.log(P['lam']), np.log(d), np.log(P['M']), np.log(eta0), lc, lc]) for d in (1.0, 4.0)]
    else:
        starts = [np.array([np.log(P['k0']), np.log(P['lam'] / 2), np.log(d), np.log(P['M'] * 2), np.log(eta0 * 2), lc]) for d in (0.7, 1.5)]
    runs = []
    for x0 in starts:
        g0 = obj(x0); t0 = time.time()
        r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-6})
        kb, bg, win, Lv_sig, info = st(r.x); g_full = float(gap(sig, bg, win, kb, Lv_sig)['gap_over_sigma'])
        runs.append({'g_start': g0, 'g_min_fast': float(r.fun), 'g_min': g_full, 'x': list(map(float, r.x)), 'nfev': r.nfev, 'seconds': time.time() - t0, 'kb': kb, 'Lv_sig': Lv_sig,
                     'windows': [list(map(float, w)) for w in win], 'Lb_sqrt_sig': bg[0][2], **info})
        print('%s sigma %g: start g %.6g -> min %.6g (full %.6g; strip g_min %.6g)  %s  %.0fs' % (kind, sig, g0, r.fun, g_full, P['g'], {k: round(float(v), 4) for k, v in info.items()}, time.time() - t0), flush=True)
        save(fn, {'strip_g': P['g'], 'runs': runs})
    best = min(runs, key=lambda q: q['g_min']); kb, bg, win, Lv_sig, info = st(np.array(best['x']))
    b = branches(sig, bg, win, kb, Lv_sig)
    Ps = {'k0': P['k0'], 'lam': P['lam'], 'M': P['M'], 'Lv_sig': P['Lv_sig'], 'c': P['c']}
    bs = branches(sig, [(1.0, 0.0, P['lam'] * P['M'])], [(P['c'], 0.0, P['lam'], 1)], P['k0'], P['Lv_sig'])
    save(fn, {'strip_g': P['g'], 'runs': runs, 'best_branches': b, 'strip_branches': bs, 'strip_params': Ps})
    print('%s sigma %g best g %.6g: branches %s | strip %s' % (kind, sig, best['g_min'], b, bs), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    run(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 150)
