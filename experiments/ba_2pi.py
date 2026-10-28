"""Stage BA §2, candidate 1: v-information at the next straight point m_z = 2 pi (route 2 of the map).

Exact symmetry (proof): f(z) = z - sin z satisfies f(z + 2 pi) = f(z) + 2 pi, and the Fisher informations and Cov(x_p, f(x_z) | y) are
unchanged by a translation of x that moves z by 2 pi (x_p moves by a constant).  So a state translated by z -> z + 2 pi has exactly the
same g; (H2) (a variance) is unchanged, (H1) (a second moment about the top, E x_z^2 <= pi^2) excludes the translated strip
(E x_z^2 ~ 4 pi^2).  Mixtures: by the mediant lemma (stage AQ) g >= min of the components'.
Numerics (paper E's evaluator):
 (i)   the strip optimum translated by z-shift m (0 ... 2 pi): g(m) and (H1);
 (ii)  mixtures: weight alpha at 2 pi, 1 - alpha at 0 (alpha < 1/4 by (H1)), identical and different strips;
 (iii) a connected state: a window at z = 2 pi (w_r = 2 sqrt2 pi) inside one broad Gaussian background (centre c_b, width Lw) with (H1)
       built in (c_b = sqrt(2 pi^2 - Lw^2 - Lv^2) tanh u); true g minimised (Nelder-Mead) and compared with the strip g_min.
       Evaluated in the frame shifted by -2 sqrt2 pi (window at 0; exact by the symmetry; checked against the direct frame at 1e-6).
usage: python ba_2pi.py PART SIGMA   (PART = scan | mix | opt)   -> results/ba_2pi_<part>_<sigma>.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
from ba_tools import gap, branches, strip_opt, moments

W2 = 2 * np.sqrt(2) * np.pi


def save(key, val):
    fn = 'results/ba_2pi_%s.json' % key; json.dump(val, open(fn + '.tmp', 'w'), indent=1, default=float); os.replace(fn + '.tmp', fn)


def scan(sig):
    P = strip_opt(sig); s = np.sqrt(sig); out = []
    for m in (0.0, 0.25, 0.5, 1.0, np.pi / 2, 2.0, np.pi, 4.0, 3 * np.pi / 2, 5.5, 2 * np.pi - 0.25, 2 * np.pi):
        c = np.sqrt(2) * m / s; bg = [(1.0, c, P['lam'] * P['M'])]; win = [(P['c'], c, P['lam'], 1)]
        o = gap(sig, bg, win, P['k0'], P['Lv_sig']); mo = moments(sig, bg, P['Lv_sig'])
        out.append({'m': m, 'g': float(o['gap_over_sigma']), **mo}); print('scan sigma %g m %.3f g %.6g H1 %s' % (sig, m, o['gap_over_sigma'], mo['H1']), flush=True)
    save('scan_%g' % sig, {'strip_g': P['g'], 'rows': out})


def mix(sig):
    P = strip_opt(sig); s = np.sqrt(sig); rows = []
    Lb = P['lam'] * P['M']; c2 = W2 / s
    # a second, different strip (k0 x 1.3, same other parameters) for the mediant test
    for alpha, other in ((0.1, False), (0.24, False), (0.24, True)):
        k2 = 1.3 if other else 1.0
        bg = [(1 - alpha, 0.0, Lb), (alpha, c2, Lb)]
        if not other:
            win = [(P['c'], 0.0, P['lam'], 1), (P['c'], c2, P['lam'], 1)]; o = gap(sig, bg, win, P['k0'], P['Lv_sig']); g2 = P['g']
        else:   # windows with wavenumbers k0 and 2 k0 (base k0): the 2 pi component is the strip with doubled wavenumber
            win = [(P['c'], 0.0, P['lam'], 1), (P['c'], c2, P['lam'], 2)]; o = gap(sig, bg, win, P['k0'], P['Lv_sig'])
            o2 = gap(sig, [(1.0, 0.0, Lb)], [(P['c'], 0.0, P['lam'], 2)], P['k0'], P['Lv_sig']); g2 = float(o2['gap_over_sigma'])
        # mediant from the components: Num, Den of each (J's and N are mass-weighted sums)
        mo = moments(sig, bg, P['Lv_sig'])
        r = {'alpha': alpha, 'second_component': '2 k0 strip' if other else 'same strip', 'g': float(o['gap_over_sigma']), 'g_component_0': P['g'], 'g_component_2pi': g2, **mo}
        rows.append(r); print('mix sigma %g alpha %.2f (%s): g %.8g  components %.8g, %.8g  H1 %s (E x_z^2 = %.3f)' % (sig, alpha, r['second_component'], r['g'], P['g'], g2, mo['H1'], mo['E_xz2']), flush=True)
    save('mix_%g' % sig, rows)


def opt(sig, maxfev=150):
    P = strip_opt(sig); s = np.sqrt(sig); log = []

    def state(x):
        k0, lam, Lw, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4])); Lv = eta * lam * s
        cb = np.sqrt(max(2 * np.pi ** 2 - Lw ** 2 - Lv ** 2, 0.0)) * np.tanh(x[5])
        return k0, lam, Lw, Lv / sig, c, cb

    def obj(x, shift=True):
        try:
            k0, lam, Lw, Lv_sig, c, cb = state(x)
            if not (0.3 < k0 < 12 and 0.1 < lam < 1e3 and lam * s < Lw < np.sqrt(2) * np.pi and Lv_sig > 30 / k0): return 1e300
            off = W2 if shift else 0.0
            bg = [(1.0, (cb - off) / s, Lw / s)]; win = [(c, (W2 - off) / s, lam, 1)]
            o = gap(sig, bg, win, k0, Lv_sig); g = float(o['gap_over_sigma'])
            if not np.isfinite(g) or o['y_over_sigma'] > 0.5: return 1e300
            log.append((g, list(map(float, x)))); return g
        except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError):
            return 1e300

    res = []
    for Lw0 in (0.3, 1.5, 3.5):
        x0 = np.array([np.log(P['k0']), np.log(P['lam']), np.log(Lw0), np.log(P['Lv_sig'] * sig / (P['lam'] * s)), np.log(P['c'] / (1 - P['c'])), 3.0])
        t0 = time.time(); r = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-6})
        k0, lam, Lw, Lv_sig, c, cb = state(r.x)
        rec = {'start_Lw': Lw0, 'g_min': float(r.fun), 'k0': k0, 'lam': lam, 'Lw': Lw, 'Lv_sig': Lv_sig, 'c': c, 'c_b': cb, 'x': list(map(float, r.x)),
               'nfev': r.nfev, 'seconds': time.time() - t0, **moments(sig, [(1.0, cb / s, Lw / s)], Lv_sig)}
        rec['window_weight'] = float(np.exp(-(W2 - cb) ** 2 / (2 * (Lw ** 2 + (lam * s) ** 2))))
        res.append(rec); print('opt sigma %g start Lw %.1f: g_min %.6g (strip %.6g)  Lw %.3g c_b %.3g k0 %.3g  window weight %.2e  %.0fs' % (
            sig, Lw0, r.fun, P['g'], Lw, cb, k0, rec['window_weight'], rec['seconds']), flush=True)
        save('opt_%g' % sig, {'strip_g': P['g'], 'runs': res})
    best = min(res, key=lambda q: q['g_min'])
    b = branches(sig, [(1.0, (best['c_b'] - W2) / s, best['Lw'] / s)], [(best['c'], 0.0, best['lam'], 1)], best['k0'], best['Lv_sig'])
    out = {'strip_g': P['g'], 'runs': res, 'best_branches': b}
    if sig >= 1e-6:     # the symmetry in the evaluator: direct frame (window at w = 2 sqrt2 pi)
        out['direct_frame_g'] = obj(np.array(best['x']), shift=False); out['shifted_frame_g'] = obj(np.array(best['x']), shift=True)
    save('opt_%g' % sig, out); print('opt sigma %g best g %.6g  branches %s' % (sig, best['g_min'], b), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore')
    part, sig = sys.argv[1], float(sys.argv[2])
    {'scan': scan, 'mix': mix, 'opt': opt}[part](sig)
