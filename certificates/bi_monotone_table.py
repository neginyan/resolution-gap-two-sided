"""Stage BI §1.1: the monotonicity table of Lemma 6 (BH1), checked on grids (numerical confirmation; the proof is the sign column of the table in App. B).

Variables: hs = sigma-hat (every occurrence of sigma as a small factor), G (threshold), eta = 1/(1 - eps_bar), l = ln(1/sigma) (range of x and the root X_P),
F (lower bound of F_c).  x_lo = max(1, U0/S_i(1), U0 - S1_P) with U0 = F^2/(4G^2) - 1/4 (increasing in F, decreasing in G).
Declared direction of each constant: +1 = non-decreasing (bad when it grows), -1 = non-increasing, 0 = independent.  Each variable is moved over a grid
with the others fixed at a base point; the chain (bf_cert_p.chain) is evaluated and the observed direction compared with the table (relative tolerance
1e-12 for 'non-decreasing').  -> results/bi_monotone.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
import bf_cert_p as B

# The suprema over x are exact sups over [x_lo, x_e]; x_e grows with l.  The certificate bounds each sup on a grid re-spaced over the current range,
# which wiggles at the 1e-4 level when x_e moves (each value is still an upper bound of the true sup).  To test the table on the mathematical
# quantities, this check uses one global grid (spacing 0.05 from x = 1) whose first cell starts at x_lo, so that a longer range only adds cells and a larger x_lo only removes or shrinks them.
STEP = 0.05
def _anchored(f, x0, x1, n=None):
    # global grid 1, 1 + STEP, ...; the first cell starts at x0, so a larger x0 or a smaller x1 only removes cells or shrinks the first one
    k0 = int(np.floor((x0 - 1.0) / STEP)); k1 = int(np.ceil((x1 - 1.0) / STEP))
    xs = np.concatenate([[x0], 1.0 + STEP * np.arange(k0 + 1, k1 + 1)]); xs = xs[xs >= x0]
    if len(xs) < 2: xs = np.array([x0, x1])
    return max(float(max(f(a, b) for a, b in zip(xs[:-1], xs[1:]))), 1e-300)
B.cell_sup = _anchored

S1 = 1.4064694014551398; SI1 = 2.279132089858428
# constant: (hs, G, eta, l, F)
TABLE = {
    'theta_G': (+1, +1, +1, 0, 0), 'U0': (0, -1, 0, 0, +1), 'x_low': (0, -1, 0, 0, +1), 'U_P': (0, -1, 0, 0, +1),
    'phi1': (-1, -1, -1, -1, +1), 'beta': (+1, +1, +1, +1, -1), 'nu': (+1, +1, +1, +1, -1), 'V_max': (+1, +1, +1, +1, -1),
    'omega': (-1, -1, -1, -1, +1), 'Gam2': (+1, +1, +1, +1, -1), 'Gam3P': (+1, +1, +1, +1, -1), 'K_P': (+1, +1, +1, +1, -1),
    'X_P': (+1, +1, +1, +1, -1), 'L2max': (+1, +1, +1, +1, -1), 'sigma_over_s_max': (+1, +1, +1, +1, -1),
    'delta_H': (+1, +1, +1, +1, -1), 'r_C': (+1, +1, +1, +1, -1), 'r_w': (+1, +1, +1, +1, -1), 'F_def': (+1, +1, +1, +1, -1),
}


def consts(hs, G, eta, l, F):
    U0 = F * F / (4 * G * G) - 0.25; x_lo = max(1.0, U0 / SI1, U0 - S1)
    c = B.chain(hs, l, G, eta, F, x_lo, S1)
    if not c['ok']: return None
    th = 1 / (1 - eta * G * hs)
    return {'theta_G': th, 'U0': U0, 'x_low': x_lo, 'U_P': c['U_P'], 'phi1': c['phi1'], 'beta': c['beta'], 'nu': c['nu'], 'V_max': c['V_max'],
            'omega': c['omega'], 'Gam2': c['Gam2'], 'Gam3P': c['Gam3P'], 'K_P': c['K'], 'X_P': c['X'], 'L2max': c['L2max'], 'sigma_over_s_max': c['sigma_over_s_max'],
            'delta_H': c['dH'], 'r_C': c['rC'], 'r_w': c['rw'], 'F_def': eta * G * hs * th + c['dH'] + c['rC'] + c['rw']}


def direction(vals):
    v = np.array(vals, float); d = np.diff(v); tol = 1e-12 * np.maximum(np.abs(v[:-1]), np.abs(v[1:])) + 1e-300
    up = bool(np.all(d >= -tol)); down = bool(np.all(d <= tol))
    return 0 if (up and down) else (+1 if up else (-1 if down else 2))       # 2 = not monotone


if __name__ == '__main__':
    bases = [dict(hs=np.exp(-60.0), G=0.134, eta=1 / 0.9, l=60.0, F=0.9), dict(hs=np.exp(-60.0), G=0.13, eta=1 / 0.95, l=60.0, F=0.5),
             dict(hs=np.exp(-300.0), G=0.069, eta=1 / 0.9, l=300.0, F=0.95)]
    names = ['hs', 'G', 'eta', 'l', 'F']; out = {'table': TABLE, 'runs': []}; bad = []
    for bi, base in enumerate(bases):
        grids = {'hs': base['hs'] * np.logspace(-6, 0, 9), 'G': np.linspace(base['G'] * 0.8, base['G'], 9), 'eta': np.linspace(1.0, 1 / 0.9, 9),
                 'l': np.linspace(base['l'], base['l'] * 1.3, 9), 'F': np.linspace(base['F'] * 0.9, base['F'], 9)}
        for j, var in enumerate(names):
            series = []
            for v in grids[var]:
                p = dict(base); p[var] = v; c = consts(**p)
                if c is None: break
                series.append(c)
            if len(series) < len(grids[var]): out['runs'].append({'base': bi, 'var': var, 'skipped': 'chain fails on part of the grid'}); continue
            for name, dirs in TABLE.items():
                obs = direction([s[name] for s in series]); dec = dirs[j]
                ok = (obs == 0) or (obs == dec)          # independent in practice is consistent with any declared direction
                out['runs'].append({'base': bi, 'var': var, 'const': name, 'declared': dec, 'observed': obs, 'ok': ok})
                if not ok: bad.append((bi, var, name, dec, obs))
    out['n_checks'] = sum(1 for r in out['runs'] if 'const' in r); out['all_ok'] = not bad; out['bad'] = bad
    out['n_nonmonotone'] = sum(1 for r in out['runs'] if r.get('observed') == 2)
    print('%d (constant, variable, base) checks, all consistent with the table: %s; non-monotone observations: %d; skipped runs: %d' % (
        out['n_checks'], out['all_ok'], out['n_nonmonotone'], sum(1 for r in out['runs'] if 'skipped' in r)))
    for b in bad: print('MISMATCH', b)
    if not os.environ.get('RGTS_NO_WRITE'): json.dump(out, open('results/bi_monotone.json', 'w'), indent=1, default=float)
