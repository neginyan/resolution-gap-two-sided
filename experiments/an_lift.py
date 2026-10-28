"""Stage AN: the two-dimensional lift of the hidden multipoles (two w-lines + n + 1 dotted rows in two phases).

Rows at v = x_i with linear densities w_i, phase 0 where f_i = +1 and phase a/2 where f_i = -1.  Poisson summation gives exactly
p = P0 (1 + 2 g O(w) + 2 h E(w)) with g = sum_i f_i P_i / P0 (signed) and h = sum_i P_i / P0 (unsigned), so Lemma AN4 applies verbatim:
two-sided bounds on tR and ln(1/tJ) from the v-integrals I_g, I_g', I_h, I_h' (computed exactly by an_signed1d.quantities) and q-sums.
The spacing a is chosen so that the even (unsigned) harmonics carry < 1e-6 of J.  Lambda (paper E) in leading order is Lambda_v.
-> results/an_lift.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from an_signed1d import quantities
from an_theta import qsums
from an_multipole import realise


def lemma_an4(Ig, Ih, Igp, Ihp, lMs, a):
    Q = qsums(a); eb = 2 * (Q['sig0'] + Q['e0']); vs = 2 * (Q['sig1'] + Q['e1']) / (1 - eb)
    Jlin = 4 * (Q['mu1'] * Ig + Q['me1'] * Ih)
    Nw2 = 4 * (Q['mu2'] * Ig + Q['me2'] * Ih); Nv2 = 4 * (Q['mu1'] * Igp + Q['me1'] * Ihp); Ev2 = 4 * (Q['mu0'] * Igp + Q['me0'] * Ihp)
    c = np.sqrt(1 + eb) / (1 - eb); cl = np.sqrt(1 - eb) / (1 + eb)
    up = ((c * np.sqrt(Nw2) + vs * np.sqrt(Jlin / (1 - eb))) ** 2 + (c * (np.sqrt(Nv2) + vs * np.sqrt(Ev2))) ** 2) * (1 + eb) / Jlin
    lo = (max(0.0, cl * np.sqrt(Nw2) - vs * np.sqrt(Jlin / (1 - eb))) ** 2 + max(0.0, cl * np.sqrt(Nv2) - c * vs * np.sqrt(Ev2)) ** 2) * (1 - eb) / Jlin
    ln_lo = lMs - np.log(Jlin / (1 - eb)); ln_up = lMs - np.log(Jlin / (1 + eb))
    return {'tR': [float(lo), float(up)], 'ln1_tJ': [float(ln_lo), float(ln_up)], 'even_weight': float(4 * Q['me1'] * Ih / Jlin), 'q': Q['q']}


if __name__ == '__main__':
    M = json.load(open('results/an_multipole.json')); out = {}
    for n in ('1', '2', '3', '4'):
        L = M['leading'][n]; e = M['exact'][n]['80']; D = 80.0
        xs, lw, f = realise(D, L['alpha'], np.array(L['p']), e['d'], e['z']); lw = lw - lw.max()
        og = quantities(xs, lw, f); oh = quantities(xs, lw, np.abs(f))
        s0 = og['lI']; Ig, Igp, Ih, Ihp = 1.0, np.exp(og['lD'] - s0), np.exp(oh['lI'] - s0), np.exp(oh['lD'] - s0); lMs = og['lM'] - s0
        for a in (2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
            b = lemma_an4(Ig, Ih, Igp, Ihp, lMs, a)
            if b['even_weight'] < 1e-6: break
        r = {'n': int(n), 'D': D, 'a': a, 'bracket': b, 'Rq_v': og['Rq'], 'lnMI_v': og['lnMI'], 'Lam_v': og['Lam_v'],
             'tR_over_ln_min': b['tR'][0] / b['ln1_tJ'][1], 'Lam_over_ln': og['Lam_v'] / b['ln1_tJ'][1],
             'B_n_leading': L['B_n']}
        out[n] = r
        print('n %s, D 80, a = %g (q %.1e, even weight %.1e): 2-d tR in [%.1f, %.1f], ln(1/tJ) in [%.1f, %.1f] -> tR/ln(1/tJ) >= %.3f (leading B_n %.3f); Lambda/ln(1/tJ) = %.3f' % (
            n, a, b['q'], b['even_weight'], *b['tR'], *b['ln1_tJ'], r['tR_over_ln_min'], L['B_n'], r['Lam_over_ln']), flush=True)
    # higher orders: wider spacing d keeps the finite-difference weights inside float64; scan d, z
    for n in ('6', '8'):
        L = M['leading'][n]; best = None
        for D in (80.0, 120.0):
            for d in (2.0, 3.0, 4.0):
                for z in (8.0, 16.0):
                    xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max()
                    og = quantities(xs, lw, f)
                    if not np.isfinite(og['Lam_v']) or og['lnMI'] < 10: continue
                    oh = quantities(xs, lw, np.abs(f)); s0 = og['lI']
                    Ig, Igp, Ih, Ihp = 1.0, np.exp(og['lD'] - s0), np.exp(oh['lI'] - s0), np.exp(oh['lD'] - s0); lMs = og['lM'] - s0
                    for a in (1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2):
                        b = lemma_an4(Ig, Ih, Igp, Ihp, lMs, a)
                        if b['even_weight'] < 1e-6: break
                    r = {'n': int(n), 'D': D, 'd': d, 'z': z, 'a': a, 'bracket': b, 'Rq_v': og['Rq'], 'lnMI_v': og['lnMI'], 'Lam_v': og['Lam_v'],
                         'tR_over_ln_min': b['tR'][0] / b['ln1_tJ'][1], 'Lam_over_ln': og['Lam_v'] / b['ln1_tJ'][1], 'B_n_leading': L['B_n']}
                    if best is None or r['Lam_over_ln'] > best['Lam_over_ln']: best = r
        out[n] = best
        print('n %s, D %g, d %g, a = %g (even weight %.1e): tR/ln(1/tJ) >= %.3f (leading B_n %.3f); Lambda/ln(1/tJ) = %.3f' % (
            n, best['D'], best['d'], best['a'], best['bracket']['even_weight'], best['tR_over_ln_min'], L['B_n'], best['Lam_over_ln']), flush=True)
    json.dump(out, open('results/an_lift.json', 'w'), indent=1)
