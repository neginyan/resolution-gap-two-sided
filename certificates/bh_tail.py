"""Stage BH §2(a): the analytic tail of Theorem AY1-P beyond the certified cells (l = ln(1/sigma) > l_end = 1e5 ln 10).

Lemma BH2 (tail; proof).  For l >= l_end put G = 1.2/sqrt(l) (< G_end), sigma = e^{-l} (< sigma_end); start the bootstrap from F_0/G = 1.02 as in the cells.
(i) The chain depends on F only through U0 = F^2/(4G^2) - 1/4 and x_low; with F/G fixed, every quantity is monotone in (sigma, G) (Lemma BH1), and the
    suprema over x do not depend on the right end of the x-range once x_low >= 8 (all functions decrease beyond x = 8).  So every step of the bootstrap at l
    is at least as good as at l_end: all conditions hold, F_k(l) >= F_k(l_end), x_low(l) >= x_low(l_end) (l/l_end) (U0 grows like 1/G^2).
(ii) K(l) <= K_end + 9 ln(G/G_end) = K_end - (9/2) ln(l/l_end):  Gam2 = 2 eta G theta(1 + beta) + sigma <= (G/G_end) Gam2_end and Gam3_P <= (G/G_end) Gam3P_end
    (theta, beta, zeta_2 decrease; sigma/G decreases), U_P(l) >= (G_end/G)^2 U_P,end (U0 = F^2/(4G^2) - 1/4 with F >= F_end), phi1, omega do not decrease.
(iii) F_int(l) >= 1 - Fdef(l) and every term of Fdef carries a factor sigma, sqrt(sigma X) or L2max <= C x_low e^{-x_low}, so Fdef(l) -> 0.
Hence g sqrt(l)/(1 - |eps|) >= (1 - Fdef(l)) sqrt(l/(X(l) + c))/2 with 4X = l + 5 ln X + K_end - (9/2) ln(l/l_end), X(l) = l/4 + O(ln l), and
    liminf_{sigma -> 0} g_min sqrt(ln(1/sigma)) >= 1 - |eps|   (the G-branch gives G sqrt(l) = 1.2).
Numerics here: the explicit tail bound T(l) on a logarithmic grid up to l = 1e300 (it increases to 1), and the cell certificate re-run directly at
l = 1e6, 1e7, 1e8 (all conditions hold, constants better than at l_end) as a check of (i)-(ii).  -> results/bh_tail.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
from scipy.optimize import brentq
import bh_cert_cells as H

if __name__ == '__main__':
    R = {}
    for f in sorted(glob.glob('results/bh_cert_cells_part*.json')): R.update(json.load(open(f)))
    last = R[str(max(int(k) for k in R))]; l_end = last['lb']; K_end = last['K']; Fdef_end = last['Fdef']; meta = json.load(open('results/bh_cert_cells_meta.json'))
    c = meta['S1_P'] + 0.25
    def T(l):
        K = K_end - 4.5 * np.log(l / l_end); X = brentq(lambda X: 4 * X - l - 5 * np.log(X) - K, 1.0, l + 1e3 + abs(K), maxiter=2000)
        return (1 - Fdef_end) * np.sqrt(l / (X + c)) / 2, X - l / 4
    grid = np.exp(np.linspace(np.log(l_end), np.log(1e300), 400)); vals = [T(l) for l in grid]
    Tv = np.array([v[0] for v in vals])
    out = {'l_end': l_end, 'K_end': K_end, 'Fdef_end': Fdef_end, 'T_at_l_end': float(Tv[0]), 'T_min': float(Tv.min()), 'T_increasing': bool(np.all(np.diff(Tv) >= -1e-15)),
           'T_at_1e300': float(Tv[-1]), 'samples': {('%.3g' % l): float(t) for l, t in zip(grid[::50], Tv[::50])}, 'direct': {}}
    print('tail bound: T(l_end) = %.6f, increasing %s, T(1e300) = %.15f' % (Tv[0], out['T_increasing'], Tv[-1]), flush=True)
    for l in (1e6, 1e7, 1e8):
        r = H.cell(l, l * 1.0000001, meta['S1_P'], meta['S_i_at_1'])
        out['direct'][('%g' % l)] = {k: r.get(k) for k in ('ok', 'L_cell', 'K', 'Fdef', 'n_steps', 'x_lo')}
        out['direct'][('%g' % l)]['K_bound_from_tail'] = K_end - 4.5 * np.log(l / l_end)
        print('direct cell at l = %g: ok %s  L %.9f  K %.3f <= tail bound %.3f' % (l, r['ok'], r['L_cell'], r['K'], K_end - 4.5 * np.log(l / l_end)), flush=True)
    json.dump(out, open('results/bh_tail.json', 'w'), indent=1, default=float)
