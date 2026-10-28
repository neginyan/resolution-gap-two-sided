"""Stage BE §1.4: the leading-order minimum of the periodic-window family in closed form.

Proposition BE1 (exact identity of the leading-order formula; derivation-level object).  Write the stage-AS leading-order gap with the window's
fundamental amplitude a (the strip has a = c <= 1, every positive profile a_1 <= 2, the comb a_1 -> 2):
  g_LO(sigma; k0, lam, M, eta, a) = [2 CR/(M^2 lam^2 W) + lam^2 (1/2 + eta^2 + r M^2)/8]/(1 + r),
  W = a^2 k0^2 e^{-k0^2}/(2 sqrt2 M),  CR = 1 + (sqrt2/8) a^2 e^{-k0^2} M,  r = sigma/(eta^2 lam^2 W).
With M' = a^2 M:  CR, M^2 lam^2 W and r M^2 become the a = 1 expressions at (M', sigma/a^8); only the denominator 1 + r = 1 + a^4 r' differs. Hence
  **g_LO(sigma; ., a) = g_LO(sigma/a^8; ., 1) (1 + r')/(1 + a^4 r')**,  and the family minimum is  G_a(sigma) = G_1(sigma/a^8)(1 + O(r)),  r ~ sigma/W tiny.
So the minimising amplitude is the largest one, a_1 = 2 (G_1 increases with sigma), and
  **g_min^comb(sigma) = g_min^strip(sigma/256) (1 + O(r))**,  ratio R(sigma) = G_1(sigma/256)/G_1(sigma) = 1 - 4 ln 2/ln(1/sigma) + O(ln ln/ln^2).
-> results/be_lo.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from as_strip_lo import g_lo_log
from bd_comb_lo import gmin

if __name__ == '__main__':
    rng = np.random.default_rng(3); worst = 0.0
    for _ in range(2000):                                  # the identity at random parameters
        ls = -rng.uniform(10, 2000); k0 = rng.uniform(0.5, 30); llam = rng.uniform(-2, 2); lM = rng.uniform(0, 400); leta = rng.uniform(-6, 0); a = rng.uniform(0.2, 2)
        g_a = g_lo_log(ls, k0, llam, lM, leta, a); g_1 = g_lo_log(ls - 8 * np.log(a), k0, llam, lM + 2 * np.log(a), leta, 1.0)
        lW1 = 2 * np.log(k0) - k0 * k0 - np.log(2 * np.sqrt(2)) - (lM + 2 * np.log(a)); r1 = np.exp(ls - 8 * np.log(a) - 2 * leta - 2 * llam - lW1)
        pred = g_1 * (1 + r1) / (1 + a ** 4 * r1)
        if np.isfinite(g_a) and np.isfinite(pred) and g_a < 1e300: worst = max(worst, abs(g_a / pred - 1))
    print('identity g_LO(sigma; a) = g_LO(sigma/a^8; M a^2, 1)(1 + r\')/(1 + a^4 r\'): max relative deviation over 2000 random parameter sets %.1e' % worst)
    out = {'identity_max_dev': worst, 'table': {}}
    for e in (6, 12, 24, 48, 96, 256, 1024, 4096, 16384):
        ls = -e * np.log(10); gs = gmin(ls, 1.0)[0]; gc = gmin(ls, 2.0)[0]; gsh = gmin(ls - 8 * np.log(2), 1.0)[0]; L = -ls
        out['table']['1e-%d' % e] = {'strip': gs, 'comb': gc, 'strip_at_sigma_over_256': gsh, 'ratio': gc / gs, 'ratio_pred_1st_order': 1 - 4 * np.log(2) / L,
                                     'comb_over_shifted': gc / gsh}
        print('sigma 1e-%-6d strip %.6f  comb %.6f  strip(sigma/256) %.6f  comb/shifted %.9f | ratio %.5f  1 - 4 ln2/ln %.5f' % (
            e, gs, gc, gsh, gc / gsh, gc / gs, 1 - 4 * np.log(2) / L), flush=True)
    json.dump(out, open('results/be_lo.json', 'w'), indent=1)
