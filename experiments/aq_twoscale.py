"""Stage AQ, hybrid H-B: the confined multipole family of stage AP with a GENERAL w-wavenumber k of the row modulation (including the
strip optima's k0), true gap by the two-scale formula with general k:
  sigma^2 J_ww = eps + (k^2 e^{-k^2}/2) e^{-lnMI},  g = (2 sigma^2 J_ww + N)/(sigma (Jvv + sigma^2 J_ww)),  N = (1/2)(1 - e^{-Lw^2/4})(Lw^2/(Lw^2 + sigma^2) - 1 + Jvv),
minimised over the stage-AP library x line length Lw x k; cross-checked against paper E's evaluator v3_tools at sigma = 1e-4 for k = 0.5,
k0(1e-4) = 1.078 and 2.  -> results/aq_twoscale.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
from ap_direct import v3_gap, two_scale
from an_multipole import realise

SIGMAS = [1e-4, 1e-6, 1e-12, 1e-24, 1e-48, 1e-96]
KS = [0.3, 0.5, 1.0, 1.078, 1.156, 1.847, 3.0, 4.0]          # includes the strip optima's k0 at 1e-4, 1e-6, 1e-12, 1e-24


def g_two_scale(Jvv, lnMI, sig, Lw, k):
    eps = sig ** 2 / (Lw ** 2 + sig ** 2); tJ = (k * k * np.exp(-k * k) / 2) * np.exp(-lnMI); Jww = eps + tJ
    N = -0.5 * np.expm1(-Lw ** 2 / 4) * (Lw ** 2 / (Lw ** 2 + sig ** 2) - 1 + Jvv)
    return (2 * Jww + N) / (sig * (Jvv + Jww))


if __name__ == '__main__':
    lib = []
    for fn in sorted(glob.glob('results/ap_library_*.json')): lib += json.load(open(fn))
    out = {'min': {}, 'v3_check': {}}
    for sig in SIGMAS:
        best = None
        for k in KS:
            for Lr in np.exp(np.linspace(np.log(0.25), np.log(16.0), 33)):
                Lw = Lr * np.sqrt(sig)
                for r in lib:
                    g = g_two_scale(r['Jvv'], r['lnMI'], sig, Lw, k)
                    if best is None or g < best['g']: best = {'g': g, 'k': k, 'Lw_over_sqrt_sigma': Lr, 'n': r['n'], 'D': r['D'], 'Jvv': r['Jvv'], 'bound_1_over_sqrt_Jvv': 1 / np.sqrt(r['Jvv'])}
        out['min']['%g' % sig] = best
        print('sigma %-6g: min over library x Lw x k of the true gap = %.5f (k %.3g, n %d, D %g; 1/sqrt(Jvv) = %.5f)' % (sig, best['g'], best['k'], best['n'], best['D'], best['bound_1_over_sqrt_Jvv']), flush=True)
    M = json.load(open('results/an_multipole.json')); L = M['leading']['2']; sig = 1e-4; Lw = 2 * np.sqrt(sig)
    xs, lw, f = realise(2.0, L['alpha'], np.array(L['p']), 0.5, 1.0); lw = lw - lw.max()
    for k in (0.5, 1.078, 2.0):
        ts = two_scale(xs, lw, f, sig ** 2 / (Lw ** 2 + sig ** 2), k)
        N = -0.5 * np.expm1(-Lw ** 2 / 4) * (Lw ** 2 / (Lw ** 2 + sig ** 2) - 1 + ts['Jvv']); g_ts = (2 * ts['Jww'] + N) / (sig * (ts['Jvv'] + ts['Jww']))
        gv = v3_gap(xs, lw, f, sig, Lw, k)
        out['v3_check']['%g' % k] = {'two_scale': g_ts, 'v3': gv['gap_over_sigma'], 'rel': abs(gv['gap_over_sigma'] / g_ts - 1), 'mp_share_Jww': ts['tJmp'] / ts['Jww']}
        print('  v3 check k = %g: two-scale %.5f  v3 %.5f (rel %.1e; multipole share of J_ww %.2f)' % (k, g_ts, gv['gap_over_sigma'], out['v3_check']['%g' % k]['rel'], ts['tJmp'] / ts['Jww']), flush=True)
        json.dump(out, open('results/aq_twoscale.json', 'w'), indent=1, default=float)
    json.dump(out, open('results/aq_twoscale.json', 'w'), indent=1, default=float)
