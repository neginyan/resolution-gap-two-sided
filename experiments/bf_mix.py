"""Stage BF §2: where does the comb go below the mixtures (0.884)?  True evaluator (paper E's w_full, fast settings in the search, full settings at the end).

Start: the leading-order comb minimiser (as_strip_lo.g_lo_log with a_1 <= 2, the stage-BD/BE formula) at the same sigma, mapped to the comb window
(a_1 = 2 c e^{-(k st)^2/2}, start st = 0.5/k0).  Search: bd_case3.opt (Nelder-Mead over k, lam, M, eta, logit c, log st; 100 evaluations).
usage: python bf_mix.py SIGMA -> results/bd_opt_comb_<sigma>.json (+ 'start' = the LO comb parameters).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import sys, json
import numpy as np
from scipy.optimize import minimize
import bd_case3
from bd_comb_lo import g_x


def lo_comb(sig):
    g, x = lo_min(sig, 2.0); k0, lam, M, eta = np.exp(x[:4]); a1 = 2.0 / (1 + np.exp(-x[4]))
    return {'g_LO': g, 'k0': float(k0), 'lam': float(lam), 'M': float(M), 'eta': float(eta), 'a1': float(a1)}

def lo_min(sig, cmax):
    """robust multistart of the stage-AS leading-order formula (a_1 <= cmax); bd_comb_lo.gmin's start set misses the feasible region at sigma >~ 1e-4."""
    lsig = np.log(sig); best = None; kg = np.sqrt(-lsig / 4)
    for k0 in (1.0, 1.25, 1.5, 2.0, kg):
        for lM in (0.5, 1.5, 3.0, np.log(10) + k0 * k0):
            for leta in (-3.0, -1.5, -0.5):
                x0 = np.array([np.log(k0), np.log(1.5), lM, leta, 2.0])
                r = minimize(g_x, x0, args=(lsig, cmax), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
                for _ in range(2): r = minimize(g_x, r.x, args=(lsig, cmax), method='Nelder-Mead', options={'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
                if best is None or r.fun < best.fun: best = r
    return float(best.fun), best.x



if __name__ == '__main__':
    np.seterr(all='ignore'); sig = float(sys.argv[1]); L = lo_comb(sig); s = np.sqrt(sig)
    cwin = min(0.995, L['a1'] / 2 * np.exp(0.125))
    P = {'k0': L['k0'], 'lam': L['lam'], 'M': L['M'], 'Lv_sig': L['eta'] * L['lam'] / s, 'c': cwin, 'g': L['g_LO']}
    print('LO comb start at sigma %g: %s' % (sig, L), flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == 'lo': sys.exit()
    bd_case3.strip_opt = lambda _s: P
    # disk cache of evaluator calls (the container can restart; Nelder-Mead is deterministic, so a rerun replays the cached calls instantly)
    import hashlib, pickle, os
    cdir = 'results/bf_mix_cache_%g' % sig; os.makedirs(cdir, exist_ok=True); _fs = bd_case3.full_sums
    def _key(a):
        h = hashlib.sha1()
        for v in a:
            h.update(pickle.dumps(v if not isinstance(v, np.ndarray) else v.tobytes()))
        return h.hexdigest()
    def cached(*a, **k):
        f = os.path.join(cdir, _key(list(a) + sorted(k.items())) + '.pkl')
        if os.path.exists(f): return pickle.load(open(f, 'rb'))
        o = _fs(*a, **k); pickle.dump(o, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f); return o
    bd_case3.full_sums = cached
    bd_case3.opt(sig, 'comb')
    fn = 'results/bd_opt_comb_%g.json' % sig; d = json.load(open(fn)); d['start'] = L; d['strip_g'] = None; d['note'] = 'start = leading-order comb minimiser (bf_mix.py)'
    json.dump(d, open(fn, 'w'), indent=1, default=float)

