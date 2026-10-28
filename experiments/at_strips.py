"""Stage AT: Theorem AT2 on the strip optima (true envelope) and on the class-C states of stage AR.  -> results/at_strips.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from aq_hybrid import strip_params
from at_tools import sums_envelope, sums_grid, analyse

if __name__ == '__main__':
    fn = 'results/at_strips.json'; R = json.load(open(fn)) if os.path.exists(fn) else {}
    for s in (1e-6, 1e-12, 1e-24):
        key = 'strip_%g' % s
        if key in R: continue
        P = strip_params(s); bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'], 0.0, P['lam'], 1)]
        T = terms_of(bg, win, P['k0'], s, P['Lv_sig']); wn = w_nodes(bg, win, s, 50, 400); t0 = time.time()
        r = analyse(sums_envelope(T, s, P['k0'], np.sqrt((P['Lv_sig'] * s) ** 2 + s ** 2), wn), s); r.update({'sigma': s, 'k0': P['k0'], 'family': 'strip'})
        R[key] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
        print(key, ' '.join('%s %.5g' % (k, r[k]) for k in ('g', 'bound_ThmW', 'bound_AT2', 'C_w', 'k_eff', 'k0', 'heisenberg_ratio', 'delta_w')), '%.0f s' % (time.time() - t0), flush=True)
    from ar_check import class_C_terms
    from an_multipole import realise
    M = json.load(open('results/an_multipole.json'))
    cases = [('lines', None, 8.0, 0.0, 0.0, 1.0, 1e-4, 2.0), ('mp_n1_D2_k1', '1', 2.0, 0.5, 1.0, 1.0, 1e-4, 2.0), ('mp_n2_D3_k1', '2', 3.0, 0.5, 0.5, 1.0, 1e-4, 2.0),
             ('mp_n2_D2_k2', '2', 2.0, 0.5, 1.0, 2.0, 1e-4, 2.0), ('mp_n4_D4_k1', '4', 4.0, 0.5, 1.0, 1.0, 1e-4, 2.0)]
    for name, n, D, d, z, k, sig, Lr in cases:
        key = 'C_' + name
        if key in R: continue
        Lw = Lr * np.sqrt(sig)
        if n is None: xs, lw, f = np.array([-D, D]), np.zeros(2), np.zeros(2)
        else: L = M['leading'][n]; xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max()
        T = class_C_terms(xs, lw, f, sig, Lw, k); hv = min(0.0125, 0.1 / max(xs.max(), 1.0)) * sig
        vn = np.arange(xs.min() * sig - 8 * sig, xs.max() * sig + 8 * sig, hv)
        wstep = Lw / 40 if n is None else min(2 * np.pi / k * sig / 16, Lw / 40); wn = np.arange(-6 * Lw, 6 * Lw, wstep)
        r = analyse(sums_grid(T, sig, vn, wn), sig); r.update({'sigma': sig, 'family': 'class C'}); R[key] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
        print(key, ' '.join('%s %.5g' % (k_, r[k_]) for k_ in ('g', 'bound_ThmW', 'bound_AT2', 'C_w', 'k_eff', 'heisenberg_ratio')), flush=True)
