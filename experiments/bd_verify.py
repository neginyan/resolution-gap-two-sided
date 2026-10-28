"""Stage BD: verification of the comb-window state that comes below the strip optimum at sigma = 1e-6 (and later 1e-12, 1e-24).

Evaluators:
 E1  paper E's w_full.full_sums (period averages at v0 nodes, trapezoid in v0), default and refined settings;
 E2  at_tools.sums_envelope (point_fields2) with the v0 nodes SNAPPED to integer multiples of the ripple period — the period-average oscillation
     is then sampled at one phase and cannot alias (the known O(period/Lv) error of E1 at sigma = 1e-6 disappears), refined settings;
 the strip optimum is evaluated with the same E1/E2 settings for a like-for-like comparison.
Positivity: the density 1 + e^{-w^2/2 ell^2} c (C(v) - 1) of the truncated comb is checked on a fine grid.
usage: python bd_verify.py SIGMA [FILE] -> results/bd_verify_<sigma>.json.  Python 3.11."""
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
from w_full import full_sums
import at_tools
from at_tools import analyse, _acc
from v3_tools import _pre
from u_cgm import trap_weights
from ba_tools import strip_opt


def e1(sig, bg, win, kb, Lv_sig, nper, coarse, nv, n0):
    mmax = max(w[3] for w in win); T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    return float(full_sums(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv * mmax, n0=n0, ext=6.5)['gap_over_sigma'])


def e2(sig, bg, win, kb, Lv_sig, nper, coarse, nv, n0, ext=6.5):
    mmax = max(w[3] for w in win); T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    pre = _pre(T, sig); per = 2 * np.pi * sig / kb; Lv = np.sqrt((Lv_sig * sig) ** 2 + sig ** 2); ww = trap_weights(wn)
    nmax = int(np.ceil(ext * Lv / per)); step = max(1, int(round(2 * nmax / (n0 - 1))))
    v0s = per * np.arange(-nmax, nmax + 1, step); w0 = trap_weights(v0s); NV = nv * mmax
    acc = {k: 0.0 for k in at_tools.KT}; m = 0.0; npts = max(256, int(2.5e6 // max(1, len(pre))))
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(NV) + 0.5) * per / NV - per / 2
        V, W = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = cw * np.outer(np.full(NV, 1.0 / NV), ww).ravel()
        for i0 in range(0, len(X), npts): m += _acc(pre, sig, X[i0:i0 + npts], dA[i0:i0 + npts], acc)
    return analyse({k: v / m for k, v in acc.items()}, sig), len(v0s)


def positivity(win, kb):
    v = np.linspace(-np.pi / kb, np.pi / kb, 20001); dens = 1 + sum(c * np.cos(m * kb * v) for c, _, _, m in win)   # w = 0 (window centre: the worst case)
    return float(dens.min())


if __name__ == '__main__':
    np.seterr(all='ignore'); sig = float(sys.argv[1]); fn_in = sys.argv[2] if len(sys.argv) > 2 else 'results/bd_opt_comb_%g.json' % sig
    D = json.load(open(fn_in)); win = [tuple(w[:3]) + (int(w[3]),) for w in D['windows']]; kb = D['kb']; x = np.exp(D['x'][:4])
    s = np.sqrt(sig); lam, M, eta = x[1], x[2], x[3]; bg = [(1.0, 0.0, lam * M)]; Lv_sig = eta * lam / s
    P = strip_opt(sig); bgS = [(1.0, 0.0, P['lam'] * P['M'])]; winS = [(P['c'], 0.0, P['lam'], 1)]
    out = {'sigma': sig, 'source': fn_in, 'positivity_min_density': positivity(win, kb), 'comb': {}, 'strip': {}}
    print('positivity: min density at the window centre = %.3e' % out['positivity_min_density'], flush=True)
    for lab, st in (('E1 default', (25, 200, 32, 15)), ('E1 refined', (50, 400, 64, 41))):
        t0 = time.time(); gc = e1(sig, bg, win, kb, Lv_sig, *st); gs = e1(sig, bgS, winS, P['k0'], P['Lv_sig'], *st)
        out['comb'][lab] = gc; out['strip'][lab] = gs
        print('%-11s comb %.6f  strip %.6f  ratio %.4f  %.0fs' % (lab, gc, gs, gc / gs, time.time() - t0), flush=True)
        json.dump(out, open('results/bd_verify_%g.json' % sig, 'w'), indent=1, default=float)
    for lab, st in (('E2 snapped', (40, 300, 48, 41)), ('E2 snapped refined', (60, 500, 64, 81))):
        t0 = time.time(); rc, n0c = e2(sig, bg, win, kb, Lv_sig, *st); rs, n0s = e2(sig, bgS, winS, P['k0'], P['Lv_sig'], *st)
        out['comb'][lab] = rc['g']; out['strip'][lab] = rs['g']
        out['comb'][lab + ' details'] = {k: float(rc[k]) for k in ('sig2Jvv', 'sig2Jww', 'eps', 'Phi', 'delta_c', 'k_eff', 'bound_AT2', 'bound_ThmW')}
        out['strip'][lab + ' details'] = {k: float(rs[k]) for k in ('sig2Jvv', 'sig2Jww', 'eps', 'Phi', 'delta_c', 'k_eff', 'bound_AT2', 'bound_ThmW')}
        print('%-18s comb %.6f  strip %.6f  ratio %.4f  (v0 nodes %d, %d)  %.0fs' % (lab, rc['g'], rs['g'], rc['g'] / rs['g'], n0c, n0s, time.time() - t0), flush=True)
        json.dump(out, open('results/bd_verify_%g.json' % sig, 'w'), indent=1, default=float)
