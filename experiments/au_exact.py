"""Stage AU: exact checks (paper E point fields) of the leading-order branch formulas of au_branches.py, and of the identity
k_eff^2 = Lambda/4 against stage AP's independent direct 2-d code (ap_direct.direct2d, closed-form blurred components).
-> results/au_exact.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
from at_tools import sums_grid, analyse
from ar_check import class_C_terms
from an_multipole import realise
from au_branches import branches
from ap_direct import direct2d


def state(row, D_lines=8.0):
    M = json.load(open('results/an_multipole.json'))
    if row is None: return np.array([-D_lines, D_lines]), np.zeros(2), np.zeros(2)
    L = M['leading'][str(row['n'])]; xs, lw, f = realise(row['D'], L['alpha'], np.array(L['p']), row['d'], row['z']); return xs, lw - lw.max(), f


def exact(row, sig, Lw):
    xs, lw, f = state(row); T = class_C_terms(xs, lw, f, sig, Lw, 1.0); hv = min(0.0125, 0.1 / max(xs.max(), 1.0)) * sig
    vn = np.arange(xs.min() * sig - 8 * sig, xs.max() * sig + 8 * sig, hv)
    wstep = Lw / 40 if row is None else min(2 * np.pi * sig / 16, Lw / 40); wn = np.arange(-6 * Lw, 6 * Lw, wstep)
    E = sums_grid(T, sig, vn, wn); r = analyse(E, sig); s2 = sig * sig; den = r['sig2Jvv'] + r['sig2Jww']
    Aw = s2 * E['sww_w']; Nphi = s2 / 2 * E['phi_svv']
    r.update({'a_w': 2 * Aw / (sig * den), 'a_N': Nphi / (sig * den), 'Lambda': 4 * r['k_eff'] ** 2, 'Aw_over_A': Aw / r['sig2Jww']})
    return r


if __name__ == '__main__':
    lib = []
    for fn in sorted(glob.glob('results/ap_library_*.json')): lib += json.load(open(fn))
    pick = lambda n, D, d, z: [x for x in lib if x['n'] == n and x['D'] == D and x['d'] == d and x['z'] == z][0]
    out = {'branches': {}, 'identity': {}}
    sig = 1e-4
    for name, row, Lr in (('lines_Lw2', None, 2.0), ('n1_D3_Lw2', pick(1, 3.0, 1.0, 0.5), 2.0), ('n2_D3_Lw2', pick(2, 3.0, 1.0, 1.0), 2.0), ('n2_D3_Lw5', pick(2, 3.0, 1.0, 1.0), 5.0)):
        Lw = Lr * np.sqrt(sig); ex = exact(row, sig, Lw); lo = branches(row, sig, Lw)
        ex['AT2'] = ex['bound_AT2']
        out['branches'][name] = {'exact': {k: ex[k] for k in ('g', 'a_w', 'a_N', 'bound_AT2', 'C_w', 'Lambda', 'Aw_over_A', 'Phi', 'bound_ThmW', 'eps')},
                                 'leading_order': {k: lo[k] for k in ('g_true', 'a_w', 'a_N', 'AT2', 'Lambda', 'ThmW')}}
        print('%-10s exact: g %.5f (a) %.5f (a\') %.5f AT2 %.5f C_w %.5f Lambda %.4f | LO rel. err: %s' % (name, ex['g'], ex['a_w'], ex['a_N'], ex['bound_AT2'], ex['C_w'], ex['Lambda'],
              ' '.join('%s %.1e' % (k, abs(lo[k] / ex['g' if k == 'g_true' else k] - 1)) for k in ('g_true', 'a_w', 'a_N', 'AT2', 'Lambda'))), flush=True)
        out['branches'][name]['rel'] = {k: abs(lo[k] / ex['g' if k == 'g_true' else k] - 1) for k in ('g_true', 'a_w', 'a_N', 'AT2', 'Lambda')}
        json.dump(out, open('results/au_exact.json', 'w'), indent=1, default=float)
    # identity k_eff^2 = Lambda/4: at_tools (k_eff from point fields) vs ap_direct.direct2d (Lambda by its own closed-form code), t = 1 units
    M = json.load(open('results/an_multipole.json'))
    for n, D, d, z, ell in (('2', 2.0, 0.5, 0.5, 100.0), ('1', 3.0, 1.0, 0.5, 50.0)):
        L = M['leading'][n]; xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); lw = lw - lw.max()
        dr = direct2d(xs, lw, f, ell, per=24)
        sig = 1e-4; Lw = ell * sig; T = class_C_terms(xs, lw, f, sig, Lw, 1.0); hv = min(0.0125, 0.1 / max(xs.max(), 1.0)) * sig
        vn = np.arange(xs.min() * sig - 8 * sig, xs.max() * sig + 8 * sig, hv); wn = np.arange(-6 * Lw, 6 * Lw, min(2 * np.pi * sig / 16, Lw / 40))
        r = analyse(sums_grid(T, sig, vn, wn), sig)
        key = 'n%s_D%g_ell%g' % (n, D, ell); out['identity'][key] = {'Lambda_direct2d': dr['Lambda'], 'four_keff2_pointfields': 4 * r['k_eff'] ** 2, 'rel': abs(4 * r['k_eff'] ** 2 / dr['Lambda'] - 1)}
        print('identity %s: Lambda (direct 2-d) %.6f  4 k_eff^2 (point fields) %.6f  rel %.1e' % (key, dr['Lambda'], 4 * r['k_eff'] ** 2, out['identity'][key]['rel']), flush=True)
        json.dump(out, open('results/au_exact.json', 'w'), indent=1, default=float)
