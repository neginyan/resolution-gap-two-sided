"""Stage AZ (map, item 3): the branches (a), (a') and Lambda = 4 k_eff^2 on the strip x multipole hybrids of stage AQ (H-A), at sigma = 1e-4 and 1e-12
(paper E's evaluator with the AT2 factors, at_tools.sums_envelope).  -> results/az_hybrids.json.  Python 3.11."""
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
from aq_hybrid import strip_params, rows_terms, multipole
from at_tools import sums_envelope, analyse


def branches(sig, n=None, beta=0.0):
    P = strip_params(sig); k0 = P['k0']; s = np.sqrt(sig); ell = P['lam'] * s
    kw = 1.0 if ell / sig <= 2e4 else 2 * np.pi * 8 / (ell / sig)
    bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'], 0.0, P['lam'], 1)]
    T = terms_of(bg, win, k0, sig, P['Lv_sig']); wn = w_nodes(bg, win, sig, 50, 400)
    if n is not None:
        xs, a, f = multipole(n, np.pi / k0, 0.6); a = a / a.sum(); T = T + rows_terms(sig, P, k0, xs, a, f, beta, kw)
        wn = np.unique(np.concatenate([wn, np.arange(-6 * ell, 6 * ell, 2 * np.pi * sig / kw / 10)]))
    r = analyse(sums_envelope(T, sig, k0, np.sqrt((P['Lv_sig'] * sig) ** 2 + sig ** 2), wn), sig)
    den = r['sig2Jvv'] + r['sig2Jww']
    aw = 2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (sig * den); aN = r['sig2Jvv'] * r['Phi'] * (1 - r['delta_c']) / (2 * sig * den)
    return {'g': r['g'], 'a_w': aw, 'a_N': aN, 'Lambda': 4 * r['k_eff'] ** 2, 'AT2': r['bound_AT2'], 'eps': r['eps'], 'kw': kw}


if __name__ == '__main__':
    fn = 'results/az_hybrids.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
    for sig in (1e-4, 1e-12):
        for n, beta in ((None, 0.0), (1, 1e-3), (1, 0.1), (2, 1e-3), (2, 0.1)):
            key = '%g_%s_%g' % (sig, 'strip' if n is None else 'n%d' % n, beta)
            if key in out: continue
            t0 = time.time(); r = branches(sig, n, beta); r['seconds'] = time.time() - t0; out[key] = r; json.dump(out, open(fn, 'w'), indent=1, default=float)
            ln = np.log(1 / sig)
            print('%-22s g %.5g  (a) %.4g  (a\') %.4g  Lambda %.4g (Lambda/ln %.3f)  AT2 %.4g  [(a)+(a\')] sqrt(ln) %.3f  %.0fs' % (key, r['g'], r['a_w'], r['a_N'], r['Lambda'], r['Lambda'] / ln, r['AT2'],
                  (r['a_w'] + r['a_N']) * np.sqrt(ln), r['seconds']), flush=True)
