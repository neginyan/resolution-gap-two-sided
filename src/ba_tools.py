"""Stage BA tools: general backgrounds x windows with paper E's evaluator (w_full.full_sums, cancellation-free point fields, true V-envelope)
and the branches (a), (a') and Lambda = 4 k_eff^2 (at_tools.sums_envelope + analyse), plus the confinement moments (H1), (H2).

State (w_tools.terms_of): rho = N(v; 0, Lv^2) sum_b beta_b N(w; c_b, L_b^2) [1 + sum_r c_r e^{-(w - w_r)^2/2 ell_r^2} cos(m_r k_b v/sigma)],
lengths in w in units of sqrt(sigma).  Paper coordinates: z = (v + w)/sqrt2, (H2) Var X_z <= pi^2, (H1) E y_z^2 <= pi^2 + sigma^2/2,
i.e. E x_z^2 <= pi^2 (the noise adds sigma^2/2).  The ripple terms carry no net mass (cos averages out over Lv >> period), so the moments are
those of the background mixture.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from w_full import full_sums
from at_tools import sums_envelope, analyse


def moments(sig, bg, Lv_sig):
    s = np.sqrt(sig); bs = sum(b[0] for b in bg); Lv = Lv_sig * sig
    mz = np.array([c * s / np.sqrt(2) for _, c, _ in bg]); vz = np.array([(Lv ** 2 + (L * s) ** 2) / 2 for _, _, L in bg]); w = np.array([b[0] / bs for b in bg])
    Ex2 = float(np.sum(w * (mz ** 2 + vz))); mean = float(np.sum(w * mz))
    return {'E_xz2': Ex2, 'Var_xz': Ex2 - mean ** 2, 'H1': bool(Ex2 <= np.pi ** 2), 'H2': bool(Ex2 - mean ** 2 <= np.pi ** 2)}


def gap(sig, bg, win, kb, Lv_sig, nper=25, coarse=200, n0=15, nv=32, extra_w=None):
    mmax = max([w_[3] for w_ in win] + [1])
    T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    if extra_w is not None: wn = np.unique(np.concatenate([wn, extra_w]))
    o = full_sums(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv * mmax, n0=n0, ext=6.5)
    return o


def branches(sig, bg, win, kb, Lv_sig, nper=50, coarse=400, n0=31, nv=64, extra_w=None):
    mmax = max([w_[3] for w_ in win] + [1])
    T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, nper, coarse)
    if extra_w is not None: wn = np.unique(np.concatenate([wn, extra_w]))
    r = analyse(sums_envelope(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv * mmax, n0=n0), sig)
    den = r['sig2Jvv'] + r['sig2Jww']
    aw = 2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (sig * den); aN = r['sig2Jvv'] * r['Phi'] * (1 - r['delta_c']) / (2 * sig * den)
    return {'g': float(r['g']), 'a_w': float(aw), 'a_N': float(aN), 'Lambda': float(4 * r['k_eff'] ** 2), 'eps': float(r['eps']), 'Phi': float(r['Phi']),
            'sig2Jvv': float(r['sig2Jvv']), 'sig2Jww': float(r['sig2Jww'])}


def strip_opt(sig):
    """paper E's strip optimum at sigma (w1_gmin): k0, lam, M, Lv_sig, c and its g."""
    import json
    d = json.load(open('results/w1_gmin_opt_s%g.json' % sig)); b = d['best']; x = np.array(b['x'])
    k0, lam, M, eta = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4]))
    return {'k0': float(k0), 'lam': float(lam), 'M': float(M), 'Lv_sig': float(eta * lam / np.sqrt(sig)), 'c': float(c), 'g': float(b['opt_gap_over_sigma'])}
