"""Stage AQ: strip x hidden-multipole hybrids — can they beat the strip family's g_min?

Strip (paper E, w1_gmin): rho = N(v;0,Lv^2) N(w;0,Lw^2) (1 + c e^{-w^2/2 ell^2} cos(k0 v/sigma)), ell = lam sqrt(sigma), Lw = M ell.
Hybrid H-A: add, inside the ripple window, v-periodic signed rows (period of the ripple) at the ripple trough with w-modulation:
  beta N(v;0,Lv^2) N(w;0,Lw^2) e^{-w^2/2 ell^2} sum_i a_i C(v - x_i) [1 + f_i cos(kw w/sigma + phi)],
  C = periodic Gaussian comb (mean 1, tooth width sv sigma), sources x_i at the trough with multipole weights (n = 1: +,- ; n = 2: +,-,+).
Hybrid H-C: the 'comb strip': the window's cosine is replaced by a periodic comb of thin teeth (period 2 D sigma, deep valleys when c -> 1),
  with the same multipole rows in each valley.
The TRUE gap is computed with paper E's evaluator (w_full.full_sums: exact point fields, true V-envelope, period averages).
Python 3.11."""
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
from w_full import full_sums, KEYS
from v3_tools import _pre, point_fields, _finish
from u_cgm import trap_weights


def full_sums_lowmem(terms, sig, kb, Lv, wn, nv=32, n0=15, ext=6.5, max_entries=2.5e6):
    """w_full.full_sums with chunking over grid POINTS (memory ~ points x terms <= max_entries); identical arithmetic otherwise."""
    pre = _pre(terms, sig); per = 2 * np.pi * sig / kb; ww = trap_weights(wn)
    if ext * Lv / per > 1e12: raise ValueError('positional precision')
    v0s = np.linspace(-ext * Lv, ext * Lv, n0); w0 = trap_weights(v0s); tot = {k: 0.0 for k in KEYS}; mass = 0.0
    npts = max(256, int(max_entries // max(1, len(pre))))
    for v0, cw in zip(v0s, w0):
        vn = v0 + (np.arange(nv) + 0.5) * per / nv - per / 2
        V, W = np.meshgrid(vn, wn, indexing='ij'); X = np.stack([V.ravel(), W.ravel()], 1); dA = np.outer(np.full(nv, 1.0 / nv), ww).ravel()
        acc = {k: 0.0 for k in KEYS}; m = 0.0
        for i0 in range(0, len(X), npts):
            p, q = point_fields(pre, sig, X[i0:i0 + npts]); pw = p * dA[i0:i0 + npts]; m += pw.sum()
            for k in acc: acc[k] += float(np.sum(pw * q[k]))
        mass += cw * m
        for k in tot: tot[k] += cw * acc[k]
    S = {k: v / mass for k, v in tot.items()}; out = _finish(S, sig, 'Nibp'); out['grid_mass'] = mass; out['N_ibp'] = S['Nibp']
    return out


def strip_x(sig):
    d = json.load(open('results/w1_gmin_opt_s%g.json' % sig)) if os.path.exists('results/w1_gmin_opt_s%g.json' % sig) else json.load(open('results/w1_gmin_grid_opt_s%g.json' % sig))
    b = d['best'] if 'best' in d else min(d['runs'], key=lambda r: r['opt_gap_over_sigma'])
    return b


def strip_params(sig):
    b = strip_x(sig); x = np.array(b['x']); grid = not os.path.exists('results/w1_gmin_opt_s%g.json' % sig)
    k0, lam, M, e3 = np.exp(x[:4]); c = 1 / (1 + np.exp(-x[4]))
    Lv_sig = e3 if grid else e3 * lam / np.sqrt(sig)          # w1_gmin_grid: x[3] = ln(Lv/sigma); w1_gmin: Lv = eta * ell
    return {'k0': float(k0), 'lam': float(lam), 'M': float(M), 'Lv_sig': float(Lv_sig), 'c': float(c), 'g_stored': b['opt_gap_over_sigma']}


def window_env(sig, P):
    """bg x window envelope (as terms_of): mean in w, covariance, amplitude factor."""
    s = np.sqrt(sig); Lv = P['Lv_sig'] * sig; Lb2 = (P['lam'] * P['M'] * s) ** 2; e2 = (P['lam'] * s) ** 2
    Sbr = 1 / (1 / Lb2 + 1 / e2); Abr = np.sqrt(Sbr / Lb2)
    return Lv, Sbr, Abr


def comb_coeffs(xs, a, kb, sv, mmax):
    """Fourier coefficients b_m (m = -mmax..mmax) of sum_i a_i C(v - x_i), C the mean-1 Gaussian comb of period 2 pi/kb (sigma units)."""
    ms = np.arange(-mmax, mmax + 1)
    b = np.array([np.exp(-(m * kb * sv) ** 2 / 2) * np.sum(a * np.exp(-1j * m * kb * xs)) for m in ms])
    return ms, b


def rows_terms(sig, P, kb, xs, a, f, beta, kw, phi=0.0, sv=0.8, mmax=None):
    Lv, Sbr, Abr = window_env(sig, P); mmax = mmax or int(np.ceil(5.5 / (kb * sv)))
    T = []; S = np.diag([Lv ** 2, Sbr])
    ms, b0 = comb_coeffs(xs, a, kb, sv, mmax); _, b1 = comb_coeffs(xs, a * f, kb, sv, mmax)
    for m, c0, c1 in zip(ms, b0, b1):
        kv = m * kb / sig
        if abs(c0) > 1e-300: T.append((np.log(beta * Abr * c0 + 0j), np.zeros(2), S, np.array([kv, 0.0])))
        if abs(c1) > 1e-300:
            for sg in (1, -1): T.append((np.log(beta * Abr * c1 / 2 + 0j) + 1j * sg * phi, np.zeros(2), S, np.array([kv, sg * kw / sig])))
    return T


def multipole(n, centre, d):
    if n == 1: return np.array([centre - d / 2, centre + d / 2]), np.array([1.0, 1.0]), np.array([1.0, -1.0])
    if n == 2: return np.array([centre - d, centre, centre + d]), np.array([1.0, 2.0, 1.0]), np.array([1.0, -1.0, 1.0])
    raise ValueError


def gap_of(sig, P, extra=None, kb=None, kw=None, comb=None, nper=25, coarse=200, n0=15, nv=32):
    """TRUE gap of the strip P (optionally with the window profile replaced by a comb, and with extra terms)."""
    k0 = P['k0'] if kb is None else kb
    if comb is None:
        bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'], 0.0, P['lam'], 1)]
    else:                                       # comb strip: harmonics m = 1..mmax with Gaussian-comb amplitudes (teeth at v = 0)
        bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'] * 2 * np.exp(-(m * k0 * comb['st']) ** 2 / 2), 0.0, P['lam'], m) for m in range(1, comb['mmax'] + 1)]
    T = terms_of(bg, win, k0, sig, P['Lv_sig'])
    if extra: T = T + extra
    wn = w_nodes(bg, win, sig, nper, coarse)
    if kw is not None:                          # resolve the w-modulation inside the window
        s = np.sqrt(sig); ell = P['lam'] * s; per_w = 2 * np.pi * sig / kw
        wn = np.unique(np.concatenate([wn, np.arange(-6 * ell, 6 * ell, per_w / 10)]))
    return full_sums_lowmem(T, sig, k0, np.sqrt((P['Lv_sig'] * sig) ** 2 + sig ** 2), wn, nv=nv, n0=n0, ext=6.5)


if __name__ == '__main__':
    sig = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    P = strip_params(sig); t0 = time.time(); o = gap_of(sig, P)
    print('strip sigma %g: g = %.5f (stored %.5f), %.1f s' % (sig, o['gap_over_sigma'], P['g_stored'], time.time() - t0), flush=True)
    xs, a, f = multipole(1, np.pi / P['k0'], 0.6); kw = 1.0
    ell_over_sig = P['lam'] / np.sqrt(sig)
    if ell_over_sig > 2e4: kw = 2 * np.pi * 8 / ell_over_sig          # unresolvable sigma-scale modulation: use 8 periods per window width
    t0 = time.time(); o2 = gap_of(sig, P, rows_terms(sig, P, P['k0'], xs, a, f, 1e-3, kw), kw=kw)
    print('  + dipole rows (beta 1e-3, kw %.3g): g = %.5f, %.1f s' % (kw, o2['gap_over_sigma'], time.time() - t0), flush=True)
