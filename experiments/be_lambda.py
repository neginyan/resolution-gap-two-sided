"""Stage BE §2.1: does the comb window make deep valleys?  Largest posterior excess lambda_max = max eigenvalue of (-tH) over the blur window
t in [sigma^2/4, sigma^2] (paper E's Proposition 14 / stage AG Proposition AG1 hypothesis), on a dense (theta, y_w) grid — the stage-AG method
(ag_strip_lambda.lam_field) extended to a window with harmonics m = 1..mmax (the comb): F = sum_m G_m A(y_w) cos(m theta), G_m = a_m e^{-m^2 kap^2 s_v^2/2}.
Checks: the strip optima reproduce stage AG (1.8176, 1.6034, 1.0816 at 1e-6, 1e-12, 1e-24).
Prop. AG1 constants: B = 8(1 + Lambda*)/(1 - tau^{1 + Lambda*}), c = 1/(pi sqrt(B)) (tau = 1/4).  -> results/be_lambda.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np


def lam_field(sig, kb, amps, ell, Lw, Lv, t, n_theta=1440, n_w=2501):
    kap = kb / sig
    Tv, Tw = Lv * Lv + t, Lw * Lw + t; Kv, Kw = Lv * Lv / Tv, Lw * Lw / Tw; sv2, sw2 = Lv * Lv * t / Tv, Lw * Lw * t / Tw
    e2 = ell * ell + sw2; q0 = kap * Kv
    yw = np.linspace(-9 * np.sqrt(Tw), 9 * np.sqrt(Tw), n_w); sc = np.sqrt(e2) / Kw
    yw = np.unique(np.concatenate([yw, np.linspace(-14 * sc, 14 * sc, n_w)]))
    th = 2 * np.pi * np.arange(n_theta) / n_theta; TH, YW = np.meshgrid(th, yw, indexing='ij')
    A = ell / np.sqrt(e2) * np.exp(-Kw * Kw * YW ** 2 / (2 * e2)); dA = -A * Kw * Kw * YW / e2; ddA = A * (Kw ** 4 * YW ** 2 / e2 ** 2 - Kw * Kw / e2)
    F = np.zeros_like(TH); Fv = np.zeros_like(TH); Fw = np.zeros_like(TH); Fvv = np.zeros_like(TH); Fww = np.zeros_like(TH); Fvw = np.zeros_like(TH)
    for m, a in enumerate(amps, start=1):
        G = a * np.exp(-0.5 * (m * kap) ** 2 * sv2); c, s = np.cos(m * TH), np.sin(m * TH); mq = m * q0
        F += G * A * c; Fv += -G * A * mq * s; Fw += G * dA * c; Fvv += -G * A * mq * mq * c; Fww += G * ddA * c; Fvw += -G * dA * mq * s
    den = 1 + F
    Hvv = 1 / Tv - (Fvv * den - Fv * Fv) / den ** 2; Hww = 1 / Tw - (Fww * den - Fw * Fw) / den ** 2; Hvw = -(Fvw * den - Fv * Fw) / den ** 2
    a_, b_, d_ = -t * Hvv, -t * Hvw, -t * Hww; lam = (a_ + d_) / 2 + np.sqrt((a_ - d_) ** 2 / 4 + b_ * b_)
    dens = np.exp(-YW ** 2 / (2 * Tw)) * den
    return lam, dens / dens.max(), float(den.min())


def lam_star(sig, kb, amps, ell, Lw, Lv, nt=25):
    rows = []
    for m in np.linspace(0.25, 1.0, nt):
        with np.errstate(over='ignore', invalid='ignore'):
            lam, dn, dmin = lam_field(sig, kb, amps, ell, Lw, Lv, m * sig * sig)
        rows.append({'m': float(m), 'lam_max_1e-8': float(lam[dn > 1e-8].max()), 'lam_max_all': float(np.nanmax(lam)), 'min_blurred_profile': dmin})
    L = max(r['lam_max_1e-8'] for r in rows); B = 8 * (1 + L) / (1 - 0.25 ** (1 + L))
    return {'rows': rows, 'Lambda_star': L, 'Lambda_star_all': max(r['lam_max_all'] for r in rows), 'B': B, 'c': 1 / (np.pi * np.sqrt(B))}


if __name__ == '__main__':
    out = {}
    Z = json.load(open('results/z_chain_strips.json'))
    for sg in ('1e-06', '1e-12', '1e-24'):
        z = Z['strip_gmin_%s' % sg]; sig = z['sigma']; ell = z['lam'] * np.sqrt(sig)
        r = lam_star(sig, z['k0'], [z['c']], ell, z['M'] * ell, z['eta'] * ell); out['strip_' + sg] = r
        print('strip %s: Lambda* %.4f (stage AG: 1.8176 / 1.6034 / 1.0816)  B %.2f  c %.4f' % (sg, r['Lambda_star'], r['B'], r['c']), flush=True)
        D = json.load(open('results/bd_opt_comb_%s.json' % sg)); x = np.exp(D['x'][:4]); amps = [w[0] for w in D['windows']]
        ell = x[1] * np.sqrt(sig); r = lam_star(sig, D['kb'], amps, ell, x[2] * ell, x[3] * ell); out['comb_' + sg] = r
        print('comb  %s: Lambda* %.4f (all points %.4f)  B %.2f  c %.4f  | at t = s^2/4: %.4f, s^2/2: %.4f, s^2: %.4f; min blurred profile %.3e' % (
            sg, r['Lambda_star'], r['Lambda_star_all'], r['B'], r['c'], r['rows'][0]['lam_max_1e-8'], r['rows'][8]['lam_max_1e-8'], r['rows'][-1]['lam_max_1e-8'],
            r['rows'][0]['min_blurred_profile']), flush=True)
        json.dump(out, open('results/be_lambda.json', 'w'), indent=1)
