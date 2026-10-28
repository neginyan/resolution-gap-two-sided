"""Stage AN: independent check of the leading-order formula Lambda = Lambda_v = E[(s g + 2 g')^2]/E[g^2] for the theta lift, by direct
two-dimensional summation (two w-lines + dotted rows as explicit point masses; analytic derivatives of p).  Lambda (paper E) =
t E[(s_v s_w - 2 H_vw)^2]/J, H_vw = -d_v s_w, t = 1.  Grid: one w-period x v-window, <= 8e6 entries per array (chunked over v).
-> results/an_lambda2d.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from an_signed1d import quantities


def lam2d(lines, rows, a, vlo, vhi, hv=0.005, nw=64, nper=8):
    """lines: [(v0, density)], rows: [(v0, linear density, phase)] -> (Lambda, tR, J) from direct sums."""
    w = (np.arange(nw) + 0.5) * a / nw; acc = np.zeros(4)            # sums of p s_w^2, p (s_v s_w + 2 d_v s_w)^2, p |grad s_w|^2, p
    pts = [(v0, m * a, ph * a + j * a) for v0, m, ph in rows for j in range(-nper, nper + 1)]       # point mass = density x spacing
    vs = np.arange(vlo, vhi, hv); ch = max(1, int(6e6 // (nw * (len(pts) + 1))))
    for i0 in range(0, len(vs), ch):
        V = vs[i0:i0 + ch][:, None]; W = w[None, :]
        p = np.zeros((V.shape[0], nw)); pv = np.zeros_like(p); pw = np.zeros_like(p); pvw = np.zeros_like(p); pww = np.zeros_like(p); pvv = np.zeros_like(p)
        for v0, dens in lines:
            e = dens * np.exp(-(V - v0) ** 2 / 2) / np.sqrt(2 * np.pi) * np.ones_like(W)
            p += e; pv += -(V - v0) * e; pvv += ((V - v0) ** 2 - 1) * e
        for v0, m, w0 in pts:
            e = m * np.exp(-(V - v0) ** 2 / 2 - (W - w0) ** 2 / 2) / (2 * np.pi)
            dv, dw = -(V - v0), -(W - w0)
            p += e; pv += dv * e; pw += dw * e; pvw += dv * dw * e; pww += (dw * dw - 1) * e; pvv += (dv * dv - 1) * e
        sv, sw = pv / p, pw / p; dvsw = pvw / p - pv * pw / p ** 2; dwsw = pww / p - pw ** 2 / p ** 2
        acc += [np.sum(p * sw ** 2), np.sum(p * (sv * sw + 2 * dvsw) ** 2), np.sum(p * (dvsw ** 2 + dwsw ** 2)), np.sum(p)]
    return acc[1] / acc[0], acc[2] / acc[0], acc[0] / acc[3]


if __name__ == '__main__':
    out = {}
    for D, a in ((8.0, 2.0), (10.0, 2.0)):
        z, d = 0.1, 1.0; m = np.exp(-D * (z + D / 2)) / 2
        L2, tR2, J2 = lam2d([(0.0, 1.0), (2 * D, 1.0)], [(D - d, m, 0.0), (D + d, m, 0.5)], a, -8.0, 2 * D + 8.0)
        o = quantities(np.array([0.0, 2 * D, D - d, D + d]), np.array([0.0, 0.0, np.log(m), np.log(m)]), np.array([0.0, 0.0, 1.0, -1.0]))
        k = 2 * np.pi / a
        out['%g_%g' % (D, a)] = {'Lambda_2d': L2, 'Lambda_v': o['Lam_v'], 'rel': abs(L2 / o['Lam_v'] - 1), 'tR_2d': tR2, 'tR_lead': k * k + o['Rq']}
        print('D %g a %g: Lambda 2-d direct %.5f vs Lambda_v %.5f (rel %.1e); tR 2-d %.4f vs k^2 + Rq %.4f' % (D, a, L2, o['Lam_v'], abs(L2 / o['Lam_v'] - 1), tR2, k * k + o['Rq']), flush=True)
    # a hidden multipole (n = 2) lifted to 2-d: rows at x_i with densities w_i, phase 0 for f = +1, a/2 for f = -1 (v shifted by D)
    from an_multipole import realise
    Mp = json.load(open('results/an_multipole.json')); L = Mp['leading']['2']; D = 20.0; a = 1.5
    xs, lw, f = realise(D, L['alpha'], np.array(L['p']), 1.0, 8.0); lw = lw - lw.max()
    lines = [(xs[i] + D, np.exp(lw[i])) for i in range(2)]; rows = [(xs[i] + D, np.exp(lw[i]), 0.0 if f[i] > 0 else 0.5) for i in range(2, len(xs))]
    L2, tR2, J2 = lam2d(lines, rows, a, -8.0, 2 * D + 8.0, hv=0.0025)
    o = quantities(xs, lw, f); k = 2 * np.pi / a
    out['multipole_n2_D20'] = {'Lambda_2d': L2, 'Lambda_v': o['Lam_v'], 'rel': abs(L2 / o['Lam_v'] - 1), 'tR_2d': tR2, 'tR_lead': k * k + o['Rq'], 'rel_tR': abs(tR2 / (k * k + o['Rq']) - 1)}
    print('multipole n = 2, D = 20, a = 1.5: Lambda 2-d direct %.5f vs Lambda_v %.5f (rel %.1e); tR 2-d %.4f vs k^2 + Rq %.4f' % (L2, o['Lam_v'], abs(L2 / o['Lam_v'] - 1), tR2, k * k + o['Rq']), flush=True)
    json.dump(out, open('results/an_lambda2d.json', 'w'), indent=1)
