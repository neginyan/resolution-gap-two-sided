"""Stage AH: closed form for the infinite w-line + one point (the size -> infinity limit of the stage-AG 'line + point' family).
t = 1.  rho = (line v = 0 with linear density 1) + eps * delta_(D, 0).  For y = (y_v, y_w):
  line part:  weight phi(y_v),            posterior X = (0, X_w), X_w ~ N(y_w, 1)
  point part: weight eps phi(y_v - D) phi(y_w),  posterior X = (D, 0)
  pi_P = eps phi(y_v - D) phi(y_w) / (phi(y_v) + eps phi(y_v - D) phi(y_w)),  pi_L = 1 - pi_P
  K = Cov(X|y) = pi_L diag(0, 1) + pi_L pi_P Delta Delta^T,  Delta = (D, -y_w);  s = E[X|y] - y;  H = I - K.
Both int p Y and int p s_w^2 are finite (only the neighbourhood of the point contributes to s_w), so the ratio
Y/(tJ_ww) := int p Y / int p s_w^2 is the exact infinite-size limit.  Scan over (D, eps).  -> results/ah_linepoint.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from af_tools import decompose


def fields(D, eps, h=0.04, Wd=12.0):
    yv = np.arange(-9.0, D + 9.0, h); yw = np.arange(-Wd, Wd, h); V, Wg = np.meshgrid(yv, yw, indexing='ij')
    phi = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    pl = phi(V); pp = eps * phi(V - D) * phi(Wg); p = pl + pp; piP = pp / p; piL = 1 - piP
    Dv, Dw = D, -Wg
    Kvv = piL * piP * Dv * Dv; Kvw = piL * piP * Dv * Dw; Kww = piL + piL * piP * Dw * Dw
    sv = piP * D - V; sw = piP * 0.0 + piL * Wg - Wg          # E[X_w] = pi_L y_w + pi_P * 0
    return p, sv, sw, Kvv, Kvw, Kww, h, V, Wg


def ratio(D, eps):
    p, sv, sw, Kvv, Kvw, Kww, h, V, Wg = fields(D, eps)
    W = p * h * h; Jw = np.sum(W * sw * sw)                     # unnormalised: int p s_w^2
    # decompose works with normalised weights and J: pass J = Jw / sum(W) so that ratios are int p Y / int p s_w^2
    d = decompose(W.ravel(), (1 - Kvv).ravel(), (-Kvw).ravel(), (1 - Kww).ravel(), 1.0, Jw / W.sum())
    return d, Jw


if __name__ == '__main__':
    out = {'scan': []}
    for D in np.linspace(3.0, 14.0, 23):
        for le in np.linspace(-6, 4, 21):
            d, Jw = ratio(D, np.exp(le))
            out['scan'].append({'D': float(D), 'ln_eps': float(le), 'Y3': d['A3_a0.5']['Y'], 'Y1': d['A1_a0.55']['Y'], 'X': d['X_eta'], 'kappa_H': d['kappa_H'], 'int_p_sw2': float(Jw)})
    best = max(out['scan'], key=lambda r: r['Y3']); out['max_Y3'] = best
    # resolution check at the maximiser
    d1, _ = ratio(best['D'], np.exp(best['ln_eps']))
    p, *_ = fields(best['D'], np.exp(best['ln_eps']))
    json.dump(out, open('results/ah_linepoint.json', 'w'), indent=1, default=float)
    print('max Y3 = %.4f at D = %.2f, ln eps = %.2f (Y1 %.4f, X %.4f)' % (best['Y3'], best['D'], best['ln_eps'], best['Y1'], best['X']))
    for D in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0):
        rows = [r for r in out['scan'] if abs(r['D'] - D) < 1e-9]
        if rows: m = max(rows, key=lambda r: r['Y3']); print('  D %.1f: max over eps Y3 %.4f (ln eps %.1f)' % (D, m['Y3'], m['ln_eps']))
