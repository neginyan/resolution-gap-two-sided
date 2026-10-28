"""Stage AI: independent closed-form evaluator of the two-line + point family (shares nothing with ai_mixlog): lines v = 0 (density 1)
and v = 2D + delta (density e^{lr}), point (D, 0) of weight e^{-L}; t = 1.  Posterior weights by log-softmax; with pi_P, pi_1, pi_2:
  s_w = -pi_P y_w,  s_v = E v - y_v,  H_ww = pi_P - pi_P (1 - pi_P) y_w^2,
  H_vw = -C_vw = -y_w [pi_P pi_2 (v2 - D) ... ] (general form below, written as sums of products of positive weights),
nonuniform y_v grid (fine near the two valleys), uniform y_w grid, sums in the log domain.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import numpy as np
from scipy.special import logsumexp

LN2PI = np.log(2 * np.pi)


def evaluate(D, z, lr=0.0, delta=0.0, Wmax=8.0, nw=801):
    L = D * (z + D / 2); v2 = 2 * D + delta
    fine_lo, fine_hi = D - abs(z) - 4.0 - 40.0 / D, D + abs(z) + 4.0 + 40.0 / D
    yv = np.unique(np.concatenate([np.arange(-9.0, v2 + 9.0, 0.004), np.arange(fine_lo, fine_hi, min(0.002, 0.04 / D))])); dv = np.gradient(yv)
    yw = np.linspace(-Wmax, Wmax, nw); dw = yw[1] - yw[0]
    acc = {'J': [], 'G': [], 'Q': []}
    for ch in np.array_split(yw, max(1, int(len(yv) * nw / 3e6) + 1)):
        V, W = np.meshgrid(yv, ch, indexing='ij')
        a1 = -V ** 2 / 2 - 0.5 * LN2PI; a2 = lr - (V - v2) ** 2 / 2 - 0.5 * LN2PI; aP = -L - ((V - D) ** 2 + W ** 2) / 2 - LN2PI
        lp = np.logaddexp(np.logaddexp(a1, a2), aP); l1, l2, lP = a1 - lp, a2 - lp, aP - lp
        p1, p2, pP = np.exp(l1), np.exp(l2), np.exp(lP)
        # C_vw = sum_{c<c'} pi_c pi_c' (mv_c - mv_c')(mw_c - mw_c'):  pairs (1,P): (0 - D)(y_w - 0); (2,P): (v2 - D)(y_w); (1,2): mw equal -> 0
        # so C_vw = pi_P y_w [ (v2 - D) pi_2 - D pi_1 ]  (written in logs)
        lw = lp + np.log(dv)[:, None] + np.log(dw)
        with np.errstate(divide='ignore'):
            law = np.log(np.abs(W))
            bracket = (v2 - D) * p2 - D * p1
            lHvw2 = 2 * (lP + law + np.log(np.abs(bracket)))
            # exact in logs where pi_P underflows: pi_P (1-pi_P) y_w^2 and pi_P are both proportional to e^{lP}
            lHww2 = 2 * lP + 2 * np.log(np.abs(1.0 - (p1 + p2) * W ** 2))
            lsw2 = 2 * (lP + law)
            # s_v s_w - 2H_vw = -pi_P y_w (E v - y_v) + 2 pi_P y_w [(v2 - D) pi_2 - D pi_1] = pi_P y_w [ y_v - E v + 2((v2-D) pi_2 - D pi_1) ]
            Ev = D * pP + v2 * p2
            lq2 = 2 * (lP + law) + 2 * np.log(np.abs(V - Ev + 2 * bracket))
            acc['J'].append(logsumexp(lw + lsw2)); acc['G'].append(logsumexp(lw + np.logaddexp(lHvw2, lHww2))); acc['Q'].append(logsumexp(lw + lq2))
    lJ, lG, lQ = (logsumexp(acc[k]) for k in ('J', 'G', 'Q'))
    lnM = float(logsumexp([0.0, lr, -L]))
    return {'tR': float(np.exp(lG - lJ)), 'Lambda': float(np.exp(lQ - lJ)), 'ln1_tJ': float(lnM - lJ)}


if __name__ == '__main__':
    for D, z in ((8.0, 0.0), (14.0, 0.1), (20.0, 0.1)):
        r = evaluate(D, z); print(D, z, {k: round(v, 4) for k, v in r.items()}, 'tR-ln %.3f' % (r['tR'] - r['ln1_tJ']))
