"""Stage AH (milestone: counterexample candidate): summary (results/summary_AH.json) and figure (figures/stageAH.png/.pdf).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ah_lp3 import fields

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

if __name__ == '__main__':
    C = json.load(open('results/ah_counter.json')); inf = C['infinite_line']
    out = {'rows': inf, 'finite': C['finite']}
    z0 = [r for r in inf if r['z'] == 0]; zh = [r for r in inf if abs(r['z'] - 0.5 * r['D']) < 1e-9]
    Ds = np.array([r['D'] for r in z0])
    fit = lambda y: float(np.polyfit(np.log(Ds), np.log(np.array(y)), 1)[0])
    out['growth_exponents_z0'] = {'Y3': fit([r['Y3'] for r in z0]), 'kappa_H': fit([r['kappa_H'] for r in z0]), 't2R1': fit([r['t2R1'] for r in z0]), 'tR': fit([r['tR'] for r in z0]),
                                  't2R1_over_tR': fit([r['t2R1'] / r['tR'] for r in z0]), 'K_needed': fit([r['K_needed_tau0.25'] for r in z0])}
    out['max'] = {q: max(r[q] for r in inf) for q in ('Y3', 'kappa_H', 't2R1', 'K_needed_tau0.25')}
    out['max_t2R1_over_tR'] = max(r['t2R1'] / r['tR'] for r in inf); out['max_Lambda_over_ln'] = max(r['Lambda'] / r['ln_1_over_tJ'] for r in inf)
    out['max_tR_over_ln'] = max(r['tR'] / r['ln_1_over_tJ'] for r in inf)
    fin = {(r['D'], r['z']): [] for r in C['finite']}
    for r in C['finite']: fin[(r['D'], r['z'])].append(r)
    out['finite_convergence'] = {'%g_%g' % k: {'f_env': [r['f_env'] for r in v], 'Y3': [r['Y3'] for r in v], 'infinite_Y3': next(q['Y3'] for q in inf if q['D'] == k[0] and q['z'] == k[1])} for k, v in fin.items()}
    # pointwise check of the closed-form Hessian against finite differences of ln p (independent of the posterior formulas)
    from scipy.special import expit
    lphi = lambda x, t: -x * x / (2 * t) - 0.5 * np.log(2 * np.pi * t)
    D, L, t = 10.0, 50.0, 1.0; lpf = lambda v, w: np.logaddexp(lphi(v, t), -L + lphi(v - D, t) + lphi(w, t)); hh = 1e-3; worst = 0.0
    rng = np.random.default_rng(1)
    for _ in range(200):
        v, w = rng.uniform(5, 15), rng.uniform(-3, 3)
        fd = (-(lpf(v + hh, w) - 2 * lpf(v, w) + lpf(v - hh, w)) / hh ** 2, -(lpf(v + hh, w + hh) - lpf(v + hh, w - hh) - lpf(v - hh, w + hh) + lpf(v - hh, w - hh)) / (4 * hh * hh),
              -(lpf(v, w + hh) - 2 * lpf(v, w) + lpf(v, w - hh)) / hh ** 2)
        lp, h, sv, sw, Hvv, Hvw, Hww, (Vv, Ww) = fields(D, L, t, h=0.5)          # coarse grid only to get the formula at grid nodes: evaluate formula directly instead
        lpL = lphi(v, t); lpP = -L + lphi(v - D, t) + lphi(w, t); pP = expit(lpP - lpL); pL = 1 - pP
        f = ((1 - pL * pP * D * D) / t, -(pL * pP * D * (-w)) / t / t, (pP - pL * pP * w * w) / t)
        worst = max(worst, max(abs(a - b) / (1 + abs(b)) for a, b in zip(fd, f)))
        break
    for _ in range(200):
        v, w = rng.uniform(5, 15), rng.uniform(-3, 3)
        fd = (-(lpf(v + hh, w) - 2 * lpf(v, w) + lpf(v - hh, w)) / hh ** 2, -(lpf(v + hh, w + hh) - lpf(v + hh, w - hh) - lpf(v - hh, w + hh) + lpf(v - hh, w - hh)) / (4 * hh * hh),
              -(lpf(v, w + hh) - 2 * lpf(v, w) + lpf(v, w - hh)) / hh ** 2)
        lpL = lphi(v, t); lpP = -L + lphi(v - D, t) + lphi(w, t); pP = expit(lpP - lpL); pL = 1 - pP
        f = ((1 - pL * pP * D * D) / t, -(pL * pP * D * (-w)) / t / t, (pP - pL * pP * w * w) / t)
        worst = max(worst, max(abs(a - b) / (1 + abs(b)) for a, b in zip(fd, f)))
    out['hessian_fd_check_max_rel'] = worst
    json.dump(out, open('results/summary_AH.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'finite')}, indent=1, default=float))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2)); a = ax[0]
    for q, col, lab in (('Y3', OI[6], '$Y/(tJ_{ww})$'), ('kappa_H', OI[5], '$\\kappa_H$'), ('t2R1', OI[3], "$t^2R'$"), ('tR', OI[0], '$tR$')):
        a.plot(Ds, [r[q] for r in z0], 'o-', color=col, label=lab)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('D (point distance from the line, units $\\sqrt{t}$), z = 0'); a.legend(frameon=False, fontsize=7)
    a.set_title('infinite w-line + one point: unbounded curvature terms')
    b = ax[1]
    for zz, mk in ((0.0, 'o'), (0.25, 's'), (0.5, '^')):
        R = [r for r in inf if abs(r['z'] - zz * r['D']) < 1e-9]
        b.plot([r['D'] for r in R], [r['Lambda'] / r['ln_1_over_tJ'] for r in R], mk + '-', color=OI[6], label='$\\Lambda$/ln(1/$tJ_{ww}$), z = %gD' % zz)
        b.plot([r['D'] for r in R], [r['K_needed_tau0.25'] for r in R], mk + '--', color=OI[5], label='K needed (qLC$_w$, $c_0$ = 0), z = %gD' % zz)
    b.axhline(4, color=OI[0], lw=0.6, ls=':'); b.set_xlabel('D'); b.legend(frameon=False, fontsize=6.5); b.set_title('the target ($\\Lambda \\leq 4\\ln$ + const) is not threatened')
    fig.tight_layout(); fig.savefig('figures/stageAH.png', dpi=160); fig.savefig('figures/stageAH.pdf'); plt.close(fig)
