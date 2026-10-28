"""Stage AJ summary: lever-arm representation, Donsker-Varadhan chain (Proposition AJ1, Theorem AJ2), its losses, where it fails
(two w-lines + midpoint), the two-line asymptotics (Proposition AJ3), and the (H2) curvature growth.  -> results/summary_AJ.json,
figures/stageAJ.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

J = lambda f: json.load(open('results/%s.json' % f))
S = {}
dv = J('aj_dv'); S['dv_chain'] = {k: {q: v[q] for q in ('D', 'z', 'tau', 'tR', 'ln1_tJ_window', 'eps', 'loss_CS', 'loss_a', 'loss_DV', 'loss_KLjensen_mgf2', 'loss_final',
                                                       'KL', 'KL_jensen', 'KL_final', 'lnmgf', 'Emu_a_over_t2', 'identity_max_rel', 'bound_AJ1', 'bound_over_tR')} for k, v in dv.items()}
sc = J('aj_scan'); S['scan'] = sc
lp = [v for v in sc.values() if v['kind'] == 'lp']; tl = [v for v in sc.values() if v['kind'] == 'tl']
S['lp_Delta_z0'] = {('%g' % v['D']): v['Delta'] for v in lp if v['z'] == 0}
S['lp_max_tR_over_Delta'] = max((v['tR_over_Delta'], v['D'], v['z'], v['Delta']) for v in lp)
S['tl_Delta_z0'] = {('%g' % v['D']): v['Delta'] for v in tl if v['z'] == 0}
S['tl_max_tR_over_Delta'] = max((v['tR_over_Delta'], v['D'], v['z'], v['Delta']) for v in tl)
# lattice ladders: tR / Delta_1/2 by interpolation in ln t
lad = J('ad_qlc_lattice'); L = {}
for k, v in lad.items():
    R = [r for r in v['rows'] if r]
    if len(R) < 3: continue
    t = np.array([r['t'] for r in R]); lJ = np.log([r['J'] for r in R]); tR = np.array([r['tR'] for r in R]); o = np.argsort(t); t, lJ, tR = t[o], lJ[o], tR[o]
    best = None
    for i in range(len(t)):
        if t[i] / 2 < t[0] or tR[i] < 1: continue
        d = float(np.interp(np.log(t[i] / 2), np.log(t), lJ) - lJ[i])
        if best is None or tR[i] / d > best[0]: best = (float(tR[i] / d), float(t[i]), float(tR[i]), d)
    if best: L[k] = best
S['lattice_max_tR_over_Delta'] = max(v[0] for v in L.values()); S['lattice_tR_over_Delta'] = L
ta = J('aj_twoline_asym'); S['twoline_asym'] = ta
h2 = J('aj_h2'); S['h2'] = {k: {q: v[q] for q in ('sigma', 'ln1_sigma', 'max_t2R1', 'at_t2R1', 'max_kappa_H', 'max_tR', 'max_t2R1_minus_1tR', 'max_t2R1_minus_2tR')} for k, v in h2.items()}
xs = [v['ln1_sigma'] for v in h2.values() if v['max_t2R1'] > 1]; ys = [v['max_t2R1'] for v in h2.values() if v['max_t2R1'] > 1]
S['h2_exponent_t2R1_vs_ln1_sigma'] = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
from aj_identity import check
S['identity_check'] = dict(zip(('tweedie_max_rel', 'hessian_cov_max_rel'), check()))
json.dump(S, open('results/summary_AJ.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('lp_Delta_z0', 'lp_max_tR_over_Delta', 'tl_Delta_z0', 'tl_max_tR_over_Delta', 'lattice_max_tR_over_Delta', 'h2_exponent_t2R1_vs_ln1_sigma', 'identity_check')}, default=float))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
for kind, mk, col, lab in (('lp', 'o', 'C1', 'one line + point'), ('tl', 's', 'C3', 'two lines + midpoint')):
    V = [v for v in sc.values() if v['kind'] == kind]; a.scatter([v['Delta'] for v in V], [v['tR'] for v in V], marker=mk, color=col, s=14, label=lab)
a.scatter([v[3] for v in L.values()], [v[2] for v in L.values()], marker='^', color='0.4', s=14, label='lattice ladders (max tR/$\\Delta$)')
xx = np.linspace(1, 260, 50); a.plot(xx, 2 * xx, 'k--', lw=0.8, label='$tR = 2\\Delta$'); a.set_xscale('symlog', linthresh=1); a.set_yscale('log')
a.set_xlabel('$\\Delta_{1/2} = \\ln(J(t/2)/J(t))$'); a.set_ylabel('$tR$'); a.legend(fontsize=7); a.set_title('(a) increment vs $tR$: the two-line family is in neither regime', fontsize=9)
a = ax[1]; R = sorted(ta['rows'].values(), key=lambda r: r['D'])
a.plot([r['D'] for r in R], [r['margin'] for r in R], 'o', color='C3', label='closed form, $z = z^*(D)$')
Dg = np.linspace(8, 62, 50); a.plot(Dg, ta['K0'] - 5 * np.log(Dg), 'k-', lw=1, label='$4.003 - 5\\ln D$ (Prop. AJ3)')
a.set_xlabel('$D$'); a.set_ylabel('$\\max_z\\,[tR - \\ln(1/tJ)]$'); a.legend(fontsize=8); a.set_title('(b) two lines + midpoint: $B\'=1$ sharp, margin $\\to -\\infty$', fontsize=9)
a = ax[2]; H = sorted(h2.values(), key=lambda v: v['ln1_sigma'])
a.loglog([v['ln1_sigma'] for v in H], [max(v['max_t2R1'], 1e-3) for v in H], 'o-', color='C0', label="max $t^2R'$")
a.loglog([v['ln1_sigma'] for v in H], [max(v['max_t2R1_minus_2tR'], 1e-3) for v in H], 's--', color='C2', label="max $(t^2R' - 2tR)$")
a.set_xlabel('$\\ln(1/\\sigma)$'); a.legend(fontsize=8); a.set_title('(c) under (H2): curvature grows like $\\ln^{%.2f}(1/\\sigma)$' % S['h2_exponent_t2R1_vs_ln1_sigma'], fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAJ.png', dpi=130)
