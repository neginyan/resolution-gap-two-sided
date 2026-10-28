"""Stage AL, plan A summary: Proposition AL1 (anisotropic chain: w-slices by the 1-d theorem, mixed part by the v-only lever arm),
kappa_v on the families, lattices, the staggered double row.  -> results/summary_AL.json, figures/stageAL.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from al_aniso import one_d_constants

J = lambda f: json.load(open('results/%s.json' % f))
S = {'one_d_constants': one_d_constants()}
R = J('al_run'); S['families'] = R
for kind in ('lp', 'tl', 'dot'):
    rows = [v for v in R.values() if v['kind'] == kind]
    S['%s_kappa_v_range' % kind] = [min(v['kappa_v'] for v in rows), max(v['kappa_v'] for v in rows)]
    S['%s_Bneeded_tau0.25' % kind] = [v['Bneeded_AL1'] for v in rows if v['tau_v'] == 0.25]
    S['%s_identity_max' % kind] = max(v['identity_c2v_rel_L2'] for v in rows)
L = J('al_lattice'); S['lattice'] = L
S['product_lattice_kappa_v'] = max(v['kappa_v'] for v in L.values() if v['s'] == 0.0)
S['oblique_lattice_kappa_v_tau0.75'] = {k: v['kappa_v'] for k, v in L.items() if v['s'] != 0.0 and v['tau_v'] == 0.75}
SG = J('al_staggered'); S['staggered'] = {k: {q: v[q] for q in ('D', 'a', 'Delta_half', 'tR', 'tR_vw', 'ln1_tJ', 'Psi_v', 'tR_over_Delta', 'Bneeded_AL1_best')} | {'kappa_v_tau': {t: w['kappa_v'] for t, w in v['taus'].items()}} for k, v in SG.items()}
AKd = J('ak_dotted'); S['dotted_compare'] = {k: {'Psi_isotropic_AK1': v['Psi'], 'B_AK1_best': v['Bneeded_AK1_best']} for k, v in AKd.items()}
json.dump(S, open('results/summary_AL.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('lp_kappa_v_range', 'tl_kappa_v_range', 'dot_kappa_v_range', 'lp_Bneeded_tau0.25', 'tl_Bneeded_tau0.25', 'dot_Bneeded_tau0.25', 'product_lattice_kappa_v')}, default=float))
for k, v in S['staggered'].items(): print('staggered', k, v)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2)); a = ax[0]
dots = sorted([v for v in R.values() if v['kind'] == 'dot' and v['tau_v'] == 0.25], key=lambda v: v['D'])
a.plot([v['D'] for v in dots], [v['Bneeded_AL1'] for v in dots], 'o-', color='C2', label="Prop. AL1 (anisotropic), $\\tau_v$ = 1/4")
ak = sorted(AKd.values(), key=lambda v: v['D']); a.plot([v['D'] for v in ak], [v['Bneeded_AK1_best'] for v in ak], 's--', color='C3', label='Prop. AK1 (isotropic), best $\\tau$')
st = sorted(S['staggered'].values(), key=lambda v: v['D']); a.plot([v['D'] for v in st], [v['Bneeded_AL1_best'] for v in st], '^:', color='C0', label='staggered double row, AL1 best $\\tau_v$')
a.set_yscale('log'); a.set_xlabel('D'); a.set_ylabel("B' needed"); a.legend(fontsize=7); a.set_title('(a) two lines + dotted row: the gap of stage AK closes', fontsize=9)
a = ax[1]
for k, v in S['staggered'].items():
    T = sorted(v['kappa_v_tau'].items(), key=lambda x: float(x[0])); a.semilogy([float(x) for x, _ in T], [y for _, y in T], 'o-', label='staggered rows D, a = %s' % k.replace('_', ', '))
for k, v in L.items():
    pass
obl = {}
for k, v in L.items():
    if v['s'] != 0: obl.setdefault('%g,%g,%g' % (v['a1'], v['s'], v['a2']), []).append((v['tau_v'], v['kappa_v']))
for k, T in obl.items(): T.sort(); a.semilogy([x for x, _ in T], [y for _, y in T], '--', lw=0.8, color='0.5')
a.axhline(1, color='C2', lw=1, label='all hidden-mass families, dotted rows, product lattices')
a.set_xlabel('$\\tau_v$'); a.set_ylabel('$\\kappa_v$'); a.legend(fontsize=6); a.set_title('(b) v-direction sign cancellation (grey: oblique lattices)', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAL.png', dpi=130)
