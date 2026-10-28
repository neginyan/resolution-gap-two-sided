"""Stage AM summary: oblique residual directions (Proposition AM1) on the staggered double row and on oblique lattices; theta-function
decomposition of the staggered row.  -> results/summary_AM.json, figures/stageAM.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

S = {}
scan = {}
for f in sorted(glob.glob('results/am_scan_*.json')): scan.update(json.load(open(f)))
S['scan'] = scan
best = {}
for k, v in scan.items():
    key = '%g_%g' % (v['D'], v['a'])
    if key not in best or v['Bneeded_AM1'] < best[key]['Bneeded_AM1']: best[key] = v
S['best_over_theta_tau'] = {k: {q: v[q] for q in ('D', 'a', 'deg', 'tau', 'kappa_e', 'Bneeded_AM1')} for k, v in best.items()}
S['theta0_is_best'] = {k: v['deg'] == 0 for k, v in best.items()}
# per tau: is theta = 0 the minimiser of kappa_e?
mins = {}
for k, v in scan.items():
    key = '%g_%g_%g' % (v['D'], v['a'], v['tau'])
    if key not in mins or v['kappa_e'] < mins[key]['kappa_e']: mins[key] = v
S['argmin_theta_kappa_per_tau'] = {k: v['deg'] for k, v in mins.items()}
st = json.load(open('results/al_staggered.json'))
S['staggered_theta0_AL'] = {k: {'Psi_v': v['Psi_v'], 'B_AL1_best': v['Bneeded_AL1_best']} for k, v in st.items()}
if os.path.exists('results/am_lattice.json'):
    L = json.load(open('results/am_lattice.json')); lb = {}
    for k, v in L.items():
        key = '%g_%g_%g_%g' % (v['a1'], v['s'], v['a2'], v['tau'])
        if key not in lb or v['kappa_e'] < lb[key]['kappa_e']: lb[key] = v
    S['lattice_best'] = {k: {'deg': v['deg'], 'kappa_e': v['kappa_e'], 'kappa_theta0': L[k + '_0']['kappa_e']} for k, v in lb.items()}
S['theta'] = json.load(open('results/am_theta.json'))
json.dump(S, open('results/summary_AM.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('best_over_theta_tau', 'argmin_theta_kappa_per_tau')}, default=float, indent=0)[:2500])
if 'lattice_best' in S: print(json.dumps(S['lattice_best'], default=float))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2)); a = ax[0]
for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
    R = sorted([v for v in scan.values() if v['D'] == 8 and v['tau'] == tau], key=lambda v: v['deg'])
    if R: a.semilogy([v['deg'] for v in R], [v['kappa_e'] for v in R], 'o-', ms=3, label='$\\tau$ = %g' % tau)
a.set_xlabel('direction of the residual noise $\\theta$ (deg)'); a.set_ylabel('$\\kappa_e$'); a.legend(fontsize=7); a.set_title('(a) staggered rows D = 8, a = 2: $\\theta$ = 0 is best', fontsize=9)
a = ax[1]
Ds = sorted({v['D'] for v in scan.values()})
a.plot(Ds, [best['%g_%g' % (D, [v['a'] for v in scan.values() if v['D'] == D][0])]['Bneeded_AM1'] for D in Ds], 'o-', color='C3', label='min over $\\theta$, $\\tau$ (Prop. AM1)')
a.plot([v['D'] for v in sorted(st.values(), key=lambda v: v['D'])], [v['Bneeded_AL1_best'] for v in sorted(st.values(), key=lambda v: v['D'])], 's--', color='C0', label='$\\theta$ = 0 (Prop. AL1), best $\\tau_v$')
a.set_xlabel('D'); a.set_ylabel("B' needed"); a.set_yscale('log'); a.legend(fontsize=7); a.set_title('(b) no direction gives a constant B\'', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAM.png', dpi=130)
