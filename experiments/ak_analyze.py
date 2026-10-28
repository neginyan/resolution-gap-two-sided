"""Stage AK summary: Proposition AK1 (single blur up to the sign-cancellation factor kappa), its losses on the three families, the dotted-row
combination, the lattice factor Psi = min_tau kappa_tau/(1 - tau), and Proposition AJ3 with explicit bounds.
-> results/summary_AK.json, figures/stageAK.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

J = lambda f: json.load(open('results/%s.json' % f))
S = {}
tl = J('ak_run_tl'); lp = J('ak_run_lp') if os.path.exists('results/ak_run_lp.json') else {}
keep = ('kind', 'D', 'z', 'tau', 'tR', 'ln1_tJ', 'kappa', 'step_split', 'step_CS', 'step_DV_jensen', 'step_final', 'Bneeded_AK1', 'identity_c2_rel_L2', 'tweedie_rel_L2', 'KL_jensen', 'KL_final', 'frac_c2', 'frac_S2')
S['twoline'] = {k: {q: v[q] for q in keep} for k, v in tl.items()}; S['linepoint'] = {k: {q: v[q] for q in keep} for k, v in lp.items()}
S['twoline_kappa_range'] = [min(v['kappa'] for v in tl.values()), max(v['kappa'] for v in tl.values())]
S['twoline_Bneeded_range'] = [min(v['Bneeded_AK1'] for v in tl.values()), max(v['Bneeded_AK1'] for v in tl.values())]
if lp:
    S['linepoint_kappa_range'] = [min(v['kappa'] for v in lp.values()), max(v['kappa'] for v in lp.values())]
    S['linepoint_Bneeded_max'] = max(v['Bneeded_AK1'] for v in lp.values())
S['identity_max_rel_L2'] = max([v['identity_c2_rel_L2'] for v in list(tl.values()) + list(lp.values())] + [v['tweedie_rel_L2'] for v in list(tl.values()) + list(lp.values())])
la = J('ak_lattice1d'); S['lattice1d'] = la
S['lattice_Psi_over_ln'] = {a: v['min_kappa_over_1mtau'] / v['rows'][0]['ln1_tJ'] for a, v in la.items() if float(a) >= 0.8}
dt = J('ak_dotted') if os.path.exists('results/ak_dotted.json') else {}
S['dotted'] = {k: {q: v[q] for q in ('D', 'a', 'z', 'Delta_half', 'tR', 'ln1_tJ', 'Psi', 'tR_over_Delta', 'Bneeded_AK1_best')} | {'kappa_tau': {t: w['kappa'] for t, w in v['taus'].items()}} for k, v in dt.items()}
S['aj3'] = J('ak_aj3')
json.dump(S, open('results/summary_AK.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('twoline_kappa_range', 'twoline_Bneeded_range', 'identity_max_rel_L2', 'lattice_Psi_over_ln')}, default=float))
for k, v in S['dotted'].items(): print('dotted', k, v)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
for i, (k, v) in enumerate(la.items()):
    if float(k) < 0.8: continue
    a.semilogy([r['tau'] for r in v['rows']], [r['kappa'] for r in v['rows']], 'o-', ms=3, label='lattice a = %s (ln %.0f)' % (k, v['rows'][0]['ln1_tJ']))
for k, v in S['dotted'].items():
    T = sorted(v['kappa_tau'].items(), key=lambda x: float(x[0])); a.semilogy([float(x) for x, _ in T], [y for _, y in T], 's--', ms=4, label='two lines + dotted row %s' % k)
a.axhline(2.0, color='C3', lw=1, label='two lines + midpoint (kappa = 2.0)'); a.set_xlabel('$\\tau$'); a.set_ylabel('$\\kappa_\\tau = J_{abs}/J$'); a.legend(fontsize=6); a.set_title('(a) sign-cancellation factor', fontsize=9)
a = ax[1]
for fam, col, lab in ((tl, 'C3', 'two lines + midpoint'), (lp, 'C1', 'one line + point')):
    if fam: a.plot([v['D'] for v in fam.values()], [v['Bneeded_AK1'] for v in fam.values()], 'o', color=col, label=lab)
for k, v in S['dotted'].items(): a.plot(v['D'], v['Bneeded_AK1_best'], 's', color='C2', label='dotted row (best tau)' if k == list(S['dotted'])[0] else None)
a.set_yscale('log'); a.set_xlabel('$D$'); a.set_ylabel("$B'$ needed by Prop. AK1"); a.legend(fontsize=7); a.set_title("(b) Prop. AK1: constant $B'$ on the two-line family", fontsize=9)
a = ax[2]; R = S['aj3']['rows']
a.plot([r['D'] for r in R], [r['margin_plus_5lnD_plus_C0'] for r in R], 'o-', color='C0', label='$tR - \\ln(1/tJ) + 5\\ln D + C_0$ (exact)')
a.fill_between([r['D'] for r in R], -4.4, 4.4, color='C0', alpha=0.15, label='proved window $\\pm 4.4$')
a.set_xlabel('$D$ ($z_D = 2\\ln D/D$)'); a.legend(fontsize=7); a.set_title('(c) Prop. AJ3 with explicit bounds', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAK.png', dpi=130)
