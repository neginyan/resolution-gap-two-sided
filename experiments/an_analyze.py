"""Stage AN summary: signed v-problem (Prop. AN3), theta remainder and Theorem AN5, general signed 1-d problem (Prop. AN1),
hidden multipoles (the slope of (S1), (T2) and (T1) is unbounded on infinite-line periodic states).
-> results/summary_AN.json, figures/stageAN.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

V = json.load(open('results/an_vproblem.json')); T = json.load(open('results/an_theta.json')); S = json.load(open('results/an_signed1d.json'))
MP = json.load(open('results/an_multipole.json')); MK = json.load(open('results/an_markov.json')); LF = json.load(open('results/an_lift.json'))
TL = json.load(open('results/an_tilted.json')); L2 = json.load(open('results/an_lambda2d.json'))
search = {}
for fn in sorted(glob.glob('results/an_signed1d_search_*.json')):
    v = json.load(open(fn))
    if v is not None: search[fn.split('search_')[1][:-5]] = v
retune = {fn.split('retune_')[1][:-5]: json.load(open(fn)) for fn in sorted(glob.glob('results/an_retune_*.json'))}
fam = V['family']; Ds = sorted(float(k) for k in fam)
S_ = {'identities': V['identities'],
      'family': {k: {q: fam[k][q] for q in ('beta', 'Rq_g', 'Rq_g_bd', 'lnMI_g', 'lnMI_g_bd', 'B_g', 'Rq_h', 'lnMI_h', 'B_h', 'split_loss', 'margin_g', 'ok_bounds')} for k in fam},
      'scan': {k: V['scan'][k] for k in ('n', 'max_margin_g', 'max_margin_h', 'max_ratio_g_ln_gt_5', 'max_ratio_h_ln_gt_5')},
      'proved_range': {k: V['proved_range'][k] for k in ('n', 'all_ok', 'max_B_g', 'max_B_h')},
      'theta_cases': T['cases'], 'theta_large_D': {k: {q: v[q] for q in ('Bprime_AN5', 'Aprime_AN5', 'lead_tR', 'lead_ln1_tJ')} for k, v in T['large_D'].items()},
      'AN1': {'max_rel_dI': max(r['dI_plus_D_rel'] for r in S['identities']), 'max_rel_Rq_prime': max(r['rel'] for r in S['identities']),
              'min_H': min(r['H_min'] for r in S['identities']), 'logconcave_min_slack': S['logconcave']['min_slack'], 'sharpness_sin': S['sharpness_sin']},
      'search': {k: {q: v[q] for q in ('Rq', 'lnMI', 'K')} for k, v in search.items()},
      'search_max_margin': max(v['Rq'] - v['lnMI'] for v in search.values()), 'search_max_ratio': max(v['Rq'] / v['lnMI'] for v in search.values() if v['lnMI'] > 5),
      'retune': {k: {q: v[q] for q in ('lam', 'start_margin', 'margin', 'Rq', 'lnMI')} for k, v in retune.items()},
      'tilted': TL, 'multipole_leading': {n: {'alpha': v['alpha'], 'B_n': v['B_n']} for n, v in MP['leading'].items()}, 'multipole_exact': MP['exact'],
      'markov': MK, 'lift': {n: {q: v[q] for q in ('n', 'D', 'a', 'tR_over_ln_min', 'Lam_over_ln', 'B_n_leading')} | {'even_weight': v['bracket']['even_weight'], 'tR_bracket': v['bracket']['tR'], 'ln_bracket': v['bracket']['ln1_tJ']} for n, v in LF.items()},
      'lambda2d': L2}
json.dump(S_, open('results/summary_AN.json', 'w'), indent=1, default=float)
print('search max margin %.2f max ratio %.4f' % (S_['search_max_margin'], S_['search_max_ratio']))
print('lift tR/ln >=', {n: round(v['tR_over_ln_min'], 3) for n, v in S_['lift'].items()}, ' Lambda/ln', {n: round(v['Lam_over_ln'], 3) for n, v in S_['lift'].items()})

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4)); ax = ax.ravel()
a = ax[0]
a.plot(Ds, [fam['%g' % D]['Rq_g'] for D in Ds], 'o-', color='C3', label='Rq(g) exact (signed v-profile)')
a.plot(Ds, [fam['%g' % D]['lnMI_g'] for D in Ds], 's-', color='C0', label='ln(M/I_g) exact')
a.plot(Ds, [fam['%g' % D]['Rq_g_bd'] for D in Ds], 'v--', color='C3', alpha=0.6, label='Rq(g): proved upper bound (Prop. AN3)')
a.plot(Ds, np.array(Ds) ** 2 / 2, ':', color='k', label='$D^2/2$')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('D'); a.legend(fontsize=7); a.set_title('(a) staggered rows, v-problem: lever arm = valley depth $D^2/2$', fontsize=9)
a = ax[1]
a.plot(Ds, [fam['%g' % D]['B_g'] for D in Ds], 'o-', color='C3', label='$B_g$ (signed), proved')
a.plot(Ds, [fam['%g' % D]['B_h'] for D in Ds], 's-', color='C2', label='$B_h$ (unsigned), proved')
for aa, c in ((2.0, 'C4'), (1.5, 'C1'), (1.2, 'C5')):
    ks = sorted([k for k in T['large_D'] if k.endswith('_%g' % aa)], key=lambda k: float(k.split('_')[0]))
    a.plot([float(k.split('_')[0]) for k in ks], [T['large_D'][k]['Bprime_AN5'] for k in ks], '^--', color=c, ms=4, label="B' of Thm AN5, a = %g" % aa)
a.axhline(1, color='k', lw=0.6); a.set_xscale('log'); a.set_xlabel('D'); a.legend(fontsize=6.5); a.set_title("(b) staggered rows: proved slopes tend to 1 (Theorem AN5)", fontsize=9)
a = ax[2]
ns = sorted(int(n) for n in MK); a.plot(ns, [MK[str(n)]['B_n'] for n in ns], 'o-', color='C3', label='$B_n$: lim tR/ln(1/tJ), multipole of order n (leading order, untilted)')
a.plot(ns, [MK[str(n)]['Lambda_over_ln_n'] for n in ns], 's-', color='C0', label='lim $\\Lambda$/ln(1/tJ) (same states)')
lf = sorted(LF.values(), key=lambda v: v['n'])
a.plot([v['n'] for v in lf], [v['tR_over_ln_min'] for v in lf], '*', ms=10, color='C3', label='exact 2-d lift, proved lower bracket (Lemma AN4)')
a.plot([v['n'] for v in lf], [v['Lam_over_ln'] for v in lf], '*', ms=10, color='C0', label='exact 2-d lift, $\\Lambda$/ln')
a.axhline(1, color='k', lw=0.6); a.axhline(4, color='gray', ls='--', lw=0.8); a.text(1.05, 4.3, "paper E, Z2-F': B = 4 claimed sharp", fontsize=7, color='gray')
a.set_xscale('log'); a.set_xlabel('order n of the hidden multipole'); a.legend(fontsize=6.5); a.set_title("(c) no finite slope: hidden multipoles (infinite w-lines)", fontsize=9)
a = ax[3]
lc = S['logconcave']['rows']; a.scatter([r['lnMI'] for r in lc], [r['tRq'] for r in lc], s=10, color='C0', label='log-concave $\\nu$ (Prop. AN1: below the line)')
x = np.linspace(0, max(r['lnMI'] for r in lc) * 1.05, 10); a.plot(x, x, 'k-', lw=0.8, label="$tRq = \\ln(M/I)$  (B' = 1, A' = 0)")
a.set_xlabel('ln(M/I)'); a.set_ylabel('t Rq'); a.legend(fontsize=7); a.set_title('(d) signed 1-d problem: log-concave case is proved', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageAN.png', dpi=130)
