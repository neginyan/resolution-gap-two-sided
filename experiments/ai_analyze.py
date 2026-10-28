"""Stage AI summary: static targets (T1) Lambda <= A + B ln(1/tJ_ww), (T2) tR <= A' + B' ln(1/tJ_ww).
Collects results/ai_*.json into results/summary_AI.json and draws figures/stageAI.png.  Python 3.11."""
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
from ai_margins import stored
_S = stored(); st = {}
_cls = {'strip': lambda r: 'strip' in r['src'], 'lattice': lambda r: ('lattice' in r['src']) or r['src'].startswith('AD-ladder'),
        'points': lambda r: r['src'].startswith('AC:points'), 'mix': lambda r: r['src'].startswith('AC:mix'), 'all': lambda r: True}
for c, f in _cls.items():
    R = [r for r in _S if f(r)]; T2 = max(R, key=lambda r: r['tR'] - r['ln']); RL = [r for r in R if r['Lambda'] is not None]; T1 = max(RL, key=lambda r: r['Lambda'] - 4 * r['ln'])
    st[c] = {'n': len(R), 'T2_margin(tR-ln)': T2['tR'] - T2['ln'], 'T2_at': [T2['src'], T2['m'], T2['tR'], T2['ln']],
             'T1_margin(L-4ln)': T1['Lambda'] - 4 * T1['ln'], 'T1_at': [T1['src'], T1['m']],
             'max_tR/ln(ln>3)': max([r['tR'] / r['ln'] for r in R if r['ln'] > 3] or [float('nan')]), 'max_L/ln(ln>3)': max([r['Lambda'] / r['ln'] for r in RL if r['ln'] > 3] or [float('nan')])}
S['stored_classes'] = st
asy = J('ai_asym')
S['line_point_asym'] = {k: {q: v[q] for q in ('D', 'z', 'tR', 'Lambda', 'ln1_tJ', 'tR/ln', 'Lambda/ln')} for k, v in asy.items()}
S['line_point_sup_tR_over_ln'] = max(v['tR/ln'] for v in asy.values())
S['line_point_sup_Lambda_over_ln'] = max(v['Lambda/ln'] for v in asy.values())
S['line_point_max_T2_margin'] = max(v['tR'] - v['ln1_tJ'] for v in asy.values())
S['line_point_plateau_tR_minus_D2'] = [asy['%g_1.5' % D]['tR'] - D * D for D in (8, 16, 24, 32, 48)]
va = J('ai_valley'); S['valley_slopes'] = {k: v['slopes_D>=8'] for k, v in va.items()}
br = J('ai_break'); S['break_max_tR_over_ln'] = max((v['tR/ln'], k) for k, v in br.items()); S['break_max_T2_margin'] = max((v['T2_margin'], k) for k, v in br.items())
S['break_max_Lambda_over_ln'] = max((v['Lambda/ln'], k) for k, v in br.items()); S['break_n'] = len(br)
tl = {k: v for k, v in J('ai_twoline').items() if k not in ('scan_D26', 'scan_D32')}   # D >= 26 under-resolved at h = 0.04 (valley width 1/D): use the fine exact evaluator
S['twoline'] = tl; S['twoline_note'] = 'mixlog scans at D = 26, 32 (h = 0.04) under-resolve the valley; replaced by ai_twoline_exact (fine grid)'
tx = J('ai_twoline_exact_large'); S['twoline_exact_large'] = tx
S['twoline_sup_tR_over_ln'] = max([v['tR/ln'] for k, v in tl.items() if k in ('scan_D8', 'scan_D14', 'scan_D20', 'nm_D14', 'nm_D20')] + [v['tR/ln'] for v in tx.values()])
S['twoline_max_T2_margin'] = max([v['T2_margin'] for v in tl.values()] + [v['T2_margin'] for v in tx.values()])
mg = J('ai_margins'); S['nm_margins'] = {k: {q: v[q] for q in ('margin', 'tR', 'Lambda', 'ln1_tJ')} for k, v in mg.items() if isinstance(v, dict) and 'margin' in v}
# cross-checks (recomputed here, cheap)
from ai_mixlog import targets
from ai_lp_exact import evaluate as lpx
from ai_twoline_exact import evaluate as tlx
a = targets([(0.0, 0.0)], [(10.0, 0.0, 10.0 * (10.0 + 5.0))], h=0.04); b = lpx(10.0, 10.0)
c = targets([(0.0, 0.0), (16.0, 0.0)], [(8.0, 0.0, 8.0 * 4.0)], h=0.04); d = tlx(8.0, 0.0)
S['crosscheck'] = {'line_point_D10_z10': {'mixlog': a, 'exact': b, 'max_rel': max(abs(a[q] / b[q] - 1) for q in ('tR', 'Lambda', 'ln1_tJ'))},
                   'twoline_D8_z0': {'mixlog': c, 'exact': d, 'max_rel': max(abs(c[q] / d[q] - 1) for q in ('tR', 'Lambda', 'ln1_tJ'))}}
from ai_fdcheck import run
S['fd_check'] = {'one_line_D8': run(8.0, 0.0, False), 'two_line_D8': run(8.0, 0.0, True)}
json.dump(S, open('results/summary_AI.json', 'w'), indent=1, default=float)
print(json.dumps({k: S[k] for k in ('line_point_sup_tR_over_ln', 'line_point_sup_Lambda_over_ln', 'line_point_max_T2_margin', 'line_point_plateau_tR_minus_D2', 'break_max_tR_over_ln', 'break_max_T2_margin', 'twoline_sup_tR_over_ln', 'twoline_max_T2_margin', 'crosscheck', 'fd_check')}, default=float, indent=0)[:3000])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for D in (8, 16, 24, 32, 48):
    al = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]; ax[0].plot(al, [asy['%g_%g' % (D, x)]['tR/ln'] for x in al], 'o-', ms=3, label='D = %d' % D)
ax[0].axhline(0.5, color='k', ls=':', lw=1); ax[0].set_xlabel(r'$z/D$'); ax[0].set_ylabel(r'$tR\,/\,\ln(1/tJ_{ww})$'); ax[0].legend(fontsize=8)
ax[0].set_title('(a) one $w$-line + one point: $\\sup \\to 1/2$', fontsize=10)
Ds = [8, 14, 20]; r_ = [tl['scan_D%d' % D]['tR/ln'] for D in Ds]; m_ = [tl['scan_D%d' % D]['T2_margin'] for D in Ds]
for k, v in sorted(tx.items(), key=lambda kv: float(kv[0].split('_')[0])):
    Ds.append(float(k.split('_')[0])); r_.append(v['tR/ln']); m_.append(v['T2_margin'])
ax[1].plot(Ds, r_, 'o-', color='C3', label=r'$tR/\ln(1/tJ)$'); ax[1].axhline(1, color='k', ls=':', lw=1); ax[1].set_xlabel('$D$'); ax[1].set_ylim(0.7, 1.05)
a2 = ax[1].twinx(); a2.plot(Ds, m_, 's--', color='C0', label=r"$tR-\ln(1/tJ)$"); a2.set_ylabel(r"$tR-\ln(1/tJ)$ (best $z$)", color='C0')
ax[1].set_title('(b) two $w$-lines + midpoint: ratio $\\to 1$, margin $<0$', fontsize=10); ax[1].legend(loc='lower left', fontsize=8)
sc = np.array(J('ai_margins')['stored_scatter'], dtype=float); ok = np.isfinite(sc[:, 0]) & (sc[:, 0] > 0)
ax[2].loglog(sc[ok, 0], np.maximum(sc[ok, 1], 1e-3), '.', ms=2, color='0.5', label='stored states (AC, AD, AF)')
ax[2].loglog([v['ln1_tJ'] for v in asy.values()], [v['tR'] for v in asy.values()], '^', ms=4, color='C1', label='line + point')
ax[2].loglog([v['ln1_tJ'] for v in br.values()], [v['tR'] for v in br.values()], 'x', ms=4, color='C2', label='more lines / points')
ax[2].loglog(Ds, [1] * 0 + [np.nan] * len(Ds), alpha=0)
ax[2].loglog([v['ln1_tJ'] for v in list(tl.values()) + list(tx.values())], [v['tR'] for v in list(tl.values()) + list(tx.values())], 'o', ms=4, color='C3', label='two lines + midpoint')
xx = np.logspace(-1, 4, 50); ax[2].loglog(xx, xx + 6.6, 'k-', lw=1, label=r"$tR = \ln(1/tJ) + 6.6$")
ax[2].set_xlabel(r'$\ln(1/tJ_{ww})$'); ax[2].set_ylabel('$tR$'); ax[2].legend(fontsize=7); ax[2].set_title('(c) all states vs (T2), $B\'=1$', fontsize=10)
plt.tight_layout(); os.makedirs('figures', exist_ok=True); plt.savefig('figures/stageAI.png', dpi=130)
