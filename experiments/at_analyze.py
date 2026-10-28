"""Stage AT: summary of Theorem AT2 / Corollary AT4 over all evaluated states, and the figure.  -> results/summary_AT.json, figures/stageAT.png.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def direct_bound(r):
    """Corollary AT4, second branch: g >= [2 s2 E[s_w^2 (1 - phi/4)] - |eps| sigma a sqrt(P_w)] / (sigma den)  (drop (s2/2) E[s_v^2 phi] >= 0)."""
    s = r['sigma']; den = r['sig2Jvv'] + r['sig2Jww']
    return (2 * r['sig2Jww'] * (1 - r['Phi'] / 4) * (1 - r['delta_w']) - abs(r['eps']) * s * r['sig2Jvv'] * np.sqrt(r['P_w'])) / (s * den)


if __name__ == '__main__':
    R = json.load(open('results/at_families.json')); R.update(json.load(open('results/at_strips.json')))
    out = {}
    for k, r in R.items():
        d = direct_bound(r); c = max(r['bound_AT2'], d); L = np.log(1 / r['sigma'])
        out[k] = {'family': r['family'], 'sigma': r['sigma'], 'g': r['g'], 'bound_ThmW': r['bound_ThmW'], 'bound_AT2': r['bound_AT2'], 'bound_direct': d,
                  'bound_AT4': c, 'AT4_sqrt_ln': c * np.sqrt(L), 'g_sqrt_ln': r['g'] * np.sqrt(L), 'C_w': r['C_w'], 'k_eff': r['k_eff'],
                  'keff2_over_ln': r['k_eff'] ** 2 / L, 'heisenberg_ratio': r['heisenberg_ratio'], 'AT2_over_ThmW': r['AT2_over_ThmW']}
    S = {'states': out, 'n': len(out), 'min_AT4_sqrt_ln': min(v['AT4_sqrt_ln'] for v in out.values()),
         'argmin_AT4_sqrt_ln': min(out, key=lambda k: out[k]['AT4_sqrt_ln']), 'max_keff2_over_ln': max(v['keff2_over_ln'] for v in out.values()),
         'min_AT2_sqrt_ln_without_pairs': min(v['bound_AT2'] * np.sqrt(np.log(1 / v['sigma'])) for k, v in out.items() if not k.startswith('pair')),
         'min_heisenberg_ratio': min(v['heisenberg_ratio'] for v in out.values())}
    json.dump(S, open('results/summary_AT.json', 'w'), indent=1, default=float)
    print('n %d  min AT4 sqrt(ln) %.4f at %s  max k_eff^2/ln %.4f  min Heisenberg ratio %.6f' % (S['n'], S['min_AT4_sqrt_ln'], S['argmin_AT4_sqrt_ln'], S['max_keff2_over_ln'], S['min_heisenberg_ratio']))
    fam = {}
    for k, v in out.items(): fam.setdefault(v['family'], []).append(v)
    LO = json.load(open('results/as_strip_lo.json'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (f, vs) in enumerate(sorted(fam.items())):
        x = [np.log(1 / v['sigma']) for v in vs]
        ax[0].plot(x, [v['bound_AT4'] / v['g'] for v in vs], 'o', ms=5, color='C%d' % i, label=f)
        ax[1].plot(x, [v['AT4_sqrt_ln'] for v in vs], 'o', ms=5, color='C%d' % i, label=f)
    lv = [float(k.replace('1e-', '')) * np.log(10) for k in LO]; ax[1].plot(lv, [LO[k]['g_sqrt_ln'] for k in LO], '-', color='0.4', label='strip g_min (AS, leading order)')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$\ln(1/\sigma)$'); ax[0].set_ylabel('bound (Cor. AT4) / true g'); ax[0].set_title('how much of the gap the proved bound sees')
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].axhline(1, color='0.6', lw=0.8); ax[1].set_xlabel(r'$\ln(1/\sigma)$'); ax[1].set_ylabel(r'bound $\times\sqrt{\ln(1/\sigma)}$')
    ax[1].set_title('proved bound times sqrt(ln 1/sigma)'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figures/stageAT.png', dpi=140); print('figure written')
