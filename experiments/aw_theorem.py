"""Stage AW: Theorem AW5 evaluated on the strip states (optima of paper E, the stage-AV sequence) and its corollaries.
k_eff^2 <= B(L/s) + 1/4 <= ln(L/s) + 1.428 + 1/4  (qbar <= 1/4, rho <= 1/8, L/s >= 10; aw_constants.py), hence g >= Fc/(2 sqrt(...)), Fc the state factor of AT2.
-> results/aw_theorem.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aw_lemmas import profile
from aw_constants import tables, B
from aq_hybrid import strip_params

if __name__ == '__main__':
    C = json.load(open('results/aw_constants.json'))['q0.25_rho0.125']['sup_excess_all']
    xs, q, k1, k2 = tables(0.25, 0.125)
    AT = json.load(open('results/at_strips.json')); AV = json.load(open('results/av_strip.json'))['sequence']; out = {'C': C, 'states': {}}
    st = {}
    for s in (1e-6, 1e-12, 1e-24):
        P = strip_params(s); st['opt_%g' % s] = (s, P['k0'], P['lam'], P['M'], P['c'], P['Lv_sig'] * np.sqrt(s) / P['lam'], AT['strip_%g' % s])
    for k, v in AV.items(): st['seq_%s' % k] = (float(k), v['k0'], v['lam'], v['M'], v['c'], v['eta'], v)
    for name, (s, k0, lam, M, c, eta, ref) in st.items():
        P = profile(s, k0, lam, M, c, eta); Fc = ref['bound_AT2'] * 2 * ref['k_eff']
        r = {'sigma': s, 'qbar': P['qbar'], 'rho': P['rho'], 'L_over_s': P['L_over_s'], 'k_eff2': ref['k_eff'] ** 2, 'g': ref['g'], 'Fc': Fc, 'covered': bool(P['qbar'] <= 0.25 and P['rho'] <= 0.125 and P['L_over_s'] >= 10)}
        if r['covered']:
            r['B'] = B(P['L_over_s'], xs, q, k1, k2) + 0.25; r['simple'] = np.log(P['L_over_s']) + C + 0.25
            r['g_bound'] = Fc / (2 * np.sqrt(r['B'])); r['g_bound_simple'] = Fc / (2 * np.sqrt(r['simple']))
            r['ln_Ls_over_ln_sigma'] = np.log(P['L_over_s']) / np.log(1 / s)
        out['states'][name] = r
        print('%-14s qbar %.3g  L/s %.3g (ln(L/s)/ln(1/s) %s)  k_eff^2 %.3f <= %s  | g %.5f >= %s  (Fc %.6f)' % (name, P['qbar'], P['L_over_s'], '%.3f' % r['ln_Ls_over_ln_sigma'] if r['covered'] else '-',
              r['k_eff2'], '%.3f' % r['B'] if r['covered'] else 'not covered (qbar > 1/4)', r['g'], '%.5f' % r['g_bound'] if r['covered'] else '-', Fc), flush=True)
    json.dump(out, open('results/aw_theorem.json', 'w'), indent=1, default=float)
