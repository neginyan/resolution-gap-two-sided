"""The A1 factor a/den of Lemma AY2-P rounds to 1 in double precision at sigma = 1e-24, so the lemma is checked there through the deficits
1 - a/den (exact point fields: J_ww and W_p stored in results/be_plemmas.json) against the bound deficit J_ww,up/(J_vv,lo + J_ww,up).
-> results/bg_compare.json (key A1_deficit).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
sys.argv = sys.argv[:1]
from be_plemmas import harmonics
from bf_fc import states

if __name__ == '__main__':
    PL = {r['label']: r for r in json.load(open('results/be_plemmas.json'))}
    ST = {s[0]: s for s in states()}; out = {'A1_deficit': {}}
    print('| state | 1 - a/den (actual, point fields) | 1 - A1 (bound) | bound >= actual |\n|---|---|---|---|')
    for k in ('comb optimum 1e-12', 'strip optimum 1e-12', 'comb optimum 1e-24', 'strip optimum 1e-24'):
        r = PL[k]; _, sig, k0, lam, M, eta, amps = ST[k]; H = harmonics(sig, k0, lam, M, eta, amps)
        Jvv = r['W_p'] / sig ** 2 + 1 / H['V']; d_act = r['J_ww'] / (Jvv + r['J_ww'])
        Jvv_lo = 1 / H['V'] + r['AX2P_lo'] / sig ** 2; d_bd = r['AX1P_up'] / (Jvv_lo + r['AX1P_up'])
        out['A1_deficit'][k] = {'actual': d_act, 'bound': d_bd, 'ok': bool(d_bd >= d_act)}
        print('| %s | %.6e | %.6e | %s |' % (k, d_act, d_bd, d_bd >= d_act))
    json.dump(out, open('results/bg_compare.json', 'w'), indent=1)
