"""Stage BB §3 (outlook): the local S-bound at leading order, tested on the two-window and chirped strips of stage BA.

Strip (Lemma AW3, exact for the strip):  k_eff^2 = k0^2 S + sigma^2/(4V),  S = the window's share of the w-information (stage AX: S <= 8 kappa2 u/(8 + u)).
Candidate local form (route 1):  k_eff^2 <~ sum_r k_r^2 s_r + sigma^2/(4V),  s_r = (J_ww(background + window r) - J_ww(background))/J_ww(full state),
the share of window r (each window evaluated alone on the same background, paper E's evaluator); k_eff^2 = Lambda/4 from stage BA's branches.
-> results/bb_local_s.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from ba_tools import gap
from ba_window import two_state, chirp_state, FAST


def jww(sig, bg, win, kb, Lv_sig):
    o = gap(sig, bg, win, kb, Lv_sig, **FAST); return float(o['J_ww'])


if __name__ == '__main__':
    out = []
    for kind in ('two', 'chirp'):
        for sg in ('1e-06', '1e-12', '1e-24'):
            O = json.load(open('results/ba_window_%s_%s.json' % (kind, sg))); sig = float(sg)
            best = min(O['runs'], key=lambda q: q['g_min']); x = np.array(best['x'])
            kb, bg, win, Lv_sig, info = (two_state if kind == 'two' else chirp_state)(x, sig)
            J0 = jww(sig, bg, [(1e-12, 0.0, win[0][2], 1)], kb, Lv_sig); Jf = jww(sig, bg, win, kb, Lv_sig)
            shares = [(jww(sig, bg, [w], kb, Lv_sig) - J0) / Jf for w in win]
            V = (Lv_sig * sig) ** 2 + sig ** 2; env = sig ** 2 / (4 * V)
            rhs = sum((w[3] * kb) ** 2 * sh for w, sh in zip(win, shares)) + env
            keff2 = O['best_branches']['Lambda'] / 4
            # the same for the strip optimum (one window): the exact AW3 form
            out.append({'family': kind, 'sigma': sig, 'k_eff2': keff2, 'rhs_local': rhs, 'ratio': keff2 / rhs, 'shares': shares, 'sum_shares': sum(shares),
                        'wavenumbers': [w[3] * kb for w in win], 'envelope_term': env, 'k_eff2_strip': O['strip_branches']['Lambda'] / 4})
            print('%-5s sigma %-6s k_eff^2 %.4f  local RHS %.4f  ratio %.4f  shares %s (sum %.4f)  k_r %s' % (kind, sg, keff2, rhs, keff2 / rhs,
                  ', '.join('%.4f' % v for v in shares), sum(shares), ', '.join('%.3f' % (w[3] * kb) for w in win)), flush=True)
    from ba_tools import strip_opt
    for sg in ('1e-06', '1e-12', '1e-24'):          # control: the strip itself (AW3 is exact there)
        sig = float(sg); P = strip_opt(sig); bg = [(1.0, 0.0, P['lam'] * P['M'])]; win = [(P['c'], 0.0, P['lam'], 1)]
        J0 = jww(sig, bg, [(1e-12, 0.0, P['lam'], 1)], P['k0'], P['Lv_sig']); Jf = jww(sig, bg, win, P['k0'], P['Lv_sig'])
        sh = (Jf - J0) / Jf; V = (P['Lv_sig'] * sig) ** 2 + sig ** 2; rhs = P['k0'] ** 2 * sh + sig ** 2 / (4 * V)
        keff2 = json.load(open('results/ba_window_two_%s.json' % sg))['strip_branches']['Lambda'] / 4
        out.append({'family': 'strip', 'sigma': sig, 'k_eff2': keff2, 'rhs_local': rhs, 'ratio': keff2 / rhs, 'shares': [sh], 'sum_shares': sh, 'wavenumbers': [P['k0']]})
        print('strip sigma %-6s k_eff^2 %.4f  local RHS %.4f  ratio %.4f  share %.4f' % (sg, keff2, rhs, keff2 / rhs, sh), flush=True)
    json.dump(out, open('results/bb_local_s.json', 'w'), indent=1, default=float)
