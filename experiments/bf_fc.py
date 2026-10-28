"""Stage BF §1.1: Lemma AY2-P — explicit lower bounds of the factors of F_c on a periodic window, from its parameters, checked against paper E's evaluator.
F_c = (a/den)(1 - eps)|C_w| sqrt((1 - Phi/4)(1 - delta_w)) >= (1 - |eps|) A1 A2 A3 (Lemma AY2 with the P-replacements):
 A1 = 1/(1 + J_ww,up/J_vv,lo),  J_ww,up = (1 + u0/(8(1 - Q)))/L^2 (AX1-P),  J_vv,lo = 1/V + x I2/(2(1 + Q) sigma^2) (AX2-P)
 A2 = (1 - dH)(1 - rC),  dH = sigma^2/L^2 + (sigma/s)^2 [Q/(1-Q) + Q^2/(e(1-Q)^2)] + sqrt(x)(sigma/s) N1/(sqrt(e)(1-Q)^2),
      rC = [(1-b)^2 (3 + V J_per,up) + 2(1 + 3 sigma^4/L^4)(L^2/V + J_per,up v)]/(8 J_vv,lo) + 6 Xi/8,   v = L^2 s^2/(s^2 + 2 L^2)
 A3 = sqrt(1 - rw),  rw = V/8 + 2(1 + 3 sigma^4/L^4)(3 + 3 v^2 sum q0_n^2 (s/sqrt(s^2 + 2L^2))/(2(1 - Q) s^4))/(8 J_ww,lo) + 6 Xi/8,
 Xi = xi^2 + sigma^4 Q^2/(e s^2 (1-Q)^2),  xi = sigma sqrt(x) N1/(1 - Q),  Q = Qbar = sum qbar_n,  N1 = sum n qbar_n.
(<s~^2 w^2> <= J_per,up v and the window part of <s_w^2 w^2> are the Gaussian moments of Gamma e^{-w^2/s^2}: one envelope for all harmonics.)
-> results/bf_fc.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, os
import numpy as np
sys.argv = sys.argv[:1]
import at_tools
from w_tools import terms_of, w_nodes
from be_plemmas import harmonics
from ba_tools import strip_opt

E = np.e


def fc_lower_P(sig, H):
    L2, s2, V, x, Q = H['L2'], H['s2'], H['V'], H['x'], H['Qbar']; q0 = H['q0']; n = np.arange(1, len(q0) + 1)
    s = np.sqrt(s2); b = sig * sig / V; N1 = float(np.sum(n * H['qbar'])); l4 = sig ** 4 / L2 ** 2; geo = s / np.sqrt(s2 + 2 * L2); v = L2 * s2 / (s2 + 2 * L2)
    Jww_up = (1 + H['u0'] / (8 * (1 - Q))) / L2; Jww_lo = (1 + H['u0'] / (8 * (1 + Q))) / L2
    Jper_up = x * H['I2'] / (2 * (1 - Q) * sig * sig); Jper_lo = x * H['I2'] / (2 * (1 + Q) * sig * sig); Jvv_lo = 1 / V + Jper_lo
    A1 = 1 / (1 + Jww_up / Jvv_lo)
    xi = sig * np.sqrt(x) * N1 / (1 - Q); Xi = xi ** 2 + sig ** 4 * Q ** 2 / (E * s2 * (1 - Q) ** 2)
    dH = sig * sig / L2 + (sig / s) ** 2 * (Q / (1 - Q) + Q * Q / (E * (1 - Q) ** 2)) + np.sqrt(x) * (sig / s) * N1 / (np.sqrt(E) * (1 - Q) ** 2)
    rC = ((1 - b) ** 2 * (3 + V * Jper_up) + 2 * (1 + 3 * l4) * (L2 / V + Jper_up * v)) / (8 * Jvv_lo) + 6 * Xi / 8
    win_sw2w2 = 3 * v * v * float(np.sum(q0 ** 2)) * geo / (2 * (1 - Q) * s2 * s2)
    rw = V / 8 + 2 * (1 + 3 * l4) * (3 + win_sw2w2) / (8 * Jww_lo) + 6 * Xi / 8
    A2 = (1 - dH) * (1 - rC); A3 = np.sqrt(max(0.0, 1 - rw))
    return {'A1_a_over_den': A1, 'A2_Cw': A2, 'A3_sqrt': A3, 'F_lower_over_1_minus_eps': A1 * A2 * A3, 'dH': dH, 'rC': rC, 'rw': rw}


def actual(sig, k0, lam, M, eta, amps):
    s = np.sqrt(sig); win = [(a, 0.0, lam, n) for n, a in enumerate(amps, 1)]
    T = terms_of([(1.0, 0.0, lam * M)], win, k0, sig, eta * lam / s); wn = w_nodes([(1.0, 0.0, lam * M)], win, sig, 40, 300)
    En = at_tools.sums_envelope(T, sig, k0, np.sqrt((eta * lam * s) ** 2 + sig ** 2), wn, nv=32 * len(amps), n0=21); return at_tools.analyse(En, sig)


def states():
    out = []
    for sg in ('1e-12', '1e-24'):
        Dd = json.load(open('results/bd_opt_comb_%s.json' % sg)); x = np.exp(Dd['x'][:4]); out.append(('comb optimum %s' % sg, float(sg), Dd['kb'], x[1], x[2], x[3], [w[0] for w in Dd['windows']]))
        P = strip_opt(float(sg)); out.append(('strip optimum %s' % sg, float(sg), P['k0'], P['lam'], P['M'], P['Lv_sig'] * float(sg) / (P['lam'] * np.sqrt(float(sg))), [P['c']]))
    Dd = json.load(open('results/bd_opt_comb_1e-06.json')); x = np.exp(Dd['x'][:4]); out.append(('comb optimum 1e-06', 1e-6, Dd['kb'], x[1], x[2], x[3], [w[0] for w in Dd['windows']]))
    for st, k0 in ((0.3, 2.5), (0.15, 2.0), (0.6, 1.6)):
        amps = [0.999 * 2 * np.exp(-(n * k0 * st) ** 2 / 2) for n in range(1, int(np.ceil(np.sqrt(2 * np.log(1e6)) / (k0 * st))) + 1)]
        out.append(('comb st %.2f k0 %.1f' % (st, k0), 1e-12, k0, 1.5, 500.0, 0.2, amps))
    return out


if __name__ == '__main__':
    np.seterr(all='ignore'); fn = os.environ.get('BF_OUT', 'results/bf_fc_full.json'); res = json.load(open(fn)) if os.path.exists(fn) else {}     # resumable
    for label, sig, k0, lam, M, eta, amps in states():
        if label in res: continue
        H = harmonics(sig, k0, lam, M, eta, amps); lo = fc_lower_P(sig, H); r = actual(sig, k0, lam, M, eta, amps)
        act = {'a_over_den': float(r['a_over_den']), 'C_w': float(abs(r['C_w'])), 'sqrt': float(np.sqrt((1 - r['Phi'] / 4) * (1 - r['delta_w']))), 'eps': float(r['eps'])}
        Fc = act['a_over_den'] * (1 - act['eps']) * act['C_w'] * act['sqrt']
        lo.update({'label': label, 'sigma': sig, 'x': H['x'], 'Qbar': H['Qbar'], 'actual': act, 'F_c_actual': Fc, 'F_lower': (1 - abs(act['eps'])) * lo['F_lower_over_1_minus_eps'],
                   'ok': bool(lo['A1_a_over_den'] <= act['a_over_den'] * (1 + 1e-12) and lo['A2_Cw'] <= act['C_w'] * (1 + 1e-9) and lo['A3_sqrt'] <= act['sqrt'] * (1 + 1e-9))})
        res[label] = lo; json.dump(res, open(fn, 'w'), indent=1, default=float)
        print('%-22s x %.2f Q %.3f | a/den %.10f >= %.10f | C_w %.8f >= %.8f | sqrt %.8f >= %.8f | F_c %.8f >= %.8f  %s' % (label, H['x'], H['Qbar'], act['a_over_den'],
              lo['A1_a_over_den'], act['C_w'], lo['A2_Cw'], act['sqrt'], lo['A3_sqrt'], Fc, lo['F_lower'], lo['ok']), flush=True)
    json.dump(res, open(fn, 'w'), indent=1, default=float)
