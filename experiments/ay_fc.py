"""Stage AY: Lemma AY2 — explicit lower bounds of the factors of F_c on a strip, from its parameters (checked against paper E's evaluator).
F_c = (a/den)(1 - eps)|C_w| sqrt((1 - Phi/4)(1 - delta_w)) >= (1 - eps) * A1 * A2 * A3:
 A1 = 1/(1 + J_ww,up/J_vv,lo),  J_ww,up = 1/L^2 + kq B^2,  J_vv,lo = 1/V + x I_q/(2 sigma^2)                 (Lemmas AX1, AX2)
 A2 = (1 - dH)(1 - rC),  dH = sigma^2/L^2 + (sigma/s)^2 [qb/(1-qb) + qb^2/(e(1-qb)^2)] + sqrt(x)(sigma/s) qb/(sqrt(e)(1-qb)^2),
      rC = [(1-b)^2 (3 + V J_per) + 2(1 + 3 sigma^4/L^4)(L^2/V + <s~^2 w^2>) + 6 Xi J_vv]/(8 J_vv)
 A3 = sqrt(1 - rw),  rw = V/8 + [2(1 + 3 sigma^4/L^4)(3 + kq Bw^2) + 6 Xi J_ww]/(8 J_ww)
 (cos(m/2) >= 1 - m^2/8, phi <= m^2/2, m_z^2 <= v'^2 + R^2, R^2 <= 2w^2 + 2(R-w)^2, Gaussian moments of Gamma, the factorisation of Lemma AW1;
  Xi = xi^2 + sigma^4 qb^2/(e s^2 (1-qb)^2), xi = sigma sqrt(x) qb/(1-qb), kq = (1/sqrt(1-qb^2) - 1)/qb^2, Bw^2 = 3 q0^2 L^4/(s^4 (1+2L^2/s^2)^{5/2})).
-> results/ay_fc.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from aw_lemmas import profile
from aq_hybrid import strip_params

E = np.e


def fc_lower(sig, k0, lam, M, c, eta, J_true=None):
    P = profile(sig, k0, lam, M, c, eta); L2 = P['L2']; s2 = P['s2']; V = P['V']; qb = P['qbar']; q0 = P['q0']; x = P['kp_sigma'] ** 2
    s = np.sqrt(s2); L = np.sqrt(L2); b = sig * sig / V
    kq = (1 / np.sqrt(1 - qb * qb) - 1) / qb ** 2
    B2 = q0 ** 2 * L2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** 1.5); Bw2 = 3 * q0 ** 2 * L2 ** 2 / (s2 ** 2 * (1 + 2 * L2 / s2) ** 2.5)
    Iq = q0 ** 2 * s / np.sqrt(s2 + 2 * L2)
    Jww_up = 1 / L2 + kq * B2; Jww_lo = 1 / L2 + B2 / 2
    Jper_lo = x * Iq / (2 * sig * sig); Jper_up = x * Iq / (sig * sig * (1 + np.sqrt(1 - qb * qb))); Jvv_lo = 1 / V + Jper_lo
    A1 = 1 / (1 + Jww_up / Jvv_lo)
    xi = sig * np.sqrt(x) * qb / (1 - qb); Xi = xi ** 2 + sig ** 4 * qb ** 2 / (E * s2 * (1 - qb) ** 2); l4 = sig ** 4 / L2 ** 2
    dH = sig * sig / L2 + (sig / s) ** 2 * (qb / (1 - qb) + qb * qb / (E * (1 - qb) ** 2)) + np.sqrt(x) * (sig / s) * qb / (np.sqrt(E) * (1 - qb) ** 2)
    st2w2 = (x / sig ** 2) * q0 ** 2 * L2 * (1 + 2 * L2 / s2) ** -1.5 / (1 + np.sqrt(1 - qb * qb))
    # rC: numerator uses upper bounds, denominator J_vv lower bound (each term separately monotone)
    rC = ((1 - b) ** 2 * (3 + V * Jper_up) + 2 * (1 + 3 * l4) * (L2 / V + st2w2)) / (8 * Jvv_lo) + 6 * Xi / 8
    rw = V / 8 + 2 * (1 + 3 * l4) * (3 + kq * Bw2) / (8 * Jww_lo) + 6 * Xi / 8
    A2 = (1 - dH) * (1 - rC); A3 = np.sqrt(max(0.0, 1 - rw))
    return {'A1_a_over_den': A1, 'A2_Cw': A2, 'A3_sqrt': A3, 'F_lower_over_1_minus_eps': A1 * A2 * A3, 'dH': dH, 'rC': rC, 'rw': rw}


if __name__ == '__main__':
    AT = json.load(open('results/at_strips.json')); AV = json.load(open('results/av_strip.json'))['sequence']; out = {}
    st = {}
    for s in (1e-6, 1e-12, 1e-24):
        P = strip_params(s); st['opt_%g' % s] = ((s, P['k0'], P['lam'], P['M'], P['c'], P['Lv_sig'] * np.sqrt(s) / P['lam']), AT['strip_%g' % s])
    for k, v in AV.items(): st['seq_%s' % k] = ((float(k), v['k0'], v['lam'], v['M'], v['c'], v['eta']), v)
    for name, (args, ref) in st.items():
        r = fc_lower(*args); act = {'a_over_den': ref['a_over_den'], 'C_w': ref['C_w'], 'sqrt': np.sqrt((1 - ref['Phi'] / 4) * (1 - ref['delta_w'])), 'eps': ref['eps']}
        Fc = ref['a_over_den'] * (1 - ref['eps']) * abs(ref['C_w']) * act['sqrt']
        r.update({'actual': act, 'F_c_actual': Fc, 'F_lower': (1 - abs(ref['eps'])) * r['F_lower_over_1_minus_eps'],
                  'ok': bool(r['A1_a_over_den'] <= ref['a_over_den'] * (1 + 1e-12) and r['A2_Cw'] <= abs(ref['C_w']) * (1 + 1e-9) and r['A3_sqrt'] <= act['sqrt'] * (1 + 1e-9))})
        out[name] = r
        print('%-10s a/den %.10f >= %.10f | C_w %.8f >= %.8f | sqrt %.8f >= %.8f | F_c %.8f >= %.8f  %s' % (name, ref['a_over_den'], r['A1_a_over_den'], ref['C_w'], r['A2_Cw'],
              act['sqrt'], r['A3_sqrt'], Fc, r['F_lower'], r['ok']), flush=True)
    json.dump(out, open('results/ay_fc.json', 'w'), indent=1, default=float)
