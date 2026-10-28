"""Stage BG §3 (outlook, not for paper F): can assumption (P8) F_c >= 1.02G be removed?  A light closed-form scan.

Where F_c is small, Lemma AY2-P says which window parameter is responsible: A1 (J_ww comparable to J_vv), or r_C, r_w, delta_H (V, L^2 or sigma/s not small).
Two lower bounds of g need no F_c at all (branch identity + the P-lemmas, every periodic window):
  (W) w-branch:  g >= (1 - |eps|) (a) >= (1 - |eps|) sigma^2 J_ww,lo/(sigma den_up),     den_up = b + x I2/(2(1 - Q)) + sigma^2 J_ww,up
  (N) N-branch:  g >= (1 - |eps|) (a') >= (1 - |eps|) N_lo/(sigma den_up),  N_lo = (1/2)[(1 - e - sigma^2) W_p,lo + (1 - C)_lo (b - sigma^2 gamma_P)]_+
                 (AX3-P), (1 - C)_lo = (1 - e^{-L^2/4}) - (L r + r^2/2)/2, r = xi_P + sigma^2 sqrt(J_ww,up), e = e^{-alpha^2 V/2}.
Scan: random periodic windows (strips and Gaussian-tooth combs) inside (P1)-(P5) at sigma = 1e-12, 1e-24; for every state with F_lower < 1.02 G, is
max(W, N) >= G?  -> results/bg_p8_scan.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
sys.argv = sys.argv[:1]
from be_plemmas import harmonics
from bf_fc import fc_lower_P

E = np.e


def direct(sig, H):
    L2, s2, V, x, Q = H['L2'], H['s2'], H['V'], H['x'], H['Qbar']; qb = H['qbar']; n = np.arange(1, len(qb) + 1)
    N1, N2 = float(np.sum(n * qb)), float(np.sum(n * n * qb)); b = sig * sig / V; s = np.sqrt(s2); L = np.sqrt(L2)
    Jww_lo = (1 + H['u0'] / (8 * (1 + Q))) / L2; Jww_up = (1 + H['u0'] / (8 * (1 - Q))) / L2
    Wp_lo = x * H['I2'] / (2 * (1 + Q)); Wp_up = x * H['I2'] / (2 * (1 - Q)); den_up = b + Wp_up + sig * sig * Jww_up
    W = sig * sig * Jww_lo / (sig * den_up)
    alpha = (1 - b) / np.sqrt(2); e = np.exp(-alpha ** 2 * V / 2)
    xi = sig * np.sqrt(x) * N1 / (1 - Q); r = xi + sig * sig * np.sqrt(Jww_up)
    omC = max(0.0, -np.expm1(-L2 / 4) - (L * r + r * r / 2) / 2)
    gam = (x * (N2 * (1 + Q) + N1 * N1) + np.sqrt(x) * N1 * (1 + 2 * Q) / np.sqrt(E)) / (1 - Q) ** 2
    Nlo = 0.5 * max(0.0, (-np.expm1(-alpha ** 2 * V / 2) - sig * sig) * Wp_lo + omC * (b - sig * sig * gam))
    return W, Nlo / (sig * den_up)


if __name__ == '__main__':
    np.seterr(all='ignore'); rng = np.random.default_rng(7); out = {}
    for sig in (1e-12, 1e-24):
        G = min(0.149 * 0.99, 1.2 / np.sqrt(np.log(1 / sig))); rows = []; tried = 0
        while len(rows) < 4000 and tried < 400000:
            tried += 1
            k0 = np.exp(rng.uniform(0, np.log(10))); lam = np.exp(rng.uniform(np.log(0.1), np.log(100))); M = np.exp(rng.uniform(0, np.log(1e10)))
            eta = np.exp(rng.uniform(np.log(1e-4), np.log(1e4)))
            if rng.random() < 0.5: amps = [rng.uniform(0.05, 1.0)]
            else:
                tau = rng.uniform(0.02, 1.0); amps = [2 * 0.999 * np.exp(-(m * tau) ** 2 / 2) for m in range(1, 60)]
            try: H = harmonics(sig, k0, lam, M, eta, amps)
            except Exception: continue
            Lw = M * lam * np.sqrt(sig)
            if not (np.isfinite(H['s2']) and H['s2'] > 0 and H['x'] >= 1 and H['Qbar'] <= 0.25 and H['rho'] <= 0.125 and Lw <= sig ** -0.5): continue
            F = 0.99 * fc_lower_P(sig, H)['F_lower_over_1_minus_eps']; W, N = direct(sig, H)
            rows.append((F, 0.99 * W, 0.99 * N, H['V'], H['L2'], np.sqrt(sig * sig / H['s2'])))
        R = np.array(rows); small = R[(~np.isfinite(R[:, 0])) | (R[:, 0] < 1.02 * G)]
        rescued = small[np.maximum(small[:, 1], small[:, 2]) >= G]
        worst = small[np.argmin(np.maximum(small[:, 1], small[:, 2]))] if len(small) else None
        out['%g' % sig] = {'G': G, 'states_in_class': len(R), 'F_lower_below_1.02G': int(len(small)), 'rescued_by_W_or_N': int(len(rescued)),
                           'min_max_WN_over_G_among_small': float(np.min(np.maximum(small[:, 1], small[:, 2])) / G) if len(small) else None,
                           'worst_state_V_L2_sigma_over_s': None if worst is None else [float(worst[3]), float(worst[4]), float(worst[5])],
                           'median_F_lower_all': float(np.nanmedian(R[:, 0]))}
        print(sig, out['%g' % sig], flush=True)
    json.dump(out, open('results/bg_p8_scan.json', 'w'), indent=1)
