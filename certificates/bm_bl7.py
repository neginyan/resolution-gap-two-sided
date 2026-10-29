"""Stage BM §1: Lemma BL7 (Lemma AZ1-P in theta_tau form) — the size of its corrections along the comb sequence S'_C, in outward-rounded
interval arithmetic (computer-assisted proof of the last step of the upper side in v1.1), plus the numerical check of the BL7 inequality.

Lemma BL7 (proof in App. D of the paper, paper/sections/D_theta.tex).  With the notation of Lemma BL3 and z = Qbar nu_0/2,
  2 N_phi <= (1+nu_0)/(1-z) * sigma^2 Esv2phi_P + [ (1-e) b tau Qbar nu_2/2 + b Qbar (nu_0 + tau nu_2) Phi/2 + sigma b E_R ]/(1-z),
  where sigma^2 Esv2phi_P = (1-e) W_per,up + sigma^2 <s~^2 R^2>/4 + b(1-e) + b Phi + sigma^2 alpha^2 e + sigma^2 gamma_P Phi is Lemma AZ1-P's bound;
  den >= den_lo kappa_d, kappa_d = (1 - nu_0 - tau Qbar nu_2/2)/(1+z);  J_ww <= (1+e1) J_ww,up.  Hence
  U_P^theta <= (1 + eps_U) U_P + A_U,   eps_U = (1+nu_0)(1+e1)/((1-z) kappa_d) - 1,   A_U = [ (1.5 tau Qbar nu_2 + Qbar nu_0)/sigma + E_R ]/(2 kappa_d)
  (b <= den_lo, 1 - e <= 1, Phi <= 2), and h^theta = U_P^theta sqrt(t) - 1 - 6.0 ln t/t <= h + 1.5 eps_U + sqrt(t) A_U  (U_P sqrt t <= 1.5 on the range).
Along S'_C (sigma = 256 k^-11 e^{-4k^2}, Lv^2 = 4 sigma k^-5):  tau = k'^2 V = 4/(k^3 sigma (1 + sigma k^5/4)), increasing in k
  (d/dk ln tau = 8/k + 8k - d/dk ln(1 + sigma k^5/4) > 0), sigma tau <= 4/k^3 <= 0.0992, 1/sigma <= tau k^3/2 <= 6.92e6 tau, x >= 11.76, Qbar <= 0.0056.
  nu_k <= 2 (1 + 1e-100) e^{-tau/2 + sigma tau} for k <= 4 (tau >= 1e22), E_R <= K_E tau^3 nu (K_E from the suprema of N_1, gamma_P, c_chi over x >= 1).
  Every correction is <= C_tot tau^5 e^{-tau/2 + 0.1}, decreasing for tau > 10, evaluated at tau_min = tau(k_lo = 3.43).
-> results/bm_bl7.json.  Python 3.11."""
import json
from decimal import Decimal as D
import numpy as np
from ba_ivl import I
import bl_theta as T

K_LO, K_HI = D('3.43'), D('240.0')


def ln_tau(k):
    k = I(k); ln256 = I(256).ln()
    lnsig = ln256 - 11 * k.ln() - 4 * k.sq()
    sig = lnsig.exp()
    corr = (1 + sig * k.sq().sq() * k / 4).ln()
    return I(4).ln() - 3 * k.ln() - lnsig - corr, lnsig


def comb_corrections():
    lt, lsig = ln_tau(K_LO); lt_hi, _ = ln_tau(K_HI)
    tau_lo = lt.exp()                                                         # tau_min enclosure
    st = I(4) / (I(K_LO) * I(K_LO) * I(K_LO))                                                 # sigma tau <= 4/k^3
    tab = T.eps_table(80.0)                                                   # suprema over x >= 1 of N1, gamma_P, c_chi
    N1, gam, cc = tab['sup_N1'], tab['sup_gamma_P'], tab['sup_c_chi']
    KE = 0.04 + 0.04 + 0.95 * N1 ** 2 + 0.1 + 1.41 * cc + 1.25 * gam + N1 * cc + 0.9 * gam + 0.16 * cc + 0.13
    KE = D(repr(float(KE))) * D('1.000001')
    # ln nu <= ln(2.0001) - tau/2 + sigma tau ;  every correction <= C_tot tau^5 nu:
    #   eps_U <= 8 (1 + tau) nu (nu_0, e1, z, kappa_d each <= 2(1 + tau) nu),  1.5 eps_U <= 12 (1 + tau) nu <= 24 tau nu;
    #   sqrt(t) A_U <= 481 * [ (0.375 tau + 0.25) nu * 6.92e6 tau + KE tau^3 nu ] / (2 * 0.99)  <= 481 (2.6e6 + KE) tau^3 nu   (tau >= 1e22)
    Ctot = D(24) + D(481) * (D('2.7e6') + KE)
    ln_nu = I(D('2.0001')).ln() - tau_lo / 2 + st
    ln_bound = I(Ctot).ln() + 5 * lt + ln_nu                                  # tau^5 >= tau^3, tau^1 (tau >= 1)
    log10_bound = ln_bound / I(10).ln()
    margin = D('1.3503519513771814e-4')                                        # bf_cert_upper.json: largest upper bound of h is -1.35e-4
    ok = bool(ln_bound.hi < I(margin).ln().lo)
    return {'k_lo': float(K_LO), 'ln_tau_min': [float(lt.lo), float(lt.hi)], 'tau_min': float(tau_lo.lo), 'log10_tau_at_k_hi': float((lt_hi / I(10).ln()).lo),
            'ln_sigma_at_k_lo': float(lsig.mid()), 'sigma_tau_max': float(st.hi), 'K_E': float(KE), 'C_tot': float(Ctot),
            'log10_correction_bound': float(log10_bound.hi), 'certificate_margin': float(margin), 'negligible': ok}


if __name__ == '__main__':
    np.seterr(all='ignore')
    C = comb_corrections()
    print('S\'_C: tau >= %.3e (k = 3.43), log10 tau(k=240) >= %.0f; every BL7 correction to h <= 10^(%.4e) << %.2e: %s' % (
        C['tau_min'], C['log10_tau_at_k_hi'], C['log10_correction_bound'], C['certificate_margin'], C['negligible']), flush=True)
    checks = []
    for sig in (1e-24, 1e-3):
        for tau in (3.0, 6.0, 12.0, 25.0):
            for (x, q0, L2, s2) in ((1.0, [0.25], 1e-3, 1e-4), (4.0, [0.12, 0.01], 3e-3, 2e-4), (2.0, [0.2], 0.3, 0.02)):
                b = x / tau
                if b >= 1: continue
                r = T.check_state(sig, b, x, q0, L2, s2)
                checks.append({k: r[k] for k in ('tau', 'sigma', 'b', 'x', 'two_Nphi', 'BL7_upper', 'BL7_ratio', 'BL7_ok', 'BL7_v10form', 'BL7_v10form_ok', 'PF_le_R2over4', 'identity_rel_err')})
                print('sigma %.0e tau %4.1f x %.0f: 2N_phi / BL7 bound = %.6f  (assembled v1.0 form ok %s)' % (sig, tau, x, r['BL7_ratio'], r['BL7_v10form_ok']), flush=True)
    out = {'comb': C, 'checks': checks, 'all_checks_ok': all(c['BL7_ok'] and c['BL7_v10form_ok'] and c['PF_le_R2over4'] for c in checks)}
    print('all BL7 checks ok:', out['all_checks_ok'])
    json.dump(out, open('results/bm_bl7.json', 'w'), indent=1, default=float)
