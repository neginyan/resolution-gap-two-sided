"""Stage BL §1: the window lemmas in theta_tau form — explicit remainders (proofs in App. D of the paper, paper/sections/D_theta.tex),
the size of every correction for tau > tau0 = 80, and numerical checks against direct 2-D quadrature at small tau (where the corrections are visible).

Notation: p = N_V(v) Gamma(w) h(k'v, w)/Z, t = k'v, tau = k'^2 V = x/b, e_n = e^{-n^2 tau/2}, alpha = (1 - b)/sqrt2, e = e^{-alpha^2 V/2},
beta = alpha k' V <= sigma tau/sqrt(2x),  nu_k = 2 sum_{n>=1} n^k e_n e^{n beta}  (>= every theta-kernel norm used below),
omega_k = 2 sum_{n>=1} e_n |P_k(n sqrt tau)|, omega'_k = 2 sum n e_n |P_k(n sqrt tau)| with E[Z^k e^{iuZ}] = e^{-u^2/2} P_k(u) (Gaussian moments).
Lemma BL0 (theta calculus):  int N_V F(k'v) dv = <theta F>, theta = 1 + 2 sum e_n cos nt >= 0, |theta - 1| <= nu_0;  int N_V e^{i alpha v} F(k'v) dv = e <Theta F>,
   |Theta - 1| <= nu_0, |theta - Theta| <= beta nu_1, |Theta_t| <= nu_1;  int N_V v^k F(k'v) dv = V^{k/2}(m_k <F> + r), |r| <= <|F|> omega_k (F >= 0: |F^_n| <= <F>).
Lemma BL1 (AX1-theta):  J_ww = J_ww,per (1 + e1),  |e1| <= nu_0 (1 + 2 Qbar)/(1 - Qbar nu_0/2),  J_ww,per = 1/L^2 + int Gamma <h_w^2/h>.
Lemma BL2 (AX2-theta):  sigma^2 J_vv = b + W exactly,  Z W = sigma^2 int Gamma <theta h'^2/h> + x int Gamma sum n^2 q_n e_n, hence
   [(1 - nu_0) W_per - b tau Qbar nu_2/2]/Z <= W <= [(1 + nu_0) W_per + b tau Qbar nu_2/2]/Z,  |Z - 1| <= Qbar nu_0/2.
Lemma BL3 (AX3-theta; the remainders (i)-(iii) of stage BK, now proved):  with Phi = 1 - C, C = int Gamma <h cos psi> (period average), Psi = 1 - e C (1 - y),
   y = alpha^2 V,  Psi >= min(Phi, 1/2),  0 <= Phi <= 2,
   2 N_phi >= (1 - e - sigma^2/(1 - nu_0)) W + b Psi/(1 + Qbar nu_0/2) - sigma^2 gamma_P Phi/(1 - Qbar nu_0/2) - [b Qbar (nu_0 + tau nu_2) Phi/2 + sigma b E_R]/(1 - Qbar nu_0/2).
Lemma BL4 (Lemma 3-theta): 1 - C >= (1 - e^{-L^2/4}) - (L r + r^2/2)/2 with r <= xi_P + sigma^2 sqrt(J_ww/(1 - e1)).
Lemma BL5 (AW3-theta):  k_eff^2 in (x S_per + b/4) [(1 - nu_tau)/(1 + nu_0), (1 + nu_tau)/(1 - nu_0)],  nu_tau = 2 sum e_n (1 + n^2 tau); S_per obeys AX4-P and
   the Wiener bound unchanged (they are statements about the periodic function F = d_w sqrt(Gamma h)).
Lemma BL6 (Lemma 5-theta): r_C, r_w of Lemma 5 hold with the factor (1 + e5), e5 = (nu_0 + omega_2 + omega_4/3 + sqrt(tau)(omega'_1 + omega'_3) + Qbar nu_0)/(1 - Qbar nu_0/2).
-> results/bl_theta.json.  Python 3.11."""
import json
import numpy as np
from be_cert_p import eta as eta_p, q1, Qb, gamma_P

E_ = np.e; SIG0 = 1e-24; TAU0 = 80.0; NN = np.arange(1, 60, dtype=float)


def nus(tau, beta):
    en = np.exp(-NN ** 2 * tau / 2 + NN * beta)
    return {k: 2 * float(np.sum(NN ** k * en)) for k in range(5)}


def Pk(k, u):
    return {1: np.abs(u), 2: np.abs(1 - u * u), 3: np.abs(u ** 3 - 3 * u), 4: np.abs(u ** 4 - 6 * u * u + 3)}[k]


def omegas(tau):
    en = np.exp(-NN ** 2 * tau / 2); u = NN * np.sqrt(tau)
    return {k: 2 * float(np.sum(en * Pk(k, u))) for k in (1, 2, 3, 4)}, {k: 2 * float(np.sum(NN * en * Pk(k, u))) for k in (1, 2, 3, 4)}


def c_chi(x, N1, Q):
    return (np.sqrt(x) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)


def E_R(x, tau, sig, nu, N1, Q, gam):
    """sigma b E_R bounds the absolute remainder of Lemma BL3 (a1..a9 in the proof)."""
    cc = c_chi(x, N1, Q)
    return (0.375 * sig * tau * Q * nu[0] / x + (1 + Q) * (1 + Q / 2) * cc * nu[0] + Q * sig * tau ** 2 * nu[2] / (8 * x)
            + tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2 * x) * (1 - Q)) + sig * tau * nu[0] * (1 + Q) * gam / x
            + tau * nu[1] * (N1 * cc + (1 + Q) * gam / np.sqrt(2 * x)) + tau * Q * nu[2] * (1 + Q) * cc / 2 + Q * sig * tau ** 3 * nu[4] / (8 * x)
            + sig * tau * Q * nu[2] / (2 * (1 - nu[0])))


def eps_table(tau=TAU0, sig=SIG0):
    """the relative corrections of Lemmas BL1-BL6 at tau (each decreases in tau for tau >= 8: tau^k e^{-n^2 tau/2 + n sigma tau} with k <= 3)."""
    nu = nus(tau, sig * tau); om, omp = omegas(tau); Q = 0.25
    # x-suprema over [1, 400] (cells; every factor decreases beyond 8) of N1, gamma_P, c_chi
    xs = np.linspace(1.0, 400.0, 4001); N1s, gs, ccs = [], [], []
    for a, b in zip(xs[:-1], xs[1:]):
        N1 = float(q1(a)) * (1 + eta_p(1, a)); N1s.append(N1); gs.append(gamma_P(a, b)); ccs.append((np.sqrt(b) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2))
    N1m, gm, ccm = max(N1s), max(gs), max(ccs)
    ER = E_R(1.0, tau, sig, nu, N1m, Q, gm)            # every x-dependence of E_R is a factor 1/x or 1/sqrt x (worst at x = 1) times the suprema
    ER = max(ER, (0.375 * sig * tau * Q * nu[0] + (1 + Q) * (1 + Q / 2) * ccm * nu[0] + Q * sig * tau ** 2 * nu[2] / 8 + tau ** 2 * N1m ** 2 * nu[1] / (np.sqrt(2) * (1 - Q))
                  + sig * tau * nu[0] * (1 + Q) * gm + tau * nu[1] * (N1m * ccm + (1 + Q) * gm / np.sqrt(2)) + tau * Q * nu[2] * (1 + Q) * ccm / 2
                  + Q * sig * tau ** 3 * nu[4] / 8 + sig * tau * Q * nu[2] / (2 * (1 - nu[0]))))
    z = Q * nu[0] / 2
    e1 = nu[0] * (1 + 2 * Q) / (1 - z)
    e2 = (nu[0] + z) / (1 - z); e2b = tau * Q * nu[2] / (2 * (1 - z))
    e3 = max(1 - (1 / (1 + z) - (2 * Q * (nu[0] + tau * nu[2]) + 2 * ER) / (1 - z)), 1 / (1 - nu[0]) - 1, 1 / (1 - z) - 1)
    e4 = (1 + 2 * float(np.sum(np.exp(-NN ** 2 * tau / 2) * (1 + NN ** 2 * tau)))) / (1 - nu[0]) - 1
    e5 = (nu[0] + om[2] + om[4] / 3 + np.sqrt(tau) * (omp[1] + omp[3]) + z * 2) / (1 - z)
    out = {'tau0': tau, 'sigma_max': sig, 'nu': nu, 'omega': om, 'omega_prime': omp, 'sup_N1': N1m, 'sup_gamma_P': gm, 'sup_c_chi': ccm, 'E_R': ER,
           'e1_Jww': e1, 'e2_W_rel': e2, 'e2_W_add_coef_of_b': e2b, 'e3_AX3': e3, 'e4_keff': e4, 'e5_Lemma5': e5}
    out['eps_theta'] = max(e1, e2 + e2b, e3, e4, e5)
    return out


# ------------------------------------------------------------------------------------------------ numerical checks (not part of the proof)
def trap(x):
    w = np.zeros_like(x); d = np.diff(x); w[:-1] += d / 2; w[1:] += d / 2; return w


def fields(sig, b, x, q0, L2, s2, v, w):
    V = sig * sig / b; kp = np.sqrt(x) / sig; q0 = np.asarray(q0, float); n = np.arange(1, len(q0) + 1)
    t = kp * v; env = np.exp(-w ** 2 / (2 * s2)); qe = np.outer(q0, env)
    C = np.cos(np.outer(n, t)); S = np.sin(np.outer(n, t))
    h = 1 + np.einsum('kv,kw->vw', C, qe); h_t = -np.einsum('kv,kw->vw', n[:, None] * S, qe); h_tt = -np.einsum('kv,kw->vw', (n * n)[:, None] * C, qe)
    h_w = np.einsum('kv,kw->vw', C, qe * (-w / s2)[None, :]); h_tw = -np.einsum('kv,kw->vw', n[:, None] * S, qe * (-w / s2)[None, :])
    return V, kp, h, h_t, h_tt, h_w, h_tw


def check_state(sig, b, x, q0, L2, s2):
    V = sig * sig / b; kp = np.sqrt(x) / sig; tau = x / b; sv = np.sqrt(V); L = np.sqrt(L2)
    per = 2 * np.pi / kp; nv = int(max(1201, 64 * 22 * sv / per + 1201)); v = np.linspace(-11 * sv, 11 * sv, nv)
    w = np.unique(np.concatenate([np.linspace(-9 * L, 9 * L, 901), np.linspace(-9 * np.sqrt(s2), 9 * np.sqrt(s2), 451)]))
    V, kp, h, h_t, h_tt, h_w, h_tw = fields(sig, b, x, q0, L2, s2, v, w)
    Nv = np.exp(-v ** 2 / (2 * V)) / np.sqrt(2 * np.pi * V); Gm = np.exp(-w ** 2 / (2 * L2)) / np.sqrt(2 * np.pi * L2)
    WV, WW = trap(v), trap(w); P = (Nv[:, None] * h) * WV[:, None]                     # v-measure per w (unnormalised in w)
    Z = float((P.sum(0) * Gm * WW).sum())
    stl = kp * h_t / h; sv_ = -v[:, None] / V + stl; swl = -w[None, :] / L2 + h_w / h
    al = (1 - b) / np.sqrt(2); e = np.exp(-al * al * V / 2); y = al * al * V
    R = sig * sig * stl + w[None, :] + sig * sig * swl; psi = R / np.sqrt(2)
    mz = al * v[:, None] + psi; phi = 2 * np.sin(mz / 2) ** 2
    # per-w identity of Lemma BL3: sigma^2 int P s_v^2 phi dv = T_b + T_F + T_a + R1
    lhs = sig * sig * (P * sv_ ** 2 * phi).sum(0)
    Gv = lambda f: ((Nv * WV)[:, None] * f).sum(0)
    ph = np.exp(1j * al * v)[:, None]
    Av = lambda f: ((Nv * WV)[:, None] * ph * f).sum(0)
    hp = kp * h_t; hpp = kp * kp * h_tt; Fh = hp ** 2 / h
    psip = sig * sig / np.sqrt(2) * (kp * kp * (h_tt / h - h_t ** 2 / h ** 2) + kp * (h_tw / h - h_w * h_t / h ** 2))
    eps_ = np.exp(1j * psi); D = (hp - 1j * h * psip) * eps_
    T_b = b * (Gv(h) - np.real(Av(h * eps_))); T_F = sig * sig * (Gv(Fh) - np.real(Av(Fh * eps_)))
    T_a = sig * sig * np.real(Av(h * (al * al + 2 * al * psip) * eps_))
    AD1 = ((Nv * WV)[:, None] * (v[:, None] / V - 1j * al) * ph * D).sum(0)               # A[D'] by parts
    R1 = sig * sig * np.real(AD1) - sig * sig * Gv(hpp)
    rhs = T_b + T_F + T_a + R1
    ident = float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)))
    # exact quantities
    Nphi2 = float((lhs * Gm * WW).sum() / Z)                                              # 2 N_phi
    Jvv = float(((P * sv_ ** 2).sum(0) * Gm * WW).sum() / Z); Jww = float(((P * swl ** 2).sum(0) * Gm * WW).sum() / Z); W = sig * sig * Jvv - b
    # period averages on a t-grid (same w grid)
    tt = np.linspace(-np.pi, np.pi, 2049)[:-1]; vt = tt / kp
    _, _, hP, htP, httP, hwP, htwP = fields(sig, b, x, q0, L2, s2, vt, w)
    GWn = Gm * WW / float((Gm * WW).sum())                                               # Gamma-weights normalised on the (non-uniform) w grid
    Jper = 1 / L2 + float(((hwP ** 2 / hP).mean(0) * GWn).sum()); Wper = x * float(((htP ** 2 / hP).mean(0) * GWn).sum())
    RP = sig * sig * kp * htP / hP + w[None, :] + sig * sig * (-w[None, :] / L2 + hwP / hP); Cper = float(((hP * np.cos(RP / np.sqrt(2))).mean(0) * GWn).sum())
    Phi = float(((hP * 2 * np.sin(RP / (2 * np.sqrt(2))) ** 2).mean(0) * GWn).sum()); Psi = 1 - e * Cper * (1 - y)
    q0 = np.asarray(q0, float); n = np.arange(1, len(q0) + 1); Q = float(np.sum(np.abs(q0))); N1 = float(np.sum(n * np.abs(q0))); N2 = float(np.sum(n * n * np.abs(q0)))
    gam = (x * (N2 * (1 + Q) + N1 ** 2) + np.sqrt(x) * N1 * (1 + 2 * Q) / np.sqrt(E_)) / (1 - Q) ** 2
    beta = al * kp * V; nu = nus(tau, beta); z = Q * nu[0] / 2
    ER = E_R(x, tau, sig, nu, N1, Q, gam)
    ax3 = (1 - e - sig * sig / (1 - nu[0])) * W + b * Psi / (1 + z) - sig * sig * gam * Phi / (1 - z) - (b * Q * (nu[0] + tau * nu[2]) * Phi / 2 + sig * b * ER) / (1 - z)
    e1 = nu[0] * (1 + 2 * Q) / (1 - z)
    W_lo = ((1 - nu[0]) * Wper - b * tau * Q * nu[2] / 2) / (1 + z); W_hi = ((1 + nu[0]) * Wper + b * tau * Q * nu[2] / 2) / (1 - z)
    # k_eff (Lemma BL5): direct versus x S_per + b/4
    dvsw = kp * (h_tw / h - h_w * h_t / h ** 2)
    num = sig * sig * (((P * (sv_ * swl / 4 + dvsw / 2) ** 2)).sum(0) * Gm * WW).sum(); den_ = ((P * swl ** 2 / 4).sum(0) * Gm * WW).sum(); keff2 = float(num / den_)
    fP = -w[None, :] / L2 + hwP / hP; F2 = hP * fP ** 2 / 4; Ft2 = hP * (htP / (2 * hP) * fP / 2 + (htwP / hP - hwP * htP / hP ** 2) / 2) ** 2
    Sper = float((Ft2.mean(0) * GWn).sum() / (F2.mean(0) * GWn).sum())
    nut = 2 * float(np.sum(np.exp(-NN ** 2 * tau / 2) * (1 + NN ** 2 * tau))); base = x * Sper + b / 4
    k_lo, k_hi = base * (1 - nut) / (1 + nu[0]), base * (1 + nut) / (1 - nu[0])
    # Lemma BL7 (upper side): 2 N_phi <= (1-e) W + sigma^2 (1+nu0) P_F/(1-z) + b Psi/(1-z) + sigma^2 gamma Phi/(1-z) + [b Q (nu0 + tau nu2) Phi/2 + sigma b E_R]/(1-z)
    FP = kp * kp * htP ** 2 / hP; PF = float(((FP * 2 * np.sin(RP / (2 * np.sqrt(2))) ** 2).mean(0) * GWn).sum()); st2R2 = float(((FP * RP ** 2).mean(0) * GWn).sum())
    up = (1 - e) * W + sig * sig * (1 + nu[0]) * PF / (1 - z) + b * Psi / (1 - z) + sig * sig * gam * Phi / (1 - z) + (b * Q * (nu[0] + tau * nu[2]) * Phi / 2 + sig * b * ER) / (1 - z)
    up10 = ((1 + nu[0]) / (1 - z)) * ((1 - e) * Wper + sig * sig * st2R2 / 4 + b * (1 - e) + b * Phi + sig * sig * al * al * e + sig * sig * gam * Phi) \
        + ((1 - e) * b * tau * Q * nu[2] / 2 + b * Q * (nu[0] + tau * nu[2]) * Phi / 2 + sig * b * ER) / (1 - z)
    extra = {'BL7_upper': float(up), 'BL7_ok': bool(Nphi2 <= up * (1 + 1e-9)), 'BL7_ratio': float(Nphi2 / up), 'BL7_v10form': float(up10),
             'BL7_v10form_ok': bool(Nphi2 <= up10 * (1 + 1e-9)), 'PF_le_R2over4': bool(PF <= st2R2 / 4 * (1 + 1e-9))}
    return {**extra, 'tau': tau, 'sigma': sig, 'b': b, 'x': x, 'identity_rel_err': ident, 'two_Nphi': Nphi2, 'AX3_theta_rhs': float(ax3), 'AX3_ok': bool(Nphi2 >= ax3),
            'AX3_ratio': float(Nphi2 / ax3) if ax3 > 0 else None, 'Psi_ge_min': bool(Psi >= min(Phi, 0.5) - 1e-12), 'Phi': Phi, 'Psi': Psi,
            'Jww': Jww, 'Jww_per': Jper, 'Jww_ok': bool(abs(Jww / Jper - 1) <= e1 + 1e-9), 'Jww_rel_dev': abs(Jww / Jper - 1), 'e1': e1,
            'W': W, 'W_lo': W_lo, 'W_hi': W_hi, 'W_ok': bool(W_lo * (1 - 1e-9) <= W <= W_hi * (1 + 1e-9)), 'W_period_upper_v10': Wper / (1 - Q),
            'keff2': keff2, 'keff2_lo': k_lo, 'keff2_hi': k_hi, 'keff_ok': bool(k_lo * (1 - 1e-7) <= keff2 <= k_hi * (1 + 1e-7)), 'E_R': ER, 'nu0': nu[0]}


if __name__ == '__main__':
    np.seterr(all='ignore')
    T = eps_table(); print('tau0 = 80: nu_0 = %.2e, E_R = %.2e | e1 %.1e  e2 %.1e (+%.1e b)  e3 %.1e  e4 %.1e  e5 %.1e  -> eps_theta = %.2e' % (
        T['nu'][0], T['E_R'], T['e1_Jww'], T['e2_W_rel'], T['e2_W_add_coef_of_b'], T['e3_AX3'], T['e4_keff'], T['e5_Lemma5'], T['eps_theta']), flush=True)
    T40 = eps_table(40.0)
    checks = []
    for sig in (1e-24, 1e-3):
        for tau in (3.0, 6.0, 12.0, 25.0):
            for (x, q0, L2, s2) in ((1.0, [0.25], 1e-3, 1e-4), (4.0, [0.12, 0.01], 3e-3, 2e-4), (2.0, [0.2], 0.3, 0.02)):
                b = x / tau
                if b >= 1: continue
                if s2 < sig * sig: continue
                try: r = check_state(sig, b, x, q0, L2, s2)
                except MemoryError: continue
                checks.append(r)
                print('sigma %.0e tau %5.1f x %.0f: identity %.1e | 2N_phi/AX3theta %s | J_ww dev %.1e <= %.1e %s | W %.3e in [%.3e, %.3e] %s (v1.0 upper %.3e) | k_eff^2 %.4e in [%.4e, %.4e] %s' % (
                    sig, tau, x, r['identity_rel_err'], ('%.4f' % r['AX3_ratio']) if r['AX3_ratio'] else 'rhs<0', r['Jww_rel_dev'], r['e1'], r['Jww_ok'], r['W'], r['W_lo'], r['W_hi'],
                    r['W_ok'], r['W_period_upper_v10'], r['keff2'], r['keff2_lo'], r['keff2_hi'], r['keff_ok']), flush=True)
    out = {'eps_table_tau80': T, 'eps_table_tau40': T40, 'checks': checks,
           'all_checks_ok': all(c['AX3_ok'] and c['Jww_ok'] and c['W_ok'] and c['keff_ok'] and c['Psi_ge_min'] and c['identity_rel_err'] < 1e-6 for c in checks)}
    print('all checks ok:', out['all_checks_ok'])
    json.dump(out, open('results/bl_theta.json', 'w'), indent=1, default=float)
