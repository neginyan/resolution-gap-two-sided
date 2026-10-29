"""Stage BK §0: Lemma BK1 — a narrow v-envelope forces a large gap (computer-assisted proof; no separation of the ripple from the envelope is used).

Lemma BK1.  Let rho be a periodic window with (P2)-(P5), L_w <= sigma^{-1/2}, sigma <= 1e-24, and 0 < c <= c* = 0.134/0.9.  If
    tau := k'^2 V = x/b <= tau0 = 40    (the blurred v-envelope holds fewer than ~ sqrt(tau0)/2 ripple periods),
then (a) >= c or (a') >= c.  Hence every state with g < G (G <= 0.134, |eps| <= 0.1) has tau > 40, where the period averages of paper F hold
up to the relative factor delta_tau = 2 sum_n e^{-n^2 tau/2} <= 4.2e-9 (Lemma BK2).

Proof (all quantities exact; p = N_V(v) Gamma(w) h(k'v, w), h = 1 + sum_n q_n(w) cos nt, q_n(w) = q0_n e^{-w^2/2s^2}; q = |q0_1|; e_n = e^{-n^2 tau/2};
|q0_n| <= q e^{-(n^2-1)x/2} by (P2) and K >= x; q <= m(x) = min(1/4, 2e^{-x/2}); h >= 1 - Q, Q <= 1/4).
 (i)  (a) >= sigma^2 J_ww/(sigma den) (phi <= 2), den = b + W + sigma^2 J_ww, b + W = sigma^2 J_vv <= 1.  (a) < c gives den < theta (b + W), theta = 1/(1 - c sigma).
 (ii) w-Cramer-Rao: J_ww >= J(marginal) >= 1/Var(marginal); the marginal is Gamma(w)(1 + sum q0_n e^{-w^2/2s^2} e_n) (Gaussian average of cos nt = e_n), so
      Var <= kappa_w L^2, kappa_w = (1 + q H0 g1 r_s)/(1 - q H0 g1), g1 = s/sqrt(s^2 + L^2) <= 1/sqrt5, r_s = s^2/(s^2 + L^2) <= 1/5 ((P4)).
      With (i): L^2 > L1^2 := sigma/(c theta kappa_w (b + W)).
 (iii) W exactly: W = (x/Z) int Gamma <theta_tau (h_t^2/h - h_tt)>, theta_tau the wrapped normal (0 <= theta_tau <= 1 + delta_tau), -<theta h_tt> = sum n^2 q_n e_n,
      Z = 1 + g1 sum q0_n e_n:  W <= W_N := x[(1 + delta_tau) q^2 (1 + Z2) g_s/(2(1 - Q)) + g1 q H2]/(1 - g1 q H0),  g_s <= 1/3.
 (iv) v-Cramer-Rao on every w-slice: int P_w s_v^2 dv >= rho_w(w)/Var_w, Var_w <= kappa_s V, kappa_s = (1 + q H1)/(1 - q H0)
      (int N_V v^2 cos(n k' v) = V(1 - n^2 tau) e_n).  m_z = A(w) + delta, A = w(1 - sigma^2/L^2)/sqrt2, |delta| <= ((1 - b)|v| + d)/sqrt2,
      d = sigma sqrt(x) N1/(1 - Q) + sigma Q e^{-1/2}/(1 - Q) (|sigma^2 h_t k'/h|, |sigma^2 h_w/h|, s >= sigma), and
      1 - cos(A + delta) >= 1 - cos A - |sin A||delta| - delta^2/2.  With Gaussian moments of s_v^2 (|s_v| <= |v|/V + k' N1/(1 - Q)):
      sigma^2 int P s_v^2 phi >= b Psi (1 - q H0 g1)/kappa_s - Err,  Psi = 1 - e^{-gamma^2 L^2/2} (gamma = (1 - sigma^2/L^2)/sqrt2), Err explicit (O(sigma sqrt b)).
 (v)  (a') < c gives sigma^2 int P s_v^2 phi = 2 N_phi Z < 2 c sigma theta (b + W) Z.  With L^2 > L1^2 (Psi increasing; Err grows only like sigma sqrt(b) L):
      contradiction as soon as  b Psi(L1^2)(1 - q H0 g1)/kappa_s - Err > 2 c sigma theta (b + W_N)(1 + g1 q H0)  (checked also on the branch gamma L >= 0.71).
Cells: x in [1, 40] (b >= x/tau0 <= 1 forces x <= tau0), b in [x/tau0, 1]; each cell is evaluated at its worst corner (W_N, kappa decrease in tau, so
tau_min = x_a/b_b; x-factors at x_b, decay factors at x_a; the main term at b_a, the right side at b_b).  sigma enters only as a small factor (worst at 1e-24).
-> results/bk_narrow_lemma.json.  Python 3.11."""
import json
import numpy as np

QS, RS = 0.25, 0.125; G1 = np.sqrt(2 * RS / (1 + 2 * RS)); RSH = 2 * RS / (1 + 2 * RS); GS = np.sqrt(RS / (1 + RS))
CST = 0.134 / 0.9; SIG = 1e-24; TAU0 = 40.0; E_ = np.e
NN = np.arange(1, 40, dtype=float)


def harm(x, tau):
    dec = np.exp(-(NN ** 2 - 1) * x / 2); en = np.exp(-NN ** 2 * tau / 2)
    H0 = float(np.sum(dec * en)); H1 = float(np.sum(dec * np.abs(1 - NN ** 2 * tau) * en)); H2 = float(np.sum(NN ** 2 * dec * en))
    Z2 = float(np.sum((NN[1:] ** 2) * np.exp(-(NN[1:] ** 2 - 1) * x))); eta1 = float(np.sum(NN[1:] * np.exp(-(NN[1:] ** 2 - 1) * x / 2)))
    delta = 2 * float(np.sum(en)); return H0, H1, H2, Z2, eta1, delta


def m(x): return min(QS, 2 * np.exp(-x / 2))


def cell(xa, xb, ba, bb, c=CST, sig=SIG):
    th = 1 / (1 - c * sig); tau_min = xa / bb
    H0, H1, H2, _, _, delta = harm(xa, tau_min)                       # all decrease in tau and in x (decay factors): worst at (xa, tau_min)
    _, _, _, Z2, eta1, _ = harm(xa, 0.0)
    q = m(xa)
    kw = (1 + q * H0 * G1 * RSH) / (1 - q * H0 * G1)
    ks = (1 + q * H1) / (1 - q * H0)
    WN = xb * ((1 + delta) * q * q * (1 + Z2) * GS / (2 * (1 - QS)) + G1 * q * H2) / (1 - G1 * q * H0)
    # (ii): L1^2 at the largest b + W (right corner)
    L12 = sig / (c * th * kw * (bb + WN)); gam2 = (1 - c * th * kw * (bb + WN) * sig) ** 2 / 2
    Psi1 = -np.expm1(-gam2 * L12 / 2)
    Np = (q * (1 + eta1)) ** 2 / (1 - QS) ** 2; tauN = (xb / ba) * Np                    # tau N' at the largest tau of the cell
    d = sig * np.sqrt(xb) * q * (1 + eta1) / (1 - QS) + sig * QS * np.exp(-0.5) / (1 - QS)
    Ea = (1 + QS) / np.sqrt(2) * (sig * np.sqrt(2 * bb) * np.sqrt((1 + tauN) * (6 + 2 * tauN)) + 2 * bb * d * (1 + tauN))
    Eb = (1 + QS) * (sig * sig * (3 + tauN) + bb * d * d * (1 + tauN))
    main_fac = ba * (1 - q * H0 * G1) / ks
    rhs = 2 * c * sig * th * (bb + WN) * (1 + G1 * q * H0)
    gL1 = np.sqrt(gam2 * L12)
    f_L1 = main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb - rhs
    increasing = bool(main_fac * gL1 * np.exp(-gam2 * L12 / 2) >= Ea)                    # d/d(gamma L) of the small-L branch at L1 (Psi' = gamma L e^{-...})
    f_big = main_fac * (-np.expm1(-0.71 ** 2 / 2)) - Ea * 0.71 - Eb - rhs                  # branch gamma L >= 0.71 (Psi >= 1 - e^{-0.25})
    ok = bool(f_L1 > 0 and increasing and f_big > 0)
    return {'ok': ok, 'margin': float((main_fac * Psi1 - Ea * min(gL1, 0.71) - Eb) / rhs), 'W_N': WN, 'kappa_w': kw, 'kappa_s': ks, 'tau_min': tau_min}


if __name__ == '__main__':
    xs = np.unique(np.concatenate([np.geomspace(1.0, 40.0, 241)]))
    res = {'cells': 0, 'all_ok': True, 'worst_margin': np.inf, 'worst': None}
    for xa, xb in zip(xs[:-1], xs[1:]):
        b_lo = xa / TAU0
        bs = np.geomspace(b_lo, 1.0, 241)
        for ba, bb in zip(bs[:-1], bs[1:]):
            r = cell(xa, xb, ba, bb); res['cells'] += 1
            if not r['ok']: res['all_ok'] = False
            if r['margin'] < res['worst_margin']: res['worst_margin'] = r['margin']; res['worst'] = {'x': [xa, xb], 'b': [ba, bb], **r}
    # the cells cover {1 <= x <= 40, x/40 <= b <= 1}: the lower b-edge of each x-cell is x_a/40 <= x/40
    print('Lemma BK1: %d cells, all ok: %s; smallest margin (left side / right side) %.3f at %s' % (res['cells'], res['all_ok'], res['worst_margin'], res['worst']))
    json.dump(res, open('results/bk_narrow_lemma.json', 'w'), indent=1, default=float)
