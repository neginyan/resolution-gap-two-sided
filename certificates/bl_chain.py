"""Stage BL §2: the lower-side chain certificate (Theorem 9, 421 cells) redone with the theta_tau forms of the window lemmas
(computer-assisted proof).  Every input lemma is perturbed by a relative eps_in = 1e-6 in the bad direction; the theta-lemmas of stage BL
(bl_theta.py, Lemmas BL1-BL6) give corrections below eps_theta < 1e-11 for tau > 80, and Lemma BK1 (tau0 = 80, bl_narrow80.py) gives tau > 80
for every state with g < G.

Changes against bf_cert_p.chain (v1.0), one per input lemma (eps = eps_in):
  AX1-theta (J_ww >= (1 - eps) J_ww,per) and Lemma 3-theta (r with J_ww,per <= J_ww/(1 - eps)):  eta -> eta' = eta/(1 - eps)^2 everywhere
      (eta enters only through the upper bounds of (L1); a larger eta keeps them true, Lemma BH1).
  AW3-theta (k_eff^2 within (1 +- eps)(x S_per + b/4)):  U0 = F^2/(4 G^2 (1 + eps)) - 1/4, and the conclusion g >= F/(2 sqrt((1 + eps)(X + S1 + 1/4))).
  AX4-P, Wiener (S-bounds): statements about the periodic S_per; unchanged.
  AX3-theta:  2 N_phi >= (1 - e - (1 + eps) sigma^2) W + (1 - eps) b Psi - (1 + eps) sigma^2 gamma_P (1 - C),  Psi >= Phi_b := min(1 - C, 1/2),
      1 - C <= 4 Phi_b.  Hence: condA  (1 - eps) Phi_min >= 2(2 tau0 + kap sigma^2);  case B: b < 8 sigma^2 gamma'/(1 - eps) and the threshold
      sigma <= (1 - eps) U_eff/(128 eta' G theta (1 + sigma^2) sup x gamma');  beta' = [(tau0 + kap sigma^2) + 4 r_wp' Phi_b,min/(1 - eps)]/(Phi_b,min - tau0);
      nu' = kap sigma^2 + tau0 + 2 r_wp';  AM-GM: sqrt(Phi_b) omega' < Gamma2' sqrt(W),  omega' = sqrt(1 - eps)(1 - b_max) e^{-V_max/8} - 2 om_term',
      Gamma2' = 2 eta' G theta (1 + beta') + sqrt(kap) sigma;  gamma' = (1 + eps) gamma_P,  kap = 1 + eps.
  AX2-theta (W <= (1 + eps) W_per,up + eps b, used only after b <= beta' W):  W_p,max, x q (in L2max) and Gamma3P get the factor (1 + 3 eps).
  Lemma 5-theta (r_C, r_w): factor (1 + eps).
-> results/bl_chain_cells_part*.jsonl (one line per cell, resumable), results/bl_chain_tail.json.  usage: python bl_chain.py [PART NPARTS] | tail.  Python 3.11."""
import json, sys, os, time
_ARGV = list(sys.argv)
import numpy as np
from scipy.optimize import brentq
import bf_cert_p as B
import bh_cert_cells as H
from be_cert_p import zeta, kap2, Qb, gamma_P, S1_P, QB, RHO

EPS_IN = 1e-6
SQ2 = np.sqrt(2.0); E = np.e
cell_sup = H._cell_sup_floored            # suprema floored at 1e-300 (valid upper bound where they underflow), as in bh_cert_cells
N1 = B.N1


def chain(sig, lnsig, G, eta_, F, x_lo, S1, eps=EPS_IN):
    eta_ = eta_ / (1 - eps) ** 2; kap = 1 + eps
    th = 1 / (1 - eta_ * G * sig); U0 = F * F / (4 * G * G * (1 + eps)) - 0.25
    if U0 <= 0: return {'ok': False, 'why': 'U0 <= 0'}
    z0, z2 = zeta(0, x_lo), zeta(2, x_lo); kp = kap2(); UP = U0 / (kp * (1 + z2)); Ueff = UP / (1 + QB)
    lrhs = np.log(UP / (SQ2 * 4 * (1 + z0))) - 1.5 * lnsig
    if lrhs > -1: return {'ok': False, 'why': 'x range'}
    x_max = brentq(lambda x: np.log(x) - x - lrhs, 1.0, 1e9); xr = max(x_max, 60.0); xe = max(min(xr, 400.0), x_lo + 20.0)
    sup_xg = (1 + eps) * cell_sup(lambda a, b: b * gamma_P(a, b), x_lo, xe)                    # sup x gamma'
    L2min = sig / (eta_ * G * th)
    xi = sig * cell_sup(lambda a, b: np.sqrt(b) * N1(a) / (1 - Qb(a)), x_lo, xe)
    r = xi + sig ** 1.5 * np.sqrt(eta_ * G * th)
    dR = max(r / (2 * np.sqrt(L2min)) + r * r / (4 * L2min), (r / np.sqrt(sig) + r * r / 2) / 2)
    phi1 = 1 - np.exp(-0.25) - dR
    Phi_min = phi1 * L2min; Phi_max = 2.0; Pb_min = (1 - eps) * Phi_min                       # b-coefficient (1 - eps) Psi >= (1 - eps) Phi_b
    tau0 = 2 * eta_ * G * sig * th
    condA = Pb_min >= 2 * (2 * tau0 + kap * sig * sig)
    sig_caseB = (1 - eps) * Ueff / (128 * eta_ * G * th * (1 + sig * sig) * sup_xg)
    r_wp = sig * 16 * eta_ * G * th * (1 + sig * sig) * sup_xg / Ueff
    beta = ((tau0 + kap * sig * sig) + 4 * r_wp * Pb_min / (1 - eps)) / (Pb_min - tau0)
    Wp_max = (1 + 3 * eps) * cell_sup(lambda a, b: b * min(QB ** 2, 4 * np.exp(-a)), x_lo, max(60.0, x_lo + 20.0)) * (1 + z2) * (1 / 3) / (2 * (1 - QB))
    b_max = beta * Wp_max; nu = kap * sig * sig + tau0 + Phi_max * r_wp
    a2 = (1 - b_max) ** 2 / 2; V_max = 2 * (-np.log1p(-nu)) / a2
    om_term = 2 * (1 + eps) * cell_sup(lambda a, b: sig * gamma_P(a, b) * np.sqrt(Phi_max) * np.sqrt(16 * eta_ * G * th * b * (1 + sig * sig) / (sig * Ueff)), x_lo, xe)
    omega = np.sqrt(1 - eps) * (1 - b_max) * np.exp(-V_max / 8) - om_term
    Gam2 = 2 * eta_ * G * th * (1 + beta) + np.sqrt(kap) * sig
    L_gt_1_excluded = bool(phi1 * omega ** 2 / Gam2 ** 2 > Wp_max)
    Gam3P = (1 + 3 * eps) * 2 * eta_ * G * th * (1 + beta) * (1 + QB) * (1 + z2) * (1 + RHO) ** 1.5 / (1 - QB)
    K = np.log(256 * Gam2 ** 2 * Gam3P * (1 + z2) * (1 + z0) ** 3 / ((1 - QB) * UP ** 3 * phi1 * omega ** 2))
    X = brentq(lambda X: 4 * X - lnsig - 5 * np.log(X) - K, 1.0, 1e9) + 1e-9
    Qx = Qb(x_lo); kQ = (1 + Qx) / (1 - Qx)
    xq = (1 + 3 * eps) * cell_sup(lambda a, b: b * min(QB ** 2, 4 * np.exp(-a)), x_lo, max(X, x_lo + 1.0), 2001)
    L2max = min(1.0, 0.5 * Gam2 ** 2 * xq * (1 + z2) / (2 * SQ2 * (1 - QB) * phi1 * omega ** 2))
    s2max = L2max / 4; sos_max = np.sqrt(sig * Gam3P * X)
    xiX = sig * cell_sup(lambda a, b: np.sqrt(b) * N1(a) / (1 - Qb(a)), x_lo, max(X, x_lo + 1.0), 2001)
    sxN1 = cell_sup(lambda a, b: np.sqrt(b) * N1(a), x_lo, max(X, x_lo + 1.0), 2001)
    l4 = (eta_ * G * th * sig) ** 2
    Xi = xiX ** 2 + sig ** 2 * sos_max ** 2 * Qx ** 2 / (E * (1 - Qx) ** 2)
    dH = sig * eta_ * G * th + sos_max ** 2 * (Qx / (1 - Qx) + Qx * Qx / (E * (1 - Qx) ** 2)) + sxN1 * sos_max / (np.sqrt(E) * (1 - Qx) ** 2)
    rC = (1 + eps) * (0.5 * V_max + (1 + 3 * l4) * (L2max + s2max) / 4 + 0.75 * Xi)
    rw = (1 + eps) * (V_max / 8 + (1 + 3 * l4) * (0.75 * L2max + 0.375 * kQ * s2max) + 0.75 * Xi)
    F_int = (1 - eta_ * G * sig * th) * (1 - dH) * (1 - rC) * np.sqrt(max(0.0, 1 - rw))
    absorb = eta_ * G * th / phi1                                                             # <= 1: the absolute remainder of AX3-theta is relative
    ok = bool(condA and sig <= sig_caseB and omega > 0 and 0 < beta < 1 and nu < 1 and phi1 > 0 and L_gt_1_excluded and F_int > 0 and absorb <= 1)
    return {'ok': ok, 'U0': U0, 'x_lo': x_lo, 'phi1': phi1, 'beta': beta, 'nu': nu, 'V_max': V_max, 'omega': omega, 'Gam2': Gam2, 'Gam3P': Gam3P,
            'Wp_max': Wp_max, 'W_p_forced_if_L_gt_1': phi1 * omega ** 2 / Gam2 ** 2, 'sigma_caseB_limit': sig_caseB, 'K': K, 'X': X,
            'L2max': L2max, 'dH': dH, 'rC': rC, 'rw': rw, 'F_int': F_int, 'eta_prime': eta_, 'theta': th, 'absorb_ratio': absorb, 'condA_margin': Pb_min / (2 * (2 * tau0 + kap * sig * sig))}


def cell(la, lb, S1, Si1, eps=EPS_IN):
    s_hat = np.exp(-la) if la < 690 else 1e-300
    G = H.G_of(la); F = 1.02 * H.G_of(lb); x_lo = 1.0; steps = 0
    for k in range(60):
        c = chain(s_hat, lb, G, H.ETA, F, x_lo, S1, eps)
        if not c['ok']: return {'la': la, 'lb': lb, 'ok': False, 'fail': {kk: v for kk, v in c.items() if not isinstance(v, list)}}
        Fn = 0.9 * c['F_int']; steps += 1
        U0n = Fn * Fn / (4 * G * G * (1 + eps)) - 0.25; x_new = max(1.0, U0n / Si1, U0n - S1)
        if Fn <= F * (1 + 1e-13) and x_new <= x_lo * (1 + 1e-13): break
        F = max(F, Fn); x_lo = max(x_lo, x_new)
    th = c['theta']; etap = c['eta_prime']
    Fdef = etap * G * s_hat * th + c['dH'] + c['rC'] + c['rw']; Fint = max(0.0, 1 - Fdef * (1 + 1e-9))
    K = c['K'] + 1e-9; cc = S1 + 0.25
    Xa = H.Xroot(la, K) * (1 + 1e-12); Xb = c['X']
    d = 4 * cc - 5 - 5 * cc / Xa + 5 * np.log(Xa) + K
    Bpart = Fint * np.sqrt(la / ((1 + eps) * (Xa + cc))) / 2 if d >= 0 else Fint * np.sqrt(la / ((1 + eps) * (Xb + cc))) / 2
    Gpart = min(0.134 * np.sqrt(la), 1.2)
    L = min(Gpart, Bpart) * (1 - 1e-12)
    return {'la': la, 'lb': lb, 'ok': True, 'n_steps': steps, 'x_lo': x_lo, 'K': K, 'Xa': Xa, 'd_sign': d, 'Fdef': Fdef, 'F_int': Fint, 'Bpart': Bpart,
            'Gpart': Gpart, 'L_cell': L, 'c1_needed': (1 - L) * la / np.log(la), 'margin_c1_1.7': L - (1 - 1.7 * np.log(la) / la),
            'beta': c['beta'], 'omega': c['omega'], 'phi1': c['phi1'], 'nu': c['nu'], 'caseB_margin': c['sigma_caseB_limit'] / s_hat,
            'Lgt1_margin': c['W_p_forced_if_L_gt_1'] / c['Wp_max'], 'condA_margin': c['condA_margin'], 'absorb_ratio': c['absorb_ratio']}


FN = 'results/bl_chain_cells_part%d.jsonl'


def done_cells():
    import glob
    out = {}
    for line in (l for f in sorted(glob.glob(FN.replace('%d', '*'))) for l in open(f)):
        line = line.strip()
        if line:
            try: r = json.loads(line); out[r['i']] = r
            except json.JSONDecodeError: pass                                               # a line cut by a restart is recomputed
    return out


def tail():
    """Lemma 7 (tail) in the same form: the last cell's constants, T(l) = (1 - Fdef_end) sqrt(l/((1 + eps)(X + c)))/2 with
    4X = l + 5 ln X + K_end - (9/2) ln(l/l_end), and direct perturbed cells at l = 1e6, 1e7, 1e8."""
    R = done_cells(); last = R[max(R)]; l_end = last['lb']; K_end = last['K']; Fdef_end = last['Fdef']
    S1 = S1_P(); Si1 = B.Si_at(1.0); c = S1 + 0.25
    def T(l):
        K = K_end - 4.5 * np.log(l / l_end); X = brentq(lambda X: 4 * X - l - 5 * np.log(X) - K, 1.0, l + 1e3 + abs(K), maxiter=2000)
        return (1 - Fdef_end) * np.sqrt(l / ((1 + EPS_IN) * (X + c))) / 2
    grid = np.exp(np.linspace(np.log(l_end), np.log(1e300), 400)); Tv = np.array([T(l) for l in grid])
    out = {'eps_in': EPS_IN, 'l_end': l_end, 'K_end': K_end, 'Fdef_end': Fdef_end, 'T_at_l_end': float(Tv[0]), 'T_min': float(Tv.min()),
           'T_increasing': bool(np.all(np.diff(Tv) >= -1e-15)), 'T_at_1e300': float(Tv[-1]), 'direct': {}}
    for l in (1e6, 1e7, 1e8):
        r = cell(l, l * 1.0000001, S1, Si1)
        out['direct']['%g' % l] = {'ok': r['ok'], 'L_cell': r.get('L_cell'), 'K': r.get('K'), 'K_bound_from_tail': K_end - 4.5 * np.log(l / l_end)}
        print('tail: direct perturbed cell at l = %g: ok %s  L %.9f  K %.3f <= %.3f' % (l, r['ok'], r['L_cell'], r['K'], K_end - 4.5 * np.log(l / l_end)), flush=True)
    print('tail bound with eps_in: T(l_end) = %.6f, increasing %s, T(1e300) = %.12f' % (Tv[0], out['T_increasing'], Tv[-1]), flush=True)
    json.dump(out, open('results/bl_chain_tail.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    if len(_ARGV) > 1 and _ARGV[1] == 'tail': tail(); sys.exit(0)
    part, nparts = (int(_ARGV[1]), int(_ARGV[2])) if len(_ARGV) > 2 else (0, 1)
    S1 = S1_P(); Si1 = B.Si_at(1.0); C = H.cells(); have = done_cells(); t0 = time.time()
    for i, (a, b) in enumerate(C):
        if i % nparts != part or i in have: continue
        r = cell(a, b, S1, Si1); r['i'] = i; r['eps_in'] = EPS_IN
        with open(FN % part, 'a') as f: f.write(json.dumps(r, default=float) + '\n')
        print('cell %3d l [%.3f, %.3f] ok %s L %.6f c1 %.4f margin(1.7) %+.2e  %.0fs' % (i, a, b, r['ok'], r.get('L_cell', 0), r.get('c1_needed', 0),
              r.get('margin_c1_1.7', 0), time.time() - t0), flush=True)
