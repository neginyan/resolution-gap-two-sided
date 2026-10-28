"""Stage BH §1: Theorem AY1-P certified on a continuous range — cells in l = ln(1/sigma) covering [24 ln 10, 1e5 ln 10], and all |eps| <= 0.1 at once.

Lemma BH1 (monotonicity; proof by inspection of the chain, bf_cert_p.chain).  Write s for every occurrence of sigma as a small factor, l for the two
occurrences of ln(1/sigma) (the a-priori range of x and the root X), G for the threshold, eta = 1/(1 - |eps|) and F for the lower bound of F_c.  In the
sigma-normalised form of the chain every certified quantity is monotone:
  theta = 1/(1 - eta G s) increases in (s, G, eta);  r/sqrt(L2min) = sqrt(s eta G theta)(S_xi + sqrt(s eta G theta)),  r^2/L2min ∝ s,  (r/sqrt s + r^2/2)/2  increase
  in s  =>  phi1 decreases;  Phi_min/(Phi_min - tau0) = phi1/(phi1 - 2(eta G theta)^2) and tau/(Phi_min - tau0) = (2 eta G theta + s)/(phi1/(eta G theta) - 2 eta G theta)
  increase  =>  beta, nu (∝ s), V_max increase, omega (the om-term ∝ sqrt s) decreases, Gam2, Gam3_P increase;  the Phi_C condition reads
  phi1/(eta G theta) >= 8 eta G theta + 2s, the case-B condition s <= U_eff/(32 eta G theta sup(x gamma_P));  U0 = F^2/(4G^2) - 1/4 decreases in G and increases in F;
  K_P = ln(256 Gam2^2 Gam3_P (1+zeta_2)(1+zeta_0)^3/((1-Qbar) U_P^3 phi1 omega^2)) therefore increases in (s, G, eta) and decreases in F; the range of x and X
  increase in l; delta_H, r_C, r_w (Lemma AY2-P) increase in s, X and L2max.  Every condition of the certificate is a statement "monotone quantity <= or >= constant"
  in the bad direction.  Hence, on a cell l in [la, lb] and for all |eps| <= 0.1, evaluating the chain at
     s = sigma_a = e^{-la} (>= every sigma of the cell; 1e-300 for la > 690),  l = lb,  G_cell = G(la) >= G(l),  eta = 1/0.9,
  gives constants that are valid for every state of the cell (the chain is a proof by contradiction from g < G(l) <= G_cell; eta enters only through the
  upper bounds sigma^2 J_ww < eta G sigma den and N_phi < eta G sigma den of (L1), which stay true for every eta' >= eta).  The bootstrap starts from (P8):
  F_c >= 1.02 G(l) >= 1.02 G(lb) =: F_0, and each step F_{k+1} = 0.9 F_int(F_k) is an implication for every |eps| <= 0.1 (F_c >= (1 - |eps|) A1 A2 A3 >= 0.9 A1 A2 A3).
  G(l) := min(0.134, 1.2/sqrt l) (independent of eps; 0.134 <= 0.149 x 0.9).
Conclusion on the cell (for every |eps| <= 0.1):  g >= min(G(l), (1 - |eps|) F_int*/(2 sqrt(X(l) + S1_P + 1/4))),  X(l) <= root of 4X = l + 5 ln X + K_cell,
  F_int* >= 1 - Fdef, Fdef = eta G sigma theta + delta_H + r_C + r_w.  With h(X) = (4X - 5 ln X - K)/(X + c) = l/(X + c), c = S1_P + 1/4:  h' has the sign of
  d(X) = 4c - 5 - 5c/X + 5 ln X + K, increasing in X; if d(X(la)) >= 0 then min over the cell of l/(X(l) + c) is at la.  So
     g sqrt(l)/(1 - |eps|) >= L_cell := min( G(lb) sqrt(la)/(1 - 0.1)... see below, F_int* sqrt(la/(X(la) + c))/2 )
  (the G-branch: g >= G(l) >= (1 - |eps|) G(l), and G(l) sqrt(l) >= min(0.134 sqrt(la), 1.2)).
c1:  L_cell >= 1 - c1 ln l/l for every l in the cell  <=  c1 >= (1 - L_cell) la/ln(la)  (ln l/l decreasing for l > e).
Suprema that underflow (x_lo > ~700) are replaced by 1e-300, an upper bound of their true values.
-> results/bh_cert_cells_part*.json (one record per cell).  usage: python bh_cert_cells.py [PART NPARTS].  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, os, time
_ARGV = list(sys.argv)     # be_plemmas (imported through bf_cert_p) truncates sys.argv
import numpy as np
from scipy.optimize import brentq
import bf_cert_p as B

_cs = B.cell_sup
def _cell_sup_floored(f, x0, x1, n=4001):
    # for x_lo > ~700 the suprema underflow to 0; their true values are below 1e-300, so 1e-300 is a valid upper bound
    return max(_cs(f, x0, x1, n), 1e-300)
B.cell_sup = _cell_sup_floored
ETA = 1 / 0.9; RATIO = 1.02; L_LO = 24 * np.log(10); L_HI = 1e5 * np.log(10)


def G_of(l): return min(0.134, 1.2 / np.sqrt(l))


def cells():
    out = []; a = L_LO * (1 - 1e-12)
    while a < L_HI:
        b = min(a * RATIO, L_HI * (1 + 1e-12)) if a * RATIO < L_HI else L_HI * (1 + 1e-12); out.append((a, b)); a = b
    return out


def Xroot(l, K): return brentq(lambda X: 4 * X - l - 5 * np.log(X) - K, 1.0, 1e9)


def cell(la, lb, S1, Si1, sig_override=None):
    s_hat = (np.exp(-la) if la < 690 else 1e-300) if sig_override is None else sig_override
    G = G_of(la); F = 1.02 * G_of(lb); x_lo = 1.0; steps = []
    for k in range(60):
        c = B.chain(s_hat, lb, G, ETA, F, x_lo, S1)
        if not c['ok']: return {'la': la, 'lb': lb, 'ok': False, 'why': c.get('why', 'condition'), 'steps': steps, 'fail': {kk: v for kk, v in c.items() if not isinstance(v, list)}}
        Fn = 0.9 * c['F_int']; steps.append((F, x_lo, Fn))
        U0n = Fn * Fn / (4 * G * G) - 0.25; x_new = max(1.0, U0n / Si1, U0n - S1)
        if Fn <= F * (1 + 1e-13) and x_new <= x_lo * (1 + 1e-13): break
        F = max(F, Fn); x_lo = max(x_lo, x_new)
    th = 1 / (1 - ETA * G * s_hat)
    Fdef = ETA * G * s_hat * th + c['dH'] + c['rC'] + c['rw']; Fint = max(0.0, 1 - Fdef * (1 + 1e-9))
    K = c['K'] + 1e-9; cc = S1 + 0.25
    Xa = Xroot(la, K) * (1 + 1e-12); Xb = c['X']
    d = 4 * cc - 5 - 5 * cc / Xa + 5 * np.log(Xa) + K
    Bpart = Fint * np.sqrt(la / (Xa + cc)) / 2 if d >= 0 else Fint * np.sqrt(la / (Xb + cc)) / 2
    Gpart = min(0.134 * np.sqrt(la), 1.2)
    L = min(Gpart, Bpart) * (1 - 1e-12)
    return {'la': la, 'lb': lb, 'ok': True, 'G': G, 'F0': 1.02 * G_of(lb), 'n_steps': len(steps), 'x_lo': x_lo, 'K': K, 'Xa': Xa, 'Xb': Xb, 'd_sign': d,
            'Fdef': Fdef, 'F_int': Fint, 'Bpart': Bpart, 'Gpart': Gpart, 'L_cell': L, 'c1_needed': (1 - L) * la / np.log(la),
            'beta': c['beta'], 'omega': c['omega'], 'phi1': c['phi1'], 'nu': c['nu'], 'caseB_margin': c['sigma_caseB_limit'] / s_hat,
            'Lgt1_margin': c['W_p_forced_if_L_gt_1'] / c['Wp_max'], 'X_minus_quarter_l': Xa - la / 4}


if __name__ == '__main__':
    part, nparts = (int(_ARGV[1]), int(_ARGV[2])) if len(_ARGV) > 2 else (0, 1)
    S1 = B.S1_P(); Si1 = B.Si_at(1.0); C = cells(); mine = [(i, ab) for i, ab in enumerate(C) if i % nparts == part]
    fn = 'results/bh_cert_cells_part%d.json' % part; R = json.load(open(fn)) if os.path.exists(fn) else {}      # resumable
    t0 = time.time()
    for i, (a, b) in mine:
        if str(i) in R: continue
        r = cell(a, b, S1, Si1); r['i'] = i; R[str(i)] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
        print('cell %3d l in [%.3f, %.3f] ok %s  L %.6f  c1 %.4f  steps %d  d %+.2f  %.0fs' % (i, a, b, r['ok'], r.get('L_cell', 0), r.get('c1_needed', 0), r.get('n_steps', 0),
              r.get('d_sign', 0), time.time() - t0), flush=True)
    json.dump({'S1_P': S1, 'S_i_at_1': Si1, 'n_cells_total': len(C), 'ratio': RATIO}, open('results/bh_cert_cells_meta.json', 'w'), indent=1)
