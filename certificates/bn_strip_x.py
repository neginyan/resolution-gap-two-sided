"""Stage BN: strips (one harmonic) without (P5) x >= 1 — the range 1e-12 <= x < 1 (computer-assisted proof).

Claim (Lemma BN1).  Let rho be a strip, rho = N(v;Lv^2) N(w;Lw^2)(1 + c e^{-w^2/2l^2} cos(k0 v/sigma)), 0 < c <= 1, with (P0), (P3), (P4), (P6), (P7)
((P1), (P2) hold for one harmonic), and 1e-12 <= x < 1.  Then g >= G(l) for every G(l) <= 0.134.
Hence Theorem 9 holds for strips with (P5) replaced by x >= 1e-12: a strip with g < G has x >= 1, where the v1.1 chain applies unchanged
(same 421 cells, same c1 = 1.7).
Proof (by contradiction, g < G, c = G/0.9 <= c* = 0.134/0.9; then (a) < c and (a') < c by the branch identity):
 (1) Lemma BK1 for strips at x < 1 (part 1 below): tau = k'^2 V > 80.  Same cell inequality as bk_narrow_lemma.cell with one harmonic
     (the harmonic sums of a strip are those of n = 1 only); cells cover {1e-12 <= x <= 1, x/80 <= b <= 1}.
 (2) Lemma BJ1 for strips at x < 1 (part 2): A1 A2 A3 >= A_min, i.e. F_c >= 0.9 A_min.  bl_bj1.constants with the strip functions
     (eta_p = Z2 = 0) on x-cells covering [1e-12, 1]; step 2(i) is redone with the per-x bound q < C_q Gamma1(x)/x (Gamma1(x)/x is unbounded as
     x -> 0, but Gamma1(x) q* <= C_q (sqrt(x) A + B)^2 stays bounded).  The theta-remainder of Lemma BL3 (form A) needs E_R <= eps_in = 1e-6 on the
     range: E_R(x, tau) has factors 1/sqrt(x) and sigma/x; its supremum over x in [1e-12, 1], tau >= 80, is computed on cells (part 3).
 (3) Theorem 1 (window inequality) with Lemma BL5 and AX4-P (one harmonic: u2 = u0, S_per <= 8 kappa2 u/(8 + u/(1 + Qbar)) <= 8 kappa2 (1 + Qbar)):
     k_eff^2 <= (1 + eps)(x S_per + b/4) < (1 + eps)(8 kappa2 (1 + Qbar) + 1/4), so g >= F_c/(2 k_eff) >= 0.9 A_min/(2 k_eff) > 0.134: contradiction.
Paper v1.2 numbering: part 1 = Lemma 29(i); part 2 is superseded by Lemma 33 (bp_strip_p5.py, x down to 0) and kept as the first version.
-> results/bn_strip_x.json (and results/bn_bk1_strip.jsonl, one line per x-cell, resumable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np

X_LO, X_HI, TAU0 = 1e-12, 1.0, 80.0
NX, NB = 480, 241
EPS = 1e-6
FN = 'results/bn_bk1_strip.jsonl'


# ---------------------------------------------------------------- part 1: Lemma BK1 for strips, x in [1e-12, 1]
def bk1_strip(resume=True):
    import bk_narrow_lemma as BK
    BK.NN = np.array([1.0])                                                   # one harmonic
    xs = np.geomspace(X_LO, X_HI, NX + 1)
    have = {}
    if resume and os.path.exists(FN):
        for line in open(FN):
            if line.strip():
                try: d = json.loads(line); have[d['j']] = d
                except json.JSONDecodeError: pass
    for j, (xa, xb) in enumerate(zip(xs[:-1], xs[1:])):
        if j in have: continue
        d = bk1_strip_row(j, xa, xb)
        with open(FN, 'a') as f: f.write(json.dumps(d) + '\n')
        have[j] = d
    return [have[j] for j in sorted(have)]


def bk1_strip_row(j, xa, xb):
    """one x-cell of part 1 (all its b-cells); expects bk_narrow_lemma.NN = [1]."""
    import bk_narrow_lemma as BK
    bs = np.geomspace(xa / TAU0, 1.0, NB); wm = np.inf; ok = True
    for ba, bb in zip(bs[:-1], bs[1:]):
        r = BK.cell(xa, xb, ba, bb); ok &= r['ok']; wm = min(wm, r['margin'])
    return {'j': j, 'xa': xa, 'xb': xb, 'cells': NB - 1, 'ok': bool(ok), 'margin': float(wm)}


# ---------------------------------------------------------------- part 2: Lemma BJ1 for strips, x in [1e-12, 1]
def strip_bj1(sig=1e-24, c=None, eps=EPS):
    import bj_lemma as J
    J.eta = lambda p, x: 0.0; J.Z2 = lambda x: 0.0                             # one harmonic
    J.CELLS = np.concatenate([np.geomspace(X_LO, 0.01, 4001), np.linspace(0.01, 1.0, 4001)[1:]])
    import bl_bj1 as BL
    c = BL.CSTAR if c is None else c
    r = BL.constants(sig, c, eps)
    # step 2(i) redone with the per-x bound (the generic code used R1 = Gamma1(1), the supremum of Gamma1(x)/x on x >= 1 only)
    QS, UP, DN = J.QS, J.UP, J.DN; kap = (1 + eps) / (1 - eps); th = 1 / (1 - c * sig)
    s2g = np.sqrt(3 * sig ** 4 + 2 * sig); ep = r['step2b']['e_prime']
    Cq = 4 * (1 + QS) * s2g * (1 + eps) / ((1 - 3 * eps) * ep) * UP
    def prod(a, b):                                                          # sup over [a, b] of Gamma1(x) min(C_q Gamma1(x)/x, m(x))
        A, B = J.Gam1_parts(a)                                                 # strip: A, B constants
        return min(Cq * (np.sqrt(b) * A + B) ** 2 * UP, J.Gam1_up(a, b) * J.m(a))
    gst = J.sup_cells(prod)
    aw2b = (1 - eps) * ep / (4 * th * (1 + eps) * gst * (1 + sig ** 3)) * DN
    r['step2b'] = {'e_prime': ep, 'C_q': Cq, 'gamma_star': gst, 'a_lower_over_c': aw2b / c, 'contradiction': bool(aw2b >= c),
                   'note': 'per-x bound q < C_q Gamma1(x)/x; Gamma1 q* <= C_q (sqrt x A + B)^2'}
    r['ok'] = bool(r['step2b']['contradiction'] and r['step3a']['contradiction'] and r['step3b']['contradiction'] and r['A_min'] >= 1.02 * c and r['A_min'] > 0.93)
    r['x_range'] = [X_LO, X_HI]
    return r, BL


# ---------------------------------------------------------------- part 3: E_R of Lemma BL3 on the strip range
def E_R_sup(sig=1e-24, tau=TAU0):
    import bl_theta as T
    import bj_lemma as J
    xs = np.geomspace(X_LO, X_HI, 4001); worst = 0.0; Q = 0.25; N1 = 0.25
    for a, b in zip(xs[:-1], xs[1:]):
        nu = T.nus(tau, sig * tau / np.sqrt(2 * a))                          # beta <= sigma tau/sqrt(2x), largest at the left end
        gam = 0.25 * J.Gam1_up(a, b)                                         # gamma_P <= q Gamma1(x) (strip), increasing in x
        cc = (np.sqrt(b) * N1 / (1 - Q) + np.sqrt(2 / np.pi) + Q * np.exp(-0.5) / (1 - Q)) / np.sqrt(2)
        # every 1/x, 1/sqrt x factor at the left end a, increasing factors at b
        e = (0.375 * sig * tau * Q * nu[0] / a + (1 + Q) * (1 + Q / 2) * cc * nu[0] + Q * sig * tau ** 2 * nu[2] / (8 * a)
             + tau ** 2 * N1 ** 2 * nu[1] / (np.sqrt(2 * a) * (1 - Q)) + sig * tau * nu[0] * (1 + Q) * gam / a
             + tau * nu[1] * (N1 * cc + (1 + Q) * gam / np.sqrt(2 * a)) + tau * Q * nu[2] * (1 + Q) * cc / 2 + Q * sig * tau ** 3 * nu[4] / (8 * a)
             + sig * tau * Q * nu[2] / (2 * (1 - nu[0])))
        worst = max(worst, e)
    return worst


def keff_conclusion(A_min, eps=EPS):
    Q, rho = 0.25, 0.125
    kap2 = (1 + rho + Q / (2 * (1 - Q))) ** 2 / (8 * (1 - Q))
    Smax = 8 * kap2 * (1 + Q)
    k2 = (1 + eps) * (Smax * X_HI + 0.25)
    g_lo = 0.9 * A_min / (2 * np.sqrt(k2))
    return {'kappa2': kap2, 'S_per_max': Smax, 'k_eff2_max': k2, 'g_lower': g_lo, 'G_max': 0.134, 'ratio': g_lo / 0.134, 'contradiction': bool(g_lo > 0.134)}


if __name__ == '__main__':
    np.seterr(all='ignore')
    rows = bk1_strip()
    print('part 1, Lemma BK1 for strips: %d x-cells x %d b-cells on [1e-12, 1]; all close %s; smallest margin %.3f' % (
        len(rows), NB - 1, all(d['ok'] for d in rows), min(d['margin'] for d in rows)), flush=True)
    out = {'bk1_strip': {'x_cells': len(rows), 'b_cells_per_x': NB - 1, 'cells': len(rows) * (NB - 1), 'all_ok': all(d['ok'] for d in rows),
                         'worst_margin': min(d['margin'] for d in rows), 'worst_x': min(rows, key=lambda d: d['margin'])['xa']}}
    out['E_R_sup'] = E_R_sup(); print('part 3, E_R on [1e-12, 1], tau >= 80: %.2e (needs <= eps_in = 1e-6)' % out['E_R_sup'], flush=True)
    for e in (24, 48, 100, 300):
        r, BL = strip_bj1(10.0 ** (-e)); out['bj1_strip_1e-%d' % e] = r
        print('part 2, BJ1 for strips at sigma 1e-%d: ok %s | A_min %.6f | 2(i) (a)/c >= %.2e | 3(i) margin %.4f | 3(ii) ratio %.2e, x in [%.3f, %.3f]' % (
            e, r['ok'], r['A_min'], r['step2b']['a_lower_over_c'], r['step3a']['margin'], r['step3b']['W_lower_over_W_cap'], *r['step3b']['admissible_x']), flush=True)
    out['step3a_interval'] = BL.step3a_interval()
    out['conclusion'] = keff_conclusion(out['bj1_strip_1e-24']['A_min'])
    print('part 2, step 3(i) in interval arithmetic:', out['step3a_interval'])
    print('conclusion: k_eff^2 <= %.4f, g >= 0.9 A_min/(2 k_eff) = %.4f > 0.134 (ratio %.3f): %s' % (
        out['conclusion']['k_eff2_max'], out['conclusion']['g_lower'], out['conclusion']['ratio'], out['conclusion']['contradiction']))
    out['all_ok'] = bool(out['bk1_strip']['all_ok'] and out['E_R_sup'] <= EPS and all(out['bj1_strip_1e-%d' % e]['ok'] for e in (24, 48, 100, 300))
                         and out['step3a_interval']['g_gt_rhs'] and out['step3a_interval']['unimodal'] and out['conclusion']['contradiction'])
    print('Lemma BN1 (strips, 1e-12 <= x < 1):', out['all_ok'])
    json.dump(out, open('results/bn_strip_x.json', 'w'), indent=1, default=float)
