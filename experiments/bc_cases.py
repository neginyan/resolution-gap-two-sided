"""Stage BC §2: the three-case split — the case quantities, the proved bounds in each case, and the boundary states.

Case quantities (computable for every state; exact point fields of paper E's evaluator):
  theta = E[s_v^2 phi]/J_vv = (1 - delta_c) Phi        (bending felt by the v-information)
  Q     = 4 theta / W2,  W2 = E[y_w^2] + (E[x_w^2 (x_v + x_w)^2] + sigma^2 E[(x_v + x_w)^2])/8   (share of the envelope's own bending that the
          v-information feels; Q ~ (Lw^2/S)^2 ~ 1 on the common-envelope class C1, Q << 1 when the v-information hides in a narrow window)
  CR    = J_ww W2 (w-information over the global Cramer-Rao value; ~1 for one w-scale, >> 1 for two w-scales)
  Lambda = 4 k_eff^2 (the v-fineness of the w-gradient, paper E).
Proved bounds (every state; eps = paper E's remainder):
  BB1 : g >= (1 - |eps|) theta/(2 sigma)
  BC1 : g >= (1 - |eps|) min(2, (1 - 2 sigma) sqrt(Q)/sqrt(sigma^2 J_vv))     [the BB2 argument without the class-C1 step: steps (3), (4) hold for every state]
  AT2 : g >= (a/den)(1 - eps)|C_w| sqrt((1 - Phi/4)(1 - delta_w))/(2 k_eff)   [stage AT]
Boundary families (sigma = 1e-6, 1e-12):
  F1 'widen' : the strip optimum with the window widened, ell -> f ell (Lw fixed, f = 1 ... Lw/ell): from two w-scales (case 3) to one (case 2);
               at each f the gap is re-minimised over (k0, lam-independent ell factor fixed, c, Lv) by a short Nelder-Mead;
  F2 'rows'  : the strip with hidden signed multipole rows in the window (stage AQ H-A; n = 1, 2; beta = 1e-3, 1e-2, 0.1): case 3 with extra Lambda;
  F3 'BA'    : the optima of stage BA's candidate families (from their stored branches).
-> results/bc_cases_<family>_<sigma>.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from w_tools import terms_of, w_nodes
from w_full import full_sums
from at_tools import sums_envelope, analyse
from aq_hybrid import rows_terms, multipole
from ba_tools import strip_opt

FN = 'results/bc_cases.json'


def W2_of(sig, Lw, Lv, cw=0.0):
    """moments of a background N(v; 0, Lv^2) N(w; cw, Lw^2) (the ripple and the signed rows carry no net second-moment change beyond
    O(e^{-k^2 Lv^2/2 sigma^2})); about the top (z = 0)."""
    Ew2 = Lw ** 2 + cw ** 2; Ev2 = Lv ** 2
    Ew4 = 3 * Lw ** 4 + 6 * Lw ** 2 * cw ** 2 + cw ** 4; Ew3 = 3 * Lw ** 2 * cw + cw ** 3
    return Ew2 + sig ** 2 + (Ew2 * Ev2 + Ew4 + sig ** 2 * (Ev2 + Ew2)) / 8 + 0 * Ew3


def case_row(sig, r, W2, label, extra=None):
    s2 = sig * sig; A, B = r['sig2Jww'], r['sig2Jvv']; eps = r['eps']
    theta = (1 - r['delta_c']) * r['Phi']; Q = 4 * theta / W2; CR = A * W2 / s2
    bb1 = (1 - abs(eps)) * theta / (2 * sig)
    bc1 = (1 - abs(eps)) * (2.0 if A / (A + B) >= 2 * sig else min(2.0, (1 - 2 * sig) * np.sqrt(Q) / np.sqrt(B)))
    at2 = r['bound_AT2']; ln = np.log(1 / sig)
    best = max(bb1, bc1, at2); which = ['BB1', 'BC1', 'AT2'][int(np.argmax([bb1, bc1, at2]))]
    out = {'label': label, 'sigma': sig, 'g': r['g'], 'theta': theta, 'Q': Q, 'Q_ln': Q * ln, 'CR': CR, 'Lambda': 4 * r['k_eff'] ** 2, 'Lambda_over_ln': 4 * r['k_eff'] ** 2 / ln,
           'B': B, 'eps': eps, 'BB1': bb1, 'BC1': bc1, 'AT2': at2, 'best_bound': best, 'best_which': which, 'best_over_g': best / r['g'],
           'best_sqrt_ln': best * np.sqrt(ln), 'g_sqrt_ln': r['g'] * np.sqrt(ln), 'W2': W2}
    if extra: out.update(extra)
    return out


def analyse_state(sig, bg, win, kb, Lv_sig, extra_terms=None, extra_w=None, nv=48, n0=21):
    mmax = max([w_[3] for w_ in win] + [1])
    T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, 40, 300)
    if extra_terms: T = T + extra_terms
    if extra_w is not None: wn = np.unique(np.concatenate([wn, extra_w]))
    return analyse(sums_envelope(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=nv * mmax, n0=n0), sig)


def fast_g(sig, bg, win, kb, Lv_sig):
    T = terms_of(bg, win, kb, sig, Lv_sig); wn = w_nodes(bg, win, sig, 12, 100)
    o = full_sums(T, sig, kb, np.sqrt((Lv_sig * sig) ** 2 + sig ** 2), wn, nv=16, n0=11, ext=6.5)
    return float(o['gap_over_sigma']) if o['y_over_sigma'] <= 0.5 else 1e300


def family_widen(sig, R):
    P = strip_opt(sig); s = np.sqrt(sig); Lw = P['lam'] * P['M']                # Lw in sqrt(sigma) units
    for f in (1.0, 3.0, 10.0, 30.0, 100.0, P['M']):
        key = 'widen_%g_f%.4g' % (sig, f)
        if key in R: continue
        t0 = time.time()
        def obj(x):
            k0, lam_f, Lv_sig = np.exp(x[:3]); c = 1 / (1 + np.exp(-x[3])); ell = P['lam'] * f * lam_f
            if not (0.3 < k0 < 12 and Lv_sig > 30 / k0 and ell <= 3 * Lw): return 1e300
            try: return fast_g(sig, [(1.0, 0.0, Lw)], [(c, 0.0, ell, 1)], k0, Lv_sig)
            except (ValueError, FloatingPointError, OverflowError, MemoryError, np.linalg.LinAlgError): return 1e300
        x0 = np.array([np.log(P['k0']), 0.0, np.log(P['Lv_sig']), np.log(P['c'] / (1 - P['c']))])
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxfev': 50 if f > 1 else 1, 'xatol': 1e-3, 'fatol': 1e-6})
        k0, lam_f, Lv_sig = np.exp(res.x[:3]); c = 1 / (1 + np.exp(-res.x[3])); ell = P['lam'] * f * lam_f
        r = analyse_state(sig, [(1.0, 0.0, Lw)], [(c, 0.0, ell, 1)], k0, Lv_sig)
        R[key] = case_row(sig, r, W2_of(sig, Lw * s, Lv_sig * sig), 'widened window, Lw/ell = %.3g' % (Lw / ell),
                          {'family': 'F1 widen', 'Lw_over_ell': Lw / ell, 'k0': k0, 'c': c, 'Lv_sig': Lv_sig, 'seconds': time.time() - t0})
        json.dump(R, open(FN, 'w'), indent=1, default=float); pr(R[key])


def family_rows(sig, R):
    P = strip_opt(sig); s = np.sqrt(sig); Lw = P['lam'] * P['M']; ell = P['lam'] * s
    kw = 2 * np.pi * 8 / (ell / sig)          # 8 modulation periods per window width (resolvable at every sigma; kw = 1 at 1e-6 costs ~2.5 h per state)
    for n in (1, 2):
        for beta in (1e-3, 1e-2, 0.1):
            key = 'rows_%g_n%d_b%g' % (sig, n, beta)
            if key in R: continue
            t0 = time.time(); xs, a, f = multipole(n, np.pi / P['k0'], 0.6); a = a / a.sum()
            ex = rows_terms(sig, P, P['k0'], xs, a, f, beta, kw); ew = np.arange(-6 * ell, 6 * ell, 2 * np.pi * sig / kw / 10)
            r = analyse_state(sig, [(1.0, 0.0, Lw)], [(P['c'], 0.0, P['lam'], 1)], P['k0'], P['Lv_sig'], ex, ew, nv=32 * 6, n0=21)
            R[key] = case_row(sig, r, W2_of(sig, Lw * s, P['Lv_sig'] * sig), 'strip + hidden rows n = %d, beta = %g' % (n, beta),
                              {'family': 'F2 rows', 'n': n, 'beta': beta, 'kw': kw, 'seconds': time.time() - t0})
            json.dump(R, open(FN, 'w'), indent=1, default=float); pr(R[key])


def family_ba(sig, R):
    from ba_window import two_state, chirp_state
    s = np.sqrt(sig); sg = '%g' % sig; sg = {'1e-06': '1e-06', '1e-12': '1e-12'}.get(sg, sg)
    for kind in ('two', 'chirp'):
        key = 'ba_%s_%g' % (kind, sig)
        if key in R: continue
        O = json.load(open('results/ba_window_%s_%s.json' % (kind, sg))); x = np.array(min(O['runs'], key=lambda q: q['g_min'])['x'])
        kb, bg, win, Lv_sig, info = (two_state if kind == 'two' else chirp_state)(x, sig)
        r = analyse_state(sig, bg, win, kb, Lv_sig, nv=32); R[key] = case_row(sig, r, W2_of(sig, bg[0][2] * s, Lv_sig * sig), 'stage-BA optimum: %s' % kind, {'family': 'F3 BA'})
        json.dump(R, open(FN, 'w'), indent=1, default=float); pr(R[key])
    key = 'ba_2pi_%g' % sig
    if key not in R:
        O = json.load(open('results/ba_2pi_opt_%s.json' % sg)); b = min(O['runs'], key=lambda q: q['g_min']); W2pi = 2 * np.sqrt(2) * np.pi
        r = analyse_state(sig, [(1.0, (b['c_b'] - W2pi) / s, b['Lw'] / s)], [(b['c'], 0.0, b['lam'], 1)], b['k0'], b['Lv_sig'])
        R[key] = case_row(sig, r, W2_of(sig, b['Lw'], b['Lv_sig'] * sig, b['c_b']), 'stage-BA optimum: window at z = 2 pi', {'family': 'F3 BA'})
        json.dump(R, open(FN, 'w'), indent=1, default=float); pr(R[key])


def pr(q):
    print('%-44s sig %-6g g %-10.5g theta %.2e Q %.2e (Q ln %.3g) CR %.3g Lambda/ln %.3f | BB1 %.3g BC1 %.3g AT2 %.4g -> best %s %.4g (best/g %.3f, best sqrt(ln) %.3f)' % (
        q['label'], q['sigma'], q['g'], q['theta'], q['Q'], q['Q_ln'], q['CR'], q['Lambda_over_ln'], q['BB1'], q['BC1'], q['AT2'], q['best_which'], q['best_bound'],
        q['best_over_g'], q['best_sqrt_ln']), flush=True)


if __name__ == '__main__':
    np.seterr(all='ignore'); part = sys.argv[1]; sig = float(sys.argv[2]) if len(sys.argv) > 2 else None
    FN = 'results/bc_cases_%s_%g.json' % (part, sig); R = json.load(open(FN)) if os.path.exists(FN) else {}
    if part == 'widen': family_widen(sig, R)
    elif part == 'rows': family_rows(sig, R)
    elif part == 'ba': family_ba(sig, R)
