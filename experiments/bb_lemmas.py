"""Stage BB §2: numerical checks of Lemma BB1 (the decorrelation dichotomy) and Lemma BB2 (Corollary AR3 with explicit remainders on the
product subclass C0 of the common-envelope class).  Proofs: paperF/sections_F_material.md §7.21.

Lemma BB1 (every state, |eps| < 1):  theta := E[s_v^2 phi]/J_vv = (1 - delta_c) Phi  (the bending seen by the v-information)  =>
   g >= (1 - |eps|) theta/(2 sigma);  equivalently  delta_c <= 1 - c  =>  g >= (1 - |eps|) c Phi/(2 sigma).
Lemma BB2 (class C1 = class C with one modulation wavenumber k; C0 = no modulation):  with A = sigma^2 J_ww, B = sigma^2 J_vv, S = Lw^2 + sigma^2,
   lam = Lw^2/S, beta = lam/sqrt2, tau = beta^2 S/2 = Lw^4/(4S), kappa = k lam/sigma (the blurred modulation frequency),
   Z1 = sum_{n>=1} e^{-n^2 kappa^2 S/2},  Z3 = sum_{n>=1} (e^{-(n kappa - beta)^2 S/2} + e^{-(n kappa + beta)^2 S/2}),
   theta_lo = (1 - e^{-tau} - 2 Z1 - Z3)/(1 + 2 Z1)  <=  theta = E[s_v^2 phi]/J_vv          (exact Fourier bound),
   W2 = E[y_w^2] + (E[x_w^2 (x_v + x_w)^2] + sigma^2 E[(x_v + x_w)^2])/8  (moments of the state; every state:  sigma^2 E[s_w^2 (1 - phi/4)] >= sigma^2/W2),
   =>  either (a) >= 2, or  max((a), (a')) >= (1 - 2 sigma) sqrt(theta_lo/W2)  and  g >= (1 - |eps|) 2 (1 - 2 sigma) sqrt(theta_lo/W2)/sqrt(B).
   On C0, Lw^2 << 1:  2 sqrt(theta_lo/W2) = 1 - O(sigma^2/S + tau + V_v + Lw^2): Corollary AR3 with explicit remainders (eps as in Theorems AY1, AZ1).
-> results/bb_lemmas.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from ar_check import sums, analyse as ar_analyse


def bb1_rows():
    rows = []
    def add(name, sig, g, aN, A, B, eps, Phi=None, dc=None):
        theta = 2 * sig * aN * (A + B) / B; bound = (1 - abs(eps)) * theta / (2 * sig)
        rows.append({'state': name, 'sigma': sig, 'g': g, 'theta': theta, 'bound_BB1': bound, 'ratio': g / bound, 'Phi': Phi, 'delta_c': dc})
    for f in sorted(glob.glob('results/ba_2pi_opt_*.json')):
        O = json.load(open(f)); b = O['best_branches']; sig = float(f.split('_')[-1][:-5])
        add('2pi window (BA)', sig, b['g'], b['a_N'], b['sig2Jww'], b['sig2Jvv'], b['eps'], b['Phi'])
    for f in sorted(glob.glob('results/ba_window_*_*.json')):
        O = json.load(open(f)); sig = float(f.split('_')[-1][:-5]); kind = f.split('_')[-2]
        for lab, b in (('%s (BA)' % kind, O['best_branches']), ('strip optimum', O['strip_branches'])):
            if lab == 'strip optimum' and kind == 'chirp': continue
            add(lab, sig, b['g'], b['a_N'], b['sig2Jww'], b['sig2Jvv'], b['eps'], b['Phi'])
    R = json.load(open('results/ar_check.json'))
    for k, r in R.items():
        aN = (1 - r['delta_c']) * r['Phi'] * r['sig2Jvv'] / (2 * r['sigma'] * (r['sig2Jww'] + r['sig2Jvv']))
        add('class C: ' + k, r['sigma'], r['g'], aN, r['sig2Jww'], r['sig2Jvv'], r['eps'], r['Phi'], r['delta_c'])
    return rows


def zsums(kappa, beta, S, nmax=200):
    if kappa <= 0: return 0.0, 0.0
    n = np.arange(1, nmax + 1); e = lambda x: np.exp(-np.minimum(x, 1e300))
    return float(np.sum(e(n ** 2 * kappa ** 2 * S / 2))), float(np.sum(e((n * kappa - beta) ** 2 * S / 2) + e((n * kappa + beta) ** 2 * S / 2)))


def bb2_bound(sig, Lw, Vv, k, A, B, eps):
    S = Lw ** 2 + sig ** 2; lam = Lw ** 2 / S; beta = lam / np.sqrt(2); tau = Lw ** 4 / (4 * S); kappa = k * lam / sig
    Z1, Z3 = zsums(kappa, beta, S); th_lo = (-np.expm1(-tau) - 2 * Z1 - Z3) / (1 + 2 * Z1)
    W2 = S + (Lw ** 2 * Vv + 3 * Lw ** 4 + sig ** 2 * (Vv + Lw ** 2)) / 8          # exact moments of the C0 part (the modulation adds O(e^{-k^2 Lw^2/2 sigma^2}))
    half = (1 - 2 * sig) * np.sqrt(th_lo / W2); case1 = A / (A + B) >= 2 * sig
    gb = (1 - abs(eps)) * (2.0 if case1 else min(2.0, 2 * half / np.sqrt(B)))
    return {'tau': tau, 'kappa2S': kappa ** 2 * S, 'Z1': Z1, 'Z3': Z3, 'theta_lo': th_lo, 'W2': W2, 'W2_over_S': W2 / S, 'case_a_ge_2': bool(case1),
            'max_branch_bound': 2.0 if case1 else half, 'g_bound': gb, 'mod_moment_factor': float(np.exp(-min(k * k * Lw ** 2 / (2 * sig ** 2), 745.0))) if k > 0 else 0.0}


def c0_state(sig, Lw, kind, par):
    """product states: kind 'lines' (two thin v-lines at +-D sigma) or 'gauss' (Gaussian v-profile of sd Rv)."""
    if kind == 'lines':
        D = par; ev = 1e-3; S = np.diag([(ev * sig) ** 2, Lw ** 2])
        T = [(np.log(0.5), np.array([x * sig, 0.0]), S, np.zeros(2)) for x in (-D, D)]
        vn = np.arange(-D * sig - 8 * sig, D * sig + 8 * sig, min(0.0125, 0.1 / D) * sig); Vv = (D * sig) ** 2 + (ev * sig) ** 2
    else:
        Rv = par; T = [(0.0, np.zeros(2), np.diag([Rv ** 2, Lw ** 2]), np.zeros(2))]
        sv = np.sqrt(Rv ** 2 + sig ** 2); vn = np.linspace(-8 * sv, 8 * sv, 1601); Vv = Rv ** 2
    wn = np.linspace(-7 * np.sqrt(Lw ** 2 + sig ** 2), 7 * np.sqrt(Lw ** 2 + sig ** 2), 1201)
    return T, vn, wn, Vv


def branches_of(r):
    A, B, s = r['sig2Jww'], r['sig2Jvv'], r['sigma']
    a = 2 * A * (1 - r['Phi'] / 4) * (1 - r['delta_w']) / (s * (A + B)); ap = (1 - r['delta_c']) * r['Phi'] * B / (2 * s * (A + B))
    theta = (1 - r['delta_c']) * r['Phi']
    return a, ap, theta


def bb2_rows():
    rows = []
    cases = [(1e-4, 2 * np.sqrt(1e-4), 'lines', 8.0), (1e-4, 0.05, 'lines', 8.0), (1e-4, 0.3, 'lines', 8.0), (1e-4, 1.0, 'lines', 8.0), (1e-4, 1.4, 'lines', 8.0),
             (1e-4, 0.1, 'lines', 3.0), (1e-6, 2e-3, 'lines', 8.0), (1e-6, 0.2, 'lines', 8.0),
             (1e-4, 0.1, 'gauss', 0.05), (1e-4, 0.1, 'gauss', 0.3), (1e-4, 0.5, 'gauss', 1.0), (1e-4, 1.0, 'gauss', 2.0), (1e-6, 0.01, 'gauss', 1e-3)]
    for sig, Lw, kind, par in cases:
        T, vn, wn, Vv = c0_state(sig, Lw, kind, par)
        E = sums(T, sig, vn, wn); r = ar_analyse(E, sig, Lw ** 2); r['sigma'] = sig
        a, ap, theta = branches_of(r); b = bb2_bound(sig, Lw, Vv, 0.0, r['sig2Jww'], r['sig2Jvv'], r['eps'])
        rows.append({'class': 'C0', 'state': '%s %g' % (kind, par), 'sigma': sig, 'Lw': Lw, 'V_v': Vv, 'A': r['sig2Jww'], 'B': r['sig2Jvv'], 'g': r['g'], 'a': a, 'a_prime': ap,
                     'theta': theta, 'T': 2 * np.sqrt(a * ap), 'max_branch': max(a, ap), 'eps': r['eps'], **b})
    # class C1: the modulated states of stage AR (single wavenumber k)
    from an_multipole import realise
    M = json.load(open('results/an_multipole.json')); R = json.load(open('results/ar_check.json'))
    for name, r in R.items():
        sig, Lr, k, n, D = r['sigma'], r['Lw_over_sqrt_sigma'], r['k'], r['n'], r['D']; Lw = Lr * np.sqrt(sig)
        if n is None: xs, lw = np.array([-D, D]), np.zeros(2); kk = 0.0
        else:
            L = M['leading'][n]; d, z = {'mp_n2_D3_k1': (0.5, 0.5)}.get(name, (0.5, 1.0)); xs, lw, f = realise(D, L['alpha'], np.array(L['p']), d, z); kk = k
        w = np.exp(lw - lw.max()); w /= w.sum(); Vv = float(np.sum(w * (xs * sig) ** 2)) + (1e-3 * sig) ** 2
        a, ap, theta = branches_of(r); b = bb2_bound(sig, Lw, Vv, kk, r['sig2Jww'], r['sig2Jvv'], r['eps'])
        rows.append({'class': 'C1' if kk > 0 else 'C0', 'state': name, 'sigma': sig, 'Lw': Lw, 'V_v': Vv, 'A': r['sig2Jww'], 'B': r['sig2Jvv'], 'g': r['g'], 'a': a, 'a_prime': ap,
                     'theta': theta, 'T': 2 * np.sqrt(a * ap), 'max_branch': max(a, ap), 'eps': r['eps'], **b})
    for q in rows:
        print('%s %-16s sig %g Lw %.3g: g %.5f >= %.5f | max(a,a\') %.5f >= %.5f | theta %.4e >= theta_lo %.4e | W2/S %.5f  case(a)>=2: %s' % (
            q['class'], q['state'], q['sigma'], q['Lw'], q['g'], q['g_bound'], q['max_branch'], q['max_branch_bound'], q['theta'], q['theta_lo'], q['W2_over_S'], q['case_a_ge_2']), flush=True)
    return rows


if __name__ == '__main__':
    out = {'BB1': bb1_rows()}
    for r in out['BB1']: print('BB1 %-28s sig %-6g g %-12.6g theta %.3e bound %.6g  g/bound %.4f' % (r['state'], r['sigma'], r['g'], r['theta'], r['bound_BB1'], r['ratio']))
    out['BB2'] = bb2_rows()
    json.dump(out, open('results/bb_lemmas.json', 'w'), indent=1, default=float)
