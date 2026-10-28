"""Stage AP, goal A: confined hidden multipoles — how large can Lambda (and xbar = Lambda/4) be under (H1), (H2), and what is the TRUE gap?

States (physical units, top of the pendulum at the origin, (v, w) the paper's coordinates, z = (v + w)/sqrt2):
  two thin w-lines at v = -D sigma, +D sigma and hidden rows (an_multipole.realise: order n, spacing d, hiddenness z_h) at v = x_i sigma,
  all on the same Gaussian w-envelope of sd Lw; rows modulated by (1 + f_i cos(k w/sigma)), k = 1 (one harmonic exactly).
(H2): Var X_z = (Var v + Var w)/2 <= pi^2 and (H1): E_Q[y_z^2] <= pi^2 + sigma^2/2 hold for Lw <= 4.40 (sqrt(2) pi = 4.443).
Two-scale formulas (checked by ap_direct.py against a direct 2-d sum and against paper E's own gap evaluator v3_tools):
  sigma^2 J_ww = eps + tJmp, eps = sigma^2/(Lw^2 + sigma^2), tJmp = (k^2 e^{-k^2}/2) e^{-lnMI};
  Lambda = (Jvv eps + Lam_v tJmp)/(eps + tJmp);   xbar_w(sqrt p) = Lambda/4 (Lemma Z1);
  chain bound (paper E, Theorem 2 + Lemma 3, most favourable prefactor and kappa = 1): g >= (sqrt2/pi)/sqrt(Lambda);
  true gap g = gap/sigma = (2 sigma^2 J_ww + N)/(sigma (Jvv + sigma^2 J_ww)), N = (1/2)(1 - e^{-Lw^2/4})(Lw^2/(Lw^2 + sigma^2) - 1 + Jvv)
  (Tweedie for the v-posterior); hence g >= 1/sqrt(Jvv) >= 1 at leading order for every such state (minimise over Lw).
Two line lengths: Lw = 4.40 (the (H2) maximum: largest Lambda) and Lw = 2 sqrt(sigma) (the length minimising the true gap of the lines).
-> results/ap_confined.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np

SIGMAS = [1e-4, 1e-6, 1e-8, 1e-12, 1e-16, 1e-24, 1e-32, 1e-48, 1e-64, 1e-96]
STRIP_GMIN = {1e-4: 1.149, 1e-6: 0.740, 1e-12: 0.335, 1e-24: 0.179, 1e-48: 0.110, 1e-96: 0.073}      # paper E #13 (numerical)
K = 1.0; CK = K * K * np.exp(-K * K) / 2


def strip_gmin(sig):
    if sig in STRIP_GMIN: return STRIP_GMIN[sig]
    L = np.log(1 / sig)
    return 1.02 / np.sqrt(L - 23.4) if sig <= 1e-24 else None      # the paper's fit (sigma <= 1e-24)


def evaluate(lib, sig, Lw):
    eps = sig ** 2 / (Lw ** 2 + sig ** 2); N0 = -0.5 * np.expm1(-Lw ** 2 / 4)
    best = None
    for r in lib:
        tJ = CK * np.exp(-r['lnMI']); Jww = eps + tJ
        Lam = (r['Jvv'] * eps + r['Lam_v'] * tJ) / Jww
        if best is None or Lam > best['Lambda']:
            N = N0 * (Lw ** 2 / (Lw ** 2 + sig ** 2) - 1 + r['Jvv']); g = (2 * Jww + N) / (sig * (r['Jvv'] + Jww))
            best = {'Lambda': Lam, 'xbar': Lam / 4, 'g_chain': np.sqrt(2) / np.pi / np.sqrt(Lam), 'g_true': g, 'n': r['n'], 'D': r['D'], 'd': r['d'], 'z': r['z'],
                    'mp_share_Jww': tJ / Jww, 'Lam_v': r['Lam_v'], 'lnMI': r['lnMI']}
    # the lines alone (no multipole): Lambda = Jvv = 1, and the true gap
    Nl = N0 * Lw ** 2 / (Lw ** 2 + sig ** 2); best['Lambda_lines_alone'] = 1.0; best['g_lines_alone'] = (2 * eps + Nl) / (sig * (1 + eps))
    best['eps'] = eps; best['ln1_eps'] = float(np.log(1 / eps))
    return best


if __name__ == '__main__':
    lib = []
    for fn in sorted(glob.glob('results/ap_library_*.json')): lib += json.load(open(fn))
    print('library rows:', len(lib), ' orders:', sorted({r['n'] for r in lib}))
    out = {'library_rows': len(lib), 'orders': sorted({r['n'] for r in lib}), 'table': {}}
    for tag, Lwf in (('H2max', lambda s: 4.40), ('gapmin', lambda s: 2 * np.sqrt(s))):
        rows = {}
        for sig in SIGMAS:
            b = evaluate(lib, sig, Lwf(sig)); L = np.log(1 / sig); b['ln1_sigma'] = L; b['Lw'] = Lwf(sig)
            b['ln1_sig2Jww'] = float(np.log(1 / (b['eps'] + CK * np.exp(-b['lnMI'])))); b['Lambda_over_ln1_sig2Jww'] = b['Lambda'] / b['ln1_sig2Jww']   # Z2-F' compares with 4
            b['Lambda_over_ln'] = b['Lambda'] / L; b['xbar_over_ln'] = b['xbar'] / L; b['strip_gmin'] = strip_gmin(sig)
            rows['%g' % sig] = b
            print('%-6s sigma %-6g: Lambda %8.2f (Lambda/ln %.3f, Lambda/ln(1/s2J) %.3f, xbar/ln %.3f) n %d D %g | chain bound g >= %.4f | true g %.4g (lines alone %.4g) | strip g_min %s' % (
                tag, sig, b['Lambda'], b['Lambda_over_ln'], b['Lambda_over_ln1_sig2Jww'], b['xbar_over_ln'], b['n'], b['D'], b['g_chain'], b['g_true'], b['g_lines_alone'], '%.3f' % b['strip_gmin'] if b['strip_gmin'] else '-'), flush=True)
        out['table'][tag] = rows
        Ls = np.array([rows['%g' % s]['ln1_sigma'] for s in SIGMAS]); La = np.array([rows['%g' % s]['Lambda'] for s in SIGMAS])
        sel = Ls >= np.log(1e16); slope = np.polyfit(np.log(Ls[sel]), np.log(La[sel]), 1)[0]
        out['table'][tag + '_exponent_Lambda_vs_ln1_sigma_(sigma<=1e-16)'] = float(slope)
        print('  %s: local exponent of Lambda_max in ln(1/sigma) (sigma <= 1e-16): %.3f' % (tag, slope))
    # the TRUE gap minimised over the whole family (all library rows) and over the line length Lw (grid in units of sqrt(sigma))
    gopt = {}
    for sig in SIGMAS:
        best = None
        for Lr in np.exp(np.linspace(np.log(0.25), np.log(16.0), 49)):
            Lw = Lr * np.sqrt(sig); eps = sig ** 2 / (Lw ** 2 + sig ** 2); N0 = -0.5 * np.expm1(-Lw ** 2 / 4)
            cand = [(1.0, 0.0, 1.0, None)] + [(r['Jvv'], CK * np.exp(-r['lnMI']), r['Lam_v'], r) for r in lib]
            for Jvv, tJ, Lv, r in cand:
                Jww = eps + tJ; N = N0 * (Lw ** 2 / (Lw ** 2 + sig ** 2) - 1 + Jvv); g = (2 * Jww + N) / (sig * (Jvv + Jww))
                if best is None or g < best['g_true_min']:
                    best = {'g_true_min': g, 'Lw_over_sqrt_sigma': Lr, 'Jvv': Jvv, 'with_multipole': r is not None, 'Lambda_there': (Jvv * eps + Lv * tJ) / Jww,
                            'n': r['n'] if r else 0, 'D': r['D'] if r else None}
        best['bound_1_over_sqrt_Jvv'] = 1 / np.sqrt(best['Jvv']); best['strip_gmin'] = strip_gmin(sig); gopt['%g' % sig] = best
        print('g-min over the family, sigma %-6g: g = %.4f (multipole used: %s, n %d, Lw = %.2f sqrt(sigma)) | strip g_min %s' % (
            sig, best['g_true_min'], best['with_multipole'], best['n'], best['Lw_over_sqrt_sigma'], '%.3f' % best['strip_gmin'] if best['strip_gmin'] else '-'), flush=True)
    out['g_min_family'] = gopt
    json.dump(out, open('results/ap_confined.json', 'w'), indent=1, default=float)
