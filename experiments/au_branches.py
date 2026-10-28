"""Stage AU: the branches of paper E's numerator on the confined hidden-multipole families of stage AP (two-scale, leading order).

Exact decomposition (paper E, stage X1):  sigma den g = 2 A_w + N_phi + (B + R3),  A_w = sigma^2 E[s_w^2 (1 - phi/4)],  N_phi = (sigma^2/2) E[s_v^2 phi].
Branches (each a lower bound on g up to the common remainder eps):
  (a)  w-term       g >= 2 A_w / (sigma den)                                  (Corollary AT4)
  (a') N-term       g >= N_phi / (sigma den)
  (b)  AT2          g >= (a/den) |C_w| sqrt(A_w/A) / sqrt(Lambda),  Lambda = 4 k_eff^2;  AT2' (delta_w-free): (a/den)|C_w| / sqrt(2 Lambda)
  Theorem 2 (AM-GM of (a) and (a')):  (a/den) sqrt(P_w),  and exactly  (a)(a') = (Theorem 2)^2 / 4  =>  max(a, a') >= Theorem 2 / 2.
Leading order in the common-envelope class (lines + rows modulated with k = 1 on N(0, Lw^2); L^2 = Lw^2 + sigma^2; m_z ~ w/sqrt2):
  A = sigma^2 J_ww = eps + tJ,  eps = sigma^2/L^2,  B = sigma^2 J_vv;  Phi = 1 - e^{-Lw^2/4};
  A_w = eps [3/4 + (1 - Lw^2/2) e^{-Lw^2/4}/4] + tJ (1 - Phi/4);  N_phi = B Phi/2;  C_w = e^{-Lw^2/16}.
(checked against exact point fields in au_exact.py).  -> results/au_branches.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np

SIGMAS = [1e-4, 1e-6, 1e-8, 1e-12, 1e-16, 1e-24, 1e-32, 1e-48, 1e-64, 1e-96]
CK = np.exp(-1.0) / 2


def branches(r, sig, Lw):
    """r: library row (Jvv, lnMI, Lam_v) or None for the lines alone.  Everything in sigma^2 units; log-safe for small sigma."""
    L2 = Lw * Lw + sig * sig; eps = sig * sig / L2
    B = 1.0 if r is None else r['Jvv']; tJ = 0.0 if r is None else CK * np.exp(-r['lnMI']); A = eps + tJ
    Lam = B if r is None else (B * eps + r['Lam_v'] * tJ) / A
    Phi = -np.expm1(-Lw * Lw / 4); Cw = np.exp(-Lw * Lw / 16)
    Aw = eps * (0.75 + (1 - Lw * Lw / 2) * np.exp(-Lw * Lw / 4) / 4) + tJ * (1 - Phi / 4); Nphi = B * Phi / 2; den = A + B
    N_true = -0.5 * np.expm1(-Lw * Lw / 4) * (Lw * Lw / L2 - 1 + B)
    o = {'g_true': (2 * A + N_true) / (sig * den), 'a_w': 2 * Aw / (sig * den), 'a_N': Nphi / (sig * den),
         'AT2': (B / den) * Cw * np.sqrt(Aw / A) / np.sqrt(Lam), 'AT2_dw_free': (B / den) * Cw / np.sqrt(2 * Lam),
         'ThmW': (B / den) * np.sqrt(4 * Aw * Phi / B) / sig, 'Lambda': Lam, 'k_eff2': Lam / 4, 'mp_share': tJ / A}
    o['best'] = max(o['a_w'], o['a_N'], o['AT2']); o['branch'] = max(('a (w-term)', o['a_w']), ("a' (N-term)", o['a_N']), ('b (AT2)', o['AT2']), key=lambda t: t[1])[0]
    o['best_sqrt_ln'] = o['best'] * np.sqrt(np.log(1 / sig)); o['prod_check'] = o['a_w'] * o['a_N'] / (o['ThmW'] ** 2 / 4)
    return o


if __name__ == '__main__':
    lib = []
    for fn in sorted(glob.glob('results/ap_library_*.json')): lib += json.load(open(fn))
    AP = json.load(open('results/ap_confined.json'))['table']; out = {'tables': {}, 'scan': {}}
    for tag, Lwf in (('H2max', lambda s: 4.40), ('gapmin', lambda s: 2 * np.sqrt(s))):
        rows = {}
        for sig in SIGMAS:
            b = AP[tag]['%g' % sig]; r = {'Jvv': None, 'lnMI': b['lnMI'], 'Lam_v': b['Lam_v']}
            row = [x for x in lib if x['n'] == b['n'] and x['D'] == b['D'] and x['d'] == b['d'] and x['z'] == b['z'] and abs(x['lnMI'] - b['lnMI']) < 1e-9][0]
            o = branches(row, sig, Lwf(sig)); o.update({'n': b['n'], 'D': b['D'], 'Lambda_AP': b['Lambda'], 'g_true_AP': b['g_true']}); rows['%g' % sig] = o
            print('%-6s sigma %-6g Lambda %8.2f  g %.4g | (a) %.4g (a\') %.4g (b) AT2 %.4g AT2\' %.4g | Thm2 %.4g | best x sqrt(ln) %.4g by %s' % (
                tag, sig, o['Lambda'], o['g_true'], o['a_w'], o['a_N'], o['AT2'], o['AT2_dw_free'], o['ThmW'], o['best_sqrt_ln'], o['branch']), flush=True)
        out['tables'][tag] = rows
    # worst case over the whole library, the lines alone, and the line length Lw (from 0.25 sqrt(sigma) to the (H2) maximum 4.40)
    for sig in SIGMAS:
        worst = None
        for Lw in np.exp(np.linspace(np.log(0.25 * np.sqrt(sig)), np.log(4.40), 121)):
            for r in [None] + lib:
                o = branches(r, sig, Lw)
                if worst is None or o['best_sqrt_ln'] < worst['best_sqrt_ln']:
                    worst = dict(o); worst.update({'Lw': Lw, 'Lw_over_sqrt_sigma': Lw / np.sqrt(sig), 'row': r})
        # the largest Lambda among states whose best branch is within a factor 10 of the worst
        out['scan']['%g' % sig] = worst
        print('scan sigma %-6g: min over (rows, Lw) of max-branch x sqrt(ln) = %.4f by %s at Lw = %.3g sqrt(sigma) (Lambda %.3g, g %.4g)' % (
            sig, worst['best_sqrt_ln'], worst['branch'], worst['Lw_over_sqrt_sigma'], worst['Lambda'], worst['g_true']), flush=True)
    json.dump(out, open('results/au_branches.json', 'w'), indent=1, default=float)
