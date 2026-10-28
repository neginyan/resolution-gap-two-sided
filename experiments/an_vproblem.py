"""Stage AN, plan A''': the v-problem of the staggered double row, signed (g) and unsigned (h), exactly and with proved bounds.

Units t = 1, x = y_v - D (centre of the two rows).  Lines at x = -D, +D (unit linear density), rows at x = -d, +d (linear density m each).
  P0(x) = K e^{-x^2/2} Den(x),   Den = cosh(Dx) + beta cosh(dx),   beta = m e^{(D^2 - d^2)/2},   K = 2 (2 pi)^{-1/2} e^{-D^2/2},
  g = (P_+ - P_-)/P0 = -beta sinh(dx)/Den   (signed: difference of the rows' posterior weights),
  h = (P_+ + P_-)/P0 =  beta cosh(dx)/Den   (unsigned: total row weight).
  Rq(f) = int P0 f'^2 / int P0 f^2,   ln(M/I_f) with M = int P0, I_f = int P0 f^2.
Proposition AN3 (proof, see paperF/sections_F_material.md section 7.3): for 0 < beta <= 1, 0 < r = d/D <= 1/4,
  Rq(g) <= (1+beta) D^2 (sqrt Psi_g(r) + 2 beta sqrt Psi_g3(r))^2 / Z_g(D),
  ln(M/I_g) >= D^2/2 + 2 ln(1/beta) + 3 ln D - 2 ln d + ln(sqrt(2 pi)/Phi_g(r)),
  Rq(h) <= (1+beta) D^2 (sqrt Psi_h(r) + 2 beta sqrt Psi_h3(r))^2 / Z_h(D),
  ln(M/I_h) >= D^2/2 + 2 ln(1/beta) + ln D + ln(sqrt(2 pi)/Phi_h(r)),
  with the one-dimensional integrals below; Psi_g(0)/Z_g(inf) = Psi_h(0)/Z_h(inf) = 1/2 exactly, so B -> 1.
Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

LIM = 80.0


def lcosh(u):
    u = np.abs(u); return u + np.log1p(np.exp(-2 * u)) - np.log(2.0)


def exact(D, d, lb, n=None):
    """exact Rq and ln(M/I) for g and h (lb = ln beta).  Simpson on x in [-(D + 12), D + 12], step <= min(0.01, 0.02/D)."""
    L = D + 12.0; hstep = min(0.01, 0.02 / D, 0.02 / max(d, 1e-9)); N = int(np.ceil(2 * L / hstep)); N += N % 2
    x = np.linspace(-L, L, N + 1); hx = x[1] - x[0]
    wS = np.ones(N + 1); wS[1:-1:2] = 4; wS[2:-1:2] = 2; wS *= hx / 3
    lD = np.logaddexp(lcosh(D * x), lb + lcosh(d * x))
    cD = np.exp(lcosh(D * x) - lD); cd = np.exp(lb + lcosh(d * x) - lD)
    T = D * np.tanh(D * x) * cD + d * np.tanh(d * x) * cd                    # Den'/Den
    base = -x ** 2 / 2 - lD + 2 * lcosh(d * x)                                 # log of w cosh^2(dx)/Den
    th = np.tanh(d * x)
    out = {}
    lM = 0.5 * np.log(2 * np.pi) + np.logaddexp(D ** 2 / 2, lb + d ** 2 / 2)
    for name, F, Fp in (('g', th, d - th * T), ('h', np.ones_like(x), d * th - T)):   # f/(beta cosh dx) and (f' Den/beta)/cosh(dx) structure
        with np.errstate(divide='ignore'):
            lI = logsumexp(base + np.log(wS * F ** 2)); lN = logsumexp(base + np.log(wS * Fp ** 2))   # log domain (no underflow)
        out['Rq_' + name] = float(np.exp(lN - lI)); out['lnMI_' + name] = float(lM - 2 * lb - lI)
        out['lI_' + name] = float(2 * lb + lI); out['lD_' + name] = float(2 * lb + lN)
    out['lM'] = float(lM)
    out['kappa_split'] = float(np.exp(out['lI_h'] - out['lI_g']))        # (I_+ + I_-)/I_g up to the factor 1/2 ... see split_loss
    return out


def split_loss(D, d, lb):
    """loss of the 'two hidden masses' route: (I_{h+} + I_{h-})/I_g, h_pm = beta e^{-+dx}/(2 Den)."""
    L = D + 12.0; hstep = min(0.01, 0.02 / D); N = int(np.ceil(2 * L / hstep)); N += N % 2
    x = np.linspace(-L, L, N + 1); hx = x[1] - x[0]; wS = np.ones(N + 1); wS[1:-1:2] = 4; wS[2:-1:2] = 2; wS *= hx / 3
    lD = np.logaddexp(lcosh(D * x), lb + lcosh(d * x)); b = -x ** 2 / 2 - lD
    Ipm = np.sum(wS * np.exp(b + lcosh(2 * d * x))) / 2; Ig = np.sum(wS * np.exp(b + 2 * lcosh(d * x)) * np.tanh(d * x) ** 2)
    return float(Ipm / Ig)


def q(f, lo=-LIM, hi=LIM):
    return quad(f, lo, hi, limit=800, epsabs=1e-13, epsrel=1e-12)[0]


def sr(r, u):  # sinh(r u)/r, with the r -> 0 limit
    return u if r == 0 else np.sinh(r * u) / r


def constants(r, D):
    sech = lambda u: 1 / np.cosh(u)
    C = {}
    C['Psi_g'] = q(lambda u: np.cosh(r * u) ** 2 * ((1 - u * np.tanh(u)) ** 2 + (2 * r * r / 3) * abs(u) ** 3) * sech(u))
    C['Psi_g3'] = q(lambda u: sr(r, u) ** 2 * np.cosh(r * u) ** 2 * sech(u) ** 3)
    C['Z_g'] = q(lambda u: np.exp(-u * u / (2 * D * D)) * u * u * sech(u))
    C['Phi_g'] = q(lambda u: sr(r, u) ** 2 * sech(u))
    C['Psi_h'] = q(lambda u: np.cosh(r * u) ** 2 * np.tanh(u) ** 2 * sech(u))
    C['Psi_h3'] = q(lambda u: np.cosh(r * u) ** 4 * sech(u) ** 3)
    C['Z_h'] = q(lambda u: np.exp(-u * u / (2 * D * D)) * sech(u))
    C['Phi_h'] = q(lambda u: np.cosh(r * u) ** 2 * sech(u))
    return C


def proved(D, d, lb):
    beta = np.exp(lb); r = d / D; assert beta <= 1 and r <= 0.25
    C = constants(r, D)
    o = {'beta': float(beta), 'r': r}
    o['Rq_g_bd'] = (1 + beta) * D ** 2 * (np.sqrt(C['Psi_g']) + 2 * beta * np.sqrt(C['Psi_g3'])) ** 2 / C['Z_g']
    o['Rq_h_bd'] = (1 + beta) * D ** 2 * (np.sqrt(C['Psi_h']) + 2 * beta * np.sqrt(C['Psi_h3'])) ** 2 / C['Z_h']
    o['lowlog_g'] = 2 * (-lb) + 3 * np.log(D) - 2 * np.log(d) + np.log(np.sqrt(2 * np.pi) / C['Phi_g'])
    o['lowlog_h'] = 2 * (-lb) + np.log(D) + np.log(np.sqrt(2 * np.pi) / C['Phi_h'])
    o['lnMI_g_bd'] = D ** 2 / 2 + o['lowlog_g']; o['lnMI_h_bd'] = D ** 2 / 2 + o['lowlog_h']
    o['B_g'] = o['Rq_g_bd'] / (D ** 2 / 2); o['B_h'] = o['Rq_h_bd'] / (D ** 2 / 2)
    o['a_g'] = max(0.0, -o['lowlog_g']); o['a_h'] = max(0.0, -o['lowlog_h'])    # Rq <= B (ln(M/I) + a)
    o.update({k: float(v) for k, v in C.items()})
    return {k: float(v) for k, v in o.items()}


def staggered_lb(D, z=0.1, d=1.0):
    """ln beta of the configured family of al_staggered.py: row linear density m = e^{-D(z + D/2)}/2."""
    return -D * (z + D / 2) - np.log(2) + (D ** 2 - d ** 2) / 2


if __name__ == '__main__':
    out = {}
    # identities: int (1 - u tanh u)^2 sech = pi^3/8 = (1/2) int u^2 sech ;  int tanh^2 sech = pi/2 = (1/2) int sech
    I1 = q(lambda u: (1 - u * np.tanh(u)) ** 2 / np.cosh(u)); I2 = q(lambda u: u * u / np.cosh(u))
    I3 = q(lambda u: np.tanh(u) ** 2 / np.cosh(u)); I4 = q(lambda u: 1 / np.cosh(u))
    out['identities'] = {'int_(1-u tanh)^2 sech': I1, 'pi^3/8': np.pi ** 3 / 8, 'int u^2 sech': I2, 'int tanh^2 sech': I3, 'pi/2': np.pi / 2, 'int sech': I4}
    print(out['identities'])
    # configured family (al_staggered.py, z = 0.1, d = 1) and its large-D continuation
    fam = {}
    for D in (6.0, 8.0, 10.0, 14.0, 20.0, 28.0, 40.0):
        lb = staggered_lb(D); e = exact(D, 1.0, lb); p = proved(D, 1.0, lb); e.update(p); e['split_loss'] = split_loss(D, 1.0, lb)
        e['margin_g'] = e['Rq_g'] - e['lnMI_g']; e['margin_h'] = e['Rq_h'] - e['lnMI_h']
        e['ok_bounds'] = bool(e['Rq_g'] <= e['Rq_g_bd'] and e['lnMI_g'] >= e['lnMI_g_bd'] and e['Rq_h'] <= e['Rq_h_bd'] and e['lnMI_h'] >= e['lnMI_h_bd'])
        fam['%g' % D] = e
        print('D %4g beta %.3g Rq_g %.3f (bd %.3f) ln_g %.3f (bd %.3f) B_g %.3f | Rq_h %.3f (bd %.3f) ln_h %.3f B_h %.3f | split %.1f margin_g %.2f ok %s' % (
            D, e['beta'], e['Rq_g'], e['Rq_g_bd'], e['lnMI_g'], e['lnMI_g_bd'], e['B_g'], e['Rq_h'], e['Rq_h_bd'], e['lnMI_h'], e['B_h'], e['split_loss'], e['margin_g'], e['ok_bounds']), flush=True)
    out['family'] = fam
    # whole staggered v-family, exact: sup of Rq - ln(M/I) (B' = 1) over D, d, beta (hidden and visible rows)
    scan = []
    for D in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        for d in (0.02, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 0.25 * D, 0.5 * D, 0.75 * D, 0.95 * D):
            if d >= D: continue
            for lb in (-40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
                e = exact(D, d, lb)
                scan.append({'D': D, 'd': d, 'lb': lb, 'Rq_g': e['Rq_g'], 'lnMI_g': e['lnMI_g'], 'Rq_h': e['Rq_h'], 'lnMI_h': e['lnMI_h']})
    for s in scan: s['margin_g'] = s['Rq_g'] - s['lnMI_g']; s['margin_h'] = s['Rq_h'] - s['lnMI_h']
    wg = max(scan, key=lambda s: s['margin_g']); wh = max(scan, key=lambda s: s['margin_h'])
    out['scan'] = {'n': len(scan), 'max_margin_g': wg, 'max_margin_h': wh,
                   'max_ratio_g_ln_gt_5': max(s['Rq_g'] / s['lnMI_g'] for s in scan if s['lnMI_g'] > 5),
                   'max_ratio_h_ln_gt_5': max(s['Rq_h'] / s['lnMI_h'] for s in scan if s['lnMI_h'] > 5)}
    print('scan', len(scan), 'max margin g %.4f at' % wg['margin_g'], {k: wg[k] for k in ('D', 'd', 'lb')}, 'max margin h %.4f at' % wh['margin_h'], {k: wh[k] for k in ('D', 'd', 'lb')})
    print('max Rq/ln (ln > 5): g %.4f h %.4f' % (out['scan']['max_ratio_g_ln_gt_5'], out['scan']['max_ratio_h_ln_gt_5']))
    # proved bounds over the hidden range of the theorem (beta <= 1, r <= 1/4)
    pr = []
    for D in (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        for d in (0.1, 0.5, 1.0, 0.25 * D):
            for lb in (-20.0, -5.0, -1.0, 0.0):
                e = exact(D, d, lb); p = proved(D, d, lb)
                pr.append({'D': D, 'd': d, 'lb': lb, 'ok': bool(e['Rq_g'] <= p['Rq_g_bd'] * (1 + 1e-9) and e['lnMI_g'] >= p['lnMI_g_bd'] - 1e-9 and e['Rq_h'] <= p['Rq_h_bd'] * (1 + 1e-9) and e['lnMI_h'] >= p['lnMI_h_bd'] - 1e-9),
                           'B_g': p['B_g'], 'B_h': p['B_h'], 'slack_g': p['Rq_g_bd'] / e['Rq_g'], 'slack_h': p['Rq_h_bd'] / e['Rq_h']})
    out['proved_range'] = {'n': len(pr), 'all_ok': all(s['ok'] for s in pr), 'max_B_g': max(s['B_g'] for s in pr), 'max_B_h': max(s['B_h'] for s in pr),
                           'max_slack_g': max(s['slack_g'] for s in pr), 'rows': pr}
    print('proved range: n %d all bounds hold %s, max B_g %.3f max B_h %.3f' % (len(pr), out['proved_range']['all_ok'], out['proved_range']['max_B_g'], out['proved_range']['max_B_h']))
    json.dump(out, open('results/an_vproblem.json', 'w'), indent=1, default=float)
