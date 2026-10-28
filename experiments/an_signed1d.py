"""Stage AN: the general signed one-dimensional problem.
  nu = finite measure on R, f: R -> [-1, 1], P_u = nu * phi_u, G_u = (f nu) * phi_u, g = G/P = E[f(X) | y] (posterior mean of f),
  I(u) = int P g^2,  D(u) = int P g'^2,  Rq = D/I,  question: t Rq(t) <= A + B ln(M/I(t)) ?
(a) Proposition AN1 (proof): dI/du = -D and  I Rq' = -int P (Lg + Rq g)^2 - int P H g'^2,  Lg = g'' + s g', s = (ln P)', H = -s'.
    Hence if P_u is log-concave on [tau t, t]: (1 - tau) t Rq(t) <= ln(M/I(t)); for log-concave nu: t Rq(t) <= ln(M/I(t)) (B' = 1, A' = 0).
    Checked here by finite differences on random point configurations, and on log-concave nu with random f.
(b) Search: random and optimised point configurations (K <= 6 points, signs/values of f free) for sup [Rq - ln(M/I)] and Rq/ln(M/I).
-> results/an_signed1d.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

LN2PI = np.log(2 * np.pi)
XMAX, LWMAX = 40.0, 400.0          # search box (an_retune.py widens it)


def grid(xs, u, extra=10.0):
    sep = max(1e-3, np.max(xs) - np.min(xs)); h = min(0.01, 0.04 / sep, 0.05 * np.sqrt(u))
    y = np.arange(np.min(xs) - extra * np.sqrt(u) - 2, np.max(xs) + extra * np.sqrt(u) + 2, h)
    return y, h


def slse(la, b):
    """signed log-sum-exp over axis 0: returns (log|sum b e^la|, sign)."""
    with np.errstate(divide='ignore'):
        r, sg = logsumexp(la + np.log(np.abs(b) + 0.0), axis=0, b=np.sign(b), return_sign=True)
    return r, sg


def quantities(xs, lw, f, u=1.0, y=None, need_L=False, dense=False):
    """point measure nu = sum e^{lw_i} delta_{x_i}; exact posterior formulas on a grid, fully in the log domain (signed sums), chunked
    over y (<= 2e6 entries).  g = E[f|y], g' = Cov(X, f|y)/u = sum pi_i f_i (x_i - E X)/u, g'' = sum pi_i f_i [(x_i - E X)^2 - V]/u^2."""
    if y is None:
        if dense: y = np.arange(np.min(xs) - 10 * np.sqrt(u) - 2, np.max(xs) + 10 * np.sqrt(u) + 2, 0.02 * np.sqrt(u)); h = y[1] - y[0]
        else: y, h = grid(xs, u)
    else: h = y[1] - y[0]
    K = len(xs); ch = max(1, int(2e6 // K)); acc = {k: [] for k in ('lP', 'lg', 'sg', 'lgp', 'sgp', 'lgpp', 'sgpp', 's', 'H')}
    F = f[:, None]
    for i0 in range(0, len(y), ch):
        yy = y[i0:i0 + ch]
        la = lw[:, None] - (yy[None] - xs[:, None]) ** 2 / (2 * u) - 0.5 * (LN2PI + np.log(u)); lP = logsumexp(la, 0); W = la - lP[None]
        pi = np.exp(W); Ex = pi.T @ xs; V = pi.T @ xs ** 2 - Ex ** 2; dx = xs[:, None] - Ex[None]
        lg, sg = slse(W, np.broadcast_to(F, W.shape)); lgp, sgp = slse(W, F * dx); lgpp, sgpp = slse(W, F * (dx ** 2 - V[None]))
        for k, v in (('lP', lP), ('lg', lg), ('sg', sg), ('lgp', lgp - np.log(u)), ('sgp', sgp), ('lgpp', lgpp - 2 * np.log(u)), ('sgpp', sgpp),
                     ('s', (Ex - yy) / u), ('H', 1 / u - V / u ** 2)):
            acc[k].append(v)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    wS = np.full_like(y, h); wS[0] = wS[-1] = h / 2
    lPw = A['lP'] + np.log(wS); lM = float(logsumexp(lw))
    lI = float(logsumexp(lPw + 2 * A['lg'])); lD = float(logsumexp(lPw + 2 * A['lgp']))
    # Lambda_v = int P (s g + 2 g')^2 / int P g^2  (the v-part of paper E's Lambda after the theta lift); factor e^{lg} out
    with np.errstate(over='ignore', invalid='ignore'):
        T = A['sg'] * A['s'] + 2 * A['sgp'] * np.exp(A['lgp'] - A['lg'])
        lL = float(logsumexp(lPw + 2 * A['lg'] + 2 * np.log(np.abs(T) + 1e-300)))
    o = {'Rq': float(np.exp(lD - lI)), 'lnMI': lM - lI, 'lI': lI, 'lD': lD, 'lM': lM, 'Lam_v': float(np.exp(lL - lI)),
         'Jvv': float(np.sum(np.exp(lPw - lM) * A['s'] ** 2))}                          # u E[s^2] / u: v-Fisher information of P (per unit mass)
    if need_L:
        g = A['sg'] * np.exp(A['lg'] - lI / 2); gp = A['sgp'] * np.exp(A['lgp'] - lI / 2); gpp = A['sgpp'] * np.exp(A['lgpp'] - lI / 2)   # scaled: sum P g^2 = 1
        Lg = gpp + A['s'] * gp; Rq = o['Rq']; P = np.exp(A['lP']) * wS
        o['rhs_Rq_prime'] = float(-np.sum(P * (Lg + Rq * g) ** 2) - np.sum(P * A['H'] * gp ** 2))   # = Rq' (I = 1 in these units)
        o['H_min'] = float(np.min(A['H']))
    return o


def check_identities(rng, n=12):
    rows = []
    for _ in range(n):
        K = rng.integers(2, 6); xs = np.sort(rng.uniform(-4, 4, K)); lw = rng.uniform(-6, 0, K); f = rng.uniform(-1, 1, K); u = rng.uniform(0.3, 2.0)
        y, h = grid(xs, 0.5 * u, 14.0); du = 1e-4 * u
        a = quantities(xs, lw, f, u, y, need_L=True); ap = quantities(xs, lw, f, u + du, y); am = quantities(xs, lw, f, u - du, y)
        dI = (np.exp(ap['lI']) - np.exp(am['lI'])) / (2 * du); Dv = np.exp(a['lD'])
        dRq = (ap['Rq'] - am['Rq']) / (2 * du)
        rows.append({'dI_plus_D_rel': float(abs(dI + Dv) / Dv), 'Rq_prime_fd': float(dRq), 'Rq_prime_identity': a['rhs_Rq_prime'],
                     'rel': float(abs(dRq - a['rhs_Rq_prime']) / max(abs(dRq), 1e-12)), 'H_min': a['H_min']})
    return rows


def logconcave_check(rng, n=60):
    """nu with log-concave density (Gaussian of variance s2, or uniform on an interval), random bounded f; t Rq(t) <= ln(M/I(t))."""
    rows = []
    for i in range(n):
        if i % 2 == 0: xs = np.linspace(-6, 6, 1201); s2 = rng.uniform(0.05, 4); lw = -xs ** 2 / (2 * s2) + np.log(xs[1] - xs[0])
        else: L = rng.uniform(0.5, 8); xs = np.linspace(-L, L, 1201); lw = np.full_like(xs, np.log(xs[1] - xs[0]))
        kind = rng.integers(3)
        if kind == 0: f = np.sin(rng.uniform(0.2, 8) * xs + rng.uniform(0, 6.3))
        elif kind == 1: f = np.sign(np.sin(rng.uniform(0.2, 5) * xs + rng.uniform(0, 6.3)))
        else: f = np.tanh(rng.normal(size=12) @ np.array([np.cos((j + 1) * xs / 2) for j in range(12)]))
        t = rng.uniform(0.05, 2.0); o = quantities(xs, lw, f, t, dense=True)
        rows.append({'t': t, 'tRq': t * o['Rq'], 'lnMI': o['lnMI'], 'slack': o['lnMI'] - t * o['Rq']})
    return rows


def unpack(z, K, nz=0):
    xs = z[:K]; lw = z[K:2 * K]; f = np.tanh(z[2 * K:3 * K]); f[:nz] = 0.0; return xs, lw, f      # nz leading masses carry f = 0


def objective(z, K, B, mode, nz=0):
    xs, lw, f = unpack(z, K, nz)
    if np.max(np.abs(xs)) > XMAX or np.max(np.abs(lw)) > LWMAX or np.max(np.abs(np.diff(np.sort(xs)))) < 1e-3: return 1e6
    o = quantities(xs, lw - np.max(lw), f)
    if not np.isfinite(o['Rq']) or not np.isfinite(o['lnMI']): return 1e6
    if mode in ('margin', 'hmargin'): return -(o['Rq'] - B * o['lnMI'])
    return -(o['Rq'] / o['lnMI']) + 10 * max(0.0, 20 - o['lnMI']) / 20 if o['lnMI'] > 5 else 1e6   # ratio, pushed to ln(M/I) >= 20


def start(rng, K, structured):
    D = rng.uniform(1, 18)
    if not structured:
        return np.concatenate([np.sort(rng.uniform(-D, D, K)), rng.uniform(-D * D / 2 - 5, 0, K), rng.normal(scale=2, size=K)])
    # structured: two heavy outer masses (f ~ 0) and K - 2 light signed masses inside the valley (hidden sources)
    D = rng.uniform(4, 16); xo = np.array([-D, D]) + rng.normal(scale=0.5, size=2); xi = rng.uniform(-0.6 * D, 0.6 * D, K - 2)
    lwo = rng.uniform(-1, 0, 2); lwi = -D * D / 2 + rng.uniform(-2, 2, K - 2) * D * 0.3 - xi ** 2 / 2 + D * np.abs(xi)
    fo = rng.normal(scale=0.2, size=2); fi = rng.choice([-3.0, 3.0], K - 2) + rng.normal(scale=0.5, size=K - 2)
    return np.concatenate([np.concatenate([xo, xi]), np.concatenate([lwo, lwi]), np.concatenate([fo, fi])])


def search(rng, starts, K, mode, B=1.0):
    best = None
    for i in range(starts):
        st = (mode in ('ratio', 'hmargin') or i % 2 == 1) and K >= 3; nz = 2 if st else 0   # hmargin: hidden (structured) starts only
        z0 = start(rng, K, structured=st); v0 = objective(z0, K, B, mode, nz)
        if v0 >= 1e6: continue
        r = minimize(objective, z0, args=(K, B, mode, nz), method='Nelder-Mead', options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-6})
        if best is None or r.fun < best[0]: best = (r.fun, r.x, nz)
    if best is None: return None
    xs, lw, f = unpack(best[1], K, best[2]); o = quantities(xs, lw - np.max(lw), f)
    return {'value': float(-best[0]), 'K': K, 'xs': xs.tolist(), 'lw': (lw - np.max(lw)).tolist(), 'f': f.tolist(), 'Rq': o['Rq'], 'lnMI': o['lnMI']}


if __name__ == '__main__':
    # usage: python an_signed1d.py base SEED            -> results/an_signed1d.json (identities, log-concave, sharpness)
    #        python an_signed1d.py search SEED K MODE   -> results/an_signed1d_search_K_MODE.json (one process per file)
    job = sys.argv[1] if len(sys.argv) > 1 else 'base'; rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    if job == 'base':
        out = {'identities': check_identities(rng)}
        print('identity dI/du = -D: max rel %.2e;  Rq\' identity max rel %.2e (min H %.2f)' % (max(r['dI_plus_D_rel'] for r in out['identities']), max(r['rel'] for r in out['identities']), min(r['H_min'] for r in out['identities'])), flush=True)
        lc = logconcave_check(rng); out['logconcave'] = {'n': len(lc), 'min_slack': min(r['slack'] for r in lc), 'rows': lc}
        print('log-concave nu: n %d, min [ln(M/I) - tRq] = %.4f (>= 0 by Proposition AN1)' % (len(lc), out['logconcave']['min_slack']), flush=True)
        xs = np.linspace(-60, 60, 12001); lw = np.full_like(xs, np.log(xs[1] - xs[0])); o = quantities(xs, lw, np.sin(2.0 * xs), 0.5, dense=True)
        out['sharpness_sin'] = {'k': 2.0, 't': 0.5, 'tRq': 0.5 * o['Rq'], 'lnMI': o['lnMI'], 'slack': o['lnMI'] - 0.5 * o['Rq'], 'ln2': float(np.log(2))}
        print('sharpness (uniform nu on [-60, 60], f = sin 2x, t = 1/2): tRq %.4f ln(M/I) %.4f slack %.4f (ln 2 = %.4f; edges)' % (0.5 * o['Rq'], o['lnMI'], out['sharpness_sin']['slack'], np.log(2)), flush=True)
        json.dump(out, open('results/an_signed1d.json', 'w'), indent=1, default=float)
    else:
        K = int(sys.argv[3]); mode = sys.argv[4]
        r = search(rng, 30, K, mode)
        if r is not None: print('search K %d %-6s best %.4f (Rq %.3f ln(M/I) %.3f)' % (K, mode, r['value'], r['Rq'], r['lnMI']), flush=True)
        json.dump(r, open('results/an_signed1d_search_%d_%s.json' % (K, mode), 'w'), indent=1, default=float)
