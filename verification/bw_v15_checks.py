"""v1.5 checks for verification/run_all.py (section 11): any number of harmonics when K >= K0 (App. F, Lemmas F9, F10, Theorem F11).
Each component runs in its own process (bw_k0cert reads BW_K0 / BW_TS and br_windows reads BR_PHASE, BR_CP at import) and prints one JSON line.

  python verification/bw_v15_checks.py COMPONENT K0 [--full]

Components: bounds, sums, consts, F6, F7, F2i, F2ii, conclusion.  K0 = 0.003 (Theorem F11) or 0.002 (the remark after it, T_s just below T).
Nothing is written to results/ (recomputations go to a temporary folder).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, tempfile, math
import numpy as np

np.seterr(all='ignore')
FULL = '--full' in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
COMP, K0 = ARGS[0], (ARGS[1] if len(ARGS) > 1 else '0.003')
TS = {'0.003': 'floor', '0.002': 'near'}[K0]
_os.environ['BW_K0'] = K0; _os.environ['BW_TS'] = TS; _os.environ['BR_PHASE'] = '1'; _os.environ['BR_CP'] = '0.000000001'
_os.environ['BR_DIR'] = tempfile.mkdtemp()
J = lambda f: json.load(open('results/' + f))
JL = lambda f: [json.loads(l) for l in open('results/' + f) if l.strip()]
RT = 1e-9
S = J('bw_k0_%s.json' % K0)


def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-300)


def same(a, b, tol=RT):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None: return a == b
    return rel(float(a), float(b)) <= tol


def M():
    import bw_k0cert
    return bw_k0cert


def bounds():
    """illustrations of the proved steps on random instances: (S) and (P) of Lemma F1(vi), (vii); the dual bound >= the exact greedy value of the
    linear programme; the tail bound >= the direct sum."""
    W = M(); rng = np.random.default_rng(17); bad = []; n_cases = 400 if FULL else 120
    for _ in range(n_cases):
        K = 10 ** rng.uniform(-3, 0); n = np.arange(1, 3001, dtype=float); d = np.exp(-(n * n - 1) * K / 2)
        y = rng.uniform(0, min(0.25, 2 * math.exp(-K / 2))); q = np.zeros_like(n); q[0] = y
        k = int(rng.integers(1, 30)); idx = rng.choice(np.arange(1, min(2999, int(4 / math.sqrt(K)) + 30)), size=k, replace=False)
        q[idx] = rng.uniform(0, 1, k) * y * d[idx]; s = q[1:].sum()
        if s > 0.25 - y: q[1:] *= (0.25 - y) / s
        S2 = float(np.sum(n[1:] ** 2 * q[1:] ** 2))
        if S2 > y * q[1:].sum() * 2 * math.exp(K / 2) / (math.e * K) * (1 + 1e-12): bad.append('S')
        ph = rng.uniform(0, 2 * math.pi, len(n)); t = np.linspace(0, 2 * math.pi, 2001)[:-1]; nz = np.nonzero(q)[0][:40]
        ht = -np.sum((n[nz] * q[nz])[:, None] * np.sin(np.outer(n[nz], t) - ph[nz][:, None]), axis=0)
        tau = 10 ** rng.uniform(-3, 2); th = 1 + 2 * sum(np.exp(-j * j * tau / 2) * np.cos(j * t) for j in range(1, 200))
        if float(np.mean(th * ht ** 2)) > float(np.sum(n[nz] * q[nz])) ** 2 * (1 + 1e-9): bad.append('P')
        c = rng.uniform(0, 5, 50); cap = rng.uniform(0, 0.02, 50); B = rng.uniform(0, 0.5)
        o = np.argsort(-c); cs = np.cumsum(cap[o]); full = cs <= B; ex = float(np.sum((c[o] * cap[o])[full]))
        kk = int(np.sum(full))
        if kk < 50: ex += float(c[o][kk] * (B - (cs[kk - 1] if kk else 0.0)))
        if W.lp_up(c, cap, B) < ex * (1 - 1e-12): bad.append('LP')
        m = int(rng.integers(2, 400)); p = int(rng.integers(0, 3)); x = 10 ** rng.uniform(-3, 0)
        try:
            tb = W.tail_up(p, m, x); nn = np.arange(m, m + 200000, dtype=float)
            if float(np.sum(nn ** p * np.exp(-(nn * nn - 1) * x / 2))) > max(tb, W.TINY) * (1 + 1e-12): bad.append('tail')
        except ValueError:
            pass
    return {'ok': not bad, 'cases': n_cases, 'bad': bad[:5]}


def sums():
    W = M(); s = W.sups()
    return {'ok': bool(same(s, S['sups'])), 'N1': s['N1'], 'N2': s['N2'], 'S2': s['S2']}


def consts():
    """case C (Lemma F4) and T (Lemma F5) in interval arithmetic, eps_R (Lemma F6 with Lemma Fe): recomputed, identical to the stored ones."""
    W = M(); import br_windows as BW
    c = W.caseC_I(S['sups']); a = W.caseA1_I(S['sups']); T = a['T_lower']
    Ts = math.floor(0.9 * T) if TS == 'floor' else T * (1 - 1e-6)
    e = {'N_eff': S['eps_R']['N_eff'], 'T_s': Ts, **BW.eps_R(S['eps_R']['N_eff'], float(Ts), T)}
    idn = json.loads(json.dumps([c, a, e], default=list)) == json.loads(json.dumps([S['caseC'], S['caseA1'], S['eps_R']], default=list))
    neff = int(math.ceil(max(S['sups']['N1'] / 0.25, math.sqrt(S['sups']['N2'] / 0.25), 2)))
    return {'ok': bool(idn and c['ok'] and e['ok'] and Ts >= 1 and neff == e['N_eff']), 'identical': idn, 'ratio_C': c['ratio_lower'], 'a_prime': c['a_prime_lower'],
            'T': T, 'T_s': Ts, 'eps_R': e['eps_R']}


def F6():
    """Lemma F10(iv): Lemma F6 with eta_p, Z2 at max(x, K0), budgeted Gamma_* and the (x, q)-cells, recomputed from scratch (temporary folder)."""
    W = M(); e = S['eps_R']; eps = 1e-6 + e['eps_R'] + e['phi_remainder'] * (1 + 1e-9)
    r = W.lemma_F6(eps); A = S['F6']
    keys = ('A_min', 'Gamma_star_budget', 'e_prime', 'step2i', 'gamma_star_budget', 'step3a_interval_margin', 'step3b')
    dev = max(rel(r[k], A[k]) for k in keys)
    return {'ok': bool(r['ok'] and A['ok'] and dev <= RT and r['budget_audit_violations'] == 0 and r['step3a_interval_margin'] > 1), 'dev': dev,
            **{k: r[k] for k in keys}, 'rows': r['budget_x_rows']}


def F7():
    W = M(); v = W.fstar_sup(); a = W.fstar_audit(400 if FULL else 100)
    return {'ok': bool(rel(v, S['fstar']['sup_x_fstar']) <= RT and a[1] == 0), 'sup_x_fstar': v, 'audit_points': a[0]}


def F2i():
    W = M(); import br_windows as BW, br_cells as C
    C.cellG = W.cell_rig
    rows = JL('bw_cells_bk1_K%s.jsonl' % K0); pick = rows if FULL else rows[::12] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        bs = np.geomspace(d['xa'] / 80, 1.0, 121); cells = []; okr = True; wm = np.inf
        for ba, bb in zip(bs[:-1], bs[1:]):
            r = W.cell_rig('', d['xa'], d['xb'], ba, bb, sig=1e-24); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((d['xa'], d['xb'], ba, bb, r))
        if FULL:
            npt, nv = BW.audit_row('', cells[::4], False, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
        ok &= okr == d['ok']; dev = max(dev, rel(wm, d['margin']))
    P = S['bk1']
    return {'rows': len(pick), 'ok': bool(ok and dev <= RT and P['all_ok'] and P['audit_violations'] == 0 and len(rows) == 120), 'dev': dev, 'cells': P['cells'],
            'worst': P['worst_margin'], 'audit_points': P['audit_points']}


def F2ii():
    W = M(); import br_windows as BW, br_cells as C
    C.cellG = W.cell_rig
    rows = JL('bw_cells_caseB_K%s.jsonl' % K0); pick = rows if FULL else rows[::30] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        ba, bb = d['ba'], d['bb']; xhi = min(1e-12, 80 * bb); xs = np.concatenate([[0.0], np.geomspace(xhi * 1e-8, xhi, 41)]); wm = np.inf; okr = True; cells = []
        for xa, xb in zip(xs[:-1], xs[1:]):
            if xa / bb > 80: continue
            r = W.cell_rig('', xa, xb, ba, bb, sig=1e-24); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((xa, xb, ba, bb, r))
        ok &= okr == d['ok'] and len(cells) == d['x_cells']; dev = max(dev, rel(wm, d['margin']))
        if FULL:
            npt, nv = BW.audit_row('', cells, True, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
    cover = abs(rows[0]['ba'] - 1e-33) < 1e-45 and abs(rows[-1]['bb'] - 1) < 1e-12 and all(abs(a['bb'] - b['ba']) <= 1e-12 * a['bb'] for a, b in zip(rows, rows[1:]))
    P = S['caseB']
    return {'rows': len(pick), 'of': len(rows), 'ok': bool(ok and dev <= RT and cover and P['all_ok'] and P['audit_violations'] == 0), 'dev': dev,
            'cells': P['cells'], 'worst': P['worst_margin'], 'audit_points': P['audit_points']}


def conclusion():
    c = S['conclusion']; A = S['F6']['A_min']; eps = c['eps']
    kap2 = (1 + 0.125 + 0.25 / (2 * 0.75)) ** 2 / (8 * 0.75)
    k2 = (1 + eps) * (8 * kap2 * 1.25 * S['fstar']['sup_x_fstar'] + 0.25); g = 0.9 * A / (2 * math.sqrt(k2))
    return {'ok': bool(c['ok'] and rel(k2, c['k_eff2_max']) <= RT and rel(g, c['g_lower']) <= RT and g > 0.134 and S['caseC']['ok'] and S['F6']['ok']),
            'k_eff2': k2, 'g': g}


if __name__ == '__main__':
    print(json.dumps(globals()[COMP](), default=float))
