"""v1.4 checks for verification/run_all.py (section 10).  Each component runs in its own process (br_windows reads BR_PHASE and c' from the
environment at import) and prints one JSON line.

  python verification/bu_v14_checks.py COMPONENT [--full]

Components: eps, theta, F2i, F2ii, intervals, F6, F7, phases, rerun (rerun: --full only).  Nothing is written to results/.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, tempfile, subprocess, time, math
import numpy as np

np.seterr(all='ignore')
FULL = '--full' in sys.argv
J = lambda f: json.load(open('results/' + f))
JL = lambda f: [json.loads(l) for l in open('results/' + f) if l.strip()]
RT = 1e-9
KEY = "N=20, c'=0.000000001, phase"; CP = '0.000000001'


def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-300)


def same(a, b, tol=RT):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None: return a == b
    return rel(float(a), float(b)) <= tol


def W():
    _os.environ['BR_PHASE'] = '1'; _os.environ['BR_CP'] = CP
    import br_windows
    return br_windows


def eps():
    """Theorem G1 (bt_eps.py): rerun into a temporary file, compared exactly with results/bt_eps.json."""
    tmp = tempfile.mkdtemp(); env = dict(_os.environ, BT_OUT=_os.path.join(tmp, 'e.json'))
    p = subprocess.run([sys.executable, 'certificates/bt_eps.py'], capture_output=True, text=True, env=env)
    new = json.load(open(env['BT_OUT'])); S = J('bt_eps.json')
    return {'ok': bool(p.returncode == 0 and new == S and S['ok'] and S['form']['gamma'] == '3/2'), 'identical': new == S, 'sup': S['sup_k_ge_3.43'], 'c': S['form']['c'][1]}


def theta():
    """the theta_tau corrections for tau >= 80 with the term a_10 of the sine components (bl_theta.eps_table, phase=True)."""
    import bl_theta as T
    a = T.eps_table(80.0); b = T.eps_table(80.0, phase=True)
    return {'ok': bool(b['E_R'] <= 1.5e-14 and b['e3_AX3'] <= 3.0e-14 and b['eps_theta'] <= 7.1e-14 and b['eps_theta'] < 1e-6), 'E_R': b['E_R'], 'E_R_even': a['E_R'],
            'e3': b['e3_AX3'], 'eps_theta': b['eps_theta']}


def F2i():
    """Lemma F2(i), any phases (BR_PHASE=1): the cells at 1e-12 <= x <= 1 for N = 20, recomputed row by row (with the in-cell audit for --full)."""
    Wm = W(); import br_cells as C
    rows = JL('br_cells_bk1_N20_cp0.000000001_ph.jsonl'); pick = rows if FULL else rows[::6] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        bs = np.geomspace(d['xa'] / 80, 1.0, 121); cells = []; okr = True; wm = np.inf
        for ba, bb in zip(bs[:-1], bs[1:]):
            r = C.cellG(20, d['xa'], d['xb'], ba, bb, sig=1e-24); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((d['xa'], d['xb'], ba, bb, r))
        if FULL:
            npt, nv = Wm.audit_row(20, cells[::4], False, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
        ok &= okr == d['ok'] and C.PHASE; dev = max(dev, rel(wm, d['margin']))
    S = J('bu_windows.json')[KEY]['bk1_x_ge_1e-12']
    return {'rows': len(pick), 'ok': bool(ok and dev <= RT and S['all_ok'] and S['audit_violations'] == 0 and len(rows) == 120), 'dev': dev, 'cells': S['cells'],
            'worst': S['worst_margin'], 'audit_points': S['audit_points']}


def F2ii():
    """Lemma F2(ii), any phases: the case-B cells (x < 1e-12, d_n := 1, sigma_hat = 1e-24) for N = 20, c' = 1e-9; b-cells cover [1e-33, 1]."""
    Wm = W(); import br_cells as C
    rows = JL('br_cells_caseB_N20_cp0.000000001_ph.jsonl'); pick = rows if FULL else rows[::15] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        ba, bb = d['ba'], d['bb']; xhi = min(1e-12, 80 * bb); xs = np.concatenate([[0.0], np.geomspace(xhi * 1e-8, xhi, 41)]); wm = np.inf; okr = True; cells = []
        for xa, xb in zip(xs[:-1], xs[1:]):
            if xa / bb > 80: continue
            r = C.cellG(20, xa, xb, ba, bb, sig=1e-24, dec_one=True); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((xa, xb, ba, bb, r))
        ok &= okr == d['ok'] and len(cells) == d['x_cells']; dev = max(dev, rel(wm, d['margin']))
        if FULL:
            npt, nv = Wm.audit_row(20, cells, True, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
    cover = abs(rows[0]['ba'] - 1e-33) < 1e-45 and abs(rows[-1]['bb'] - 1) < 1e-12 and all(abs(a['bb'] - b['ba']) <= 1e-12 * a['bb'] for a, b in zip(rows, rows[1:]))
    S = J('bu_windows.json')[KEY]['caseB']
    return {'rows': len(pick), 'of': len(rows), 'ok': bool(ok and dev <= RT and cover and S['all_ok'] and S['audit_violations'] == 0), 'dev': dev, 'cover': cover,
            'cells': S['cells'], 'worst': S['worst_margin'], 'audit_points': S['audit_points']}


def intervals():
    """Lemmas F4, F5 and eps_R of Lemma F6 (with Lemma Fe and a_10): interval constants recomputed, identical to the stored ones."""
    Wm = W(); S = J('bu_windows.json')[KEY]
    c = Wm.caseC(20); a1 = Wm.caseA1(20); Ts = float(np.floor(a1['T_lower'] * 0.9)); a1['T_split'] = Ts; a1['ok'] = bool(Ts >= 1 and a1['T_lower'] > Ts)
    e = Wm.eps_R(20, Ts, a1['T_lower'])
    idn = json.loads(json.dumps([c, a1, e], default=list)) == json.loads(json.dumps([S['caseC'], S['caseA1'], S['eps_R']], default=list))
    return {'ok': bool(idn and c['ok'] and a1['ok'] and e['ok']), 'identical': idn, 'ratio_C': c['ratio_lower'], 'T': a1['T_lower'], 'eps_R': e['eps_R']}


def F6():
    """Lemma F6 (Lemma 28 with 20 harmonics, any phases, x -> 0), recomputed (floating point cells; step 3(i) in interval arithmetic)."""
    Wm = W(); S = J('bu_windows.json')[KEY]; A = S['caseA2']; e = S['eps_R']
    eps2 = Wm.EPS + e['eps_R'] + e['phi_remainder']; r = Wm.bj1(20, 0, eps2)
    dev = max(rel(r['A_min'], A['A_min']), rel(r['step3a_interval']['margin_lower'], A['step3a_margin']), rel(eps2, A['eps']), rel(r['step2b']['a_lower_over_c'], A['step2i']))
    return {'ok': bool(r['steps_ok'] and A['ok'] and dev <= RT and r['x_cells'] == 10001 and r['step3a_interval']['margin_lower'] > 1), 'A_min': r['A_min'],
            'step3a': r['step3a_interval']['margin_lower'], 'step2i': r['step2b']['a_lower_over_c'], 'dev': dev}


def F7():
    """Theorem F8: k_eff^2 and the lower bound of g, recomputed."""
    Wm = W(); S = J('bu_windows.json')[KEY]; c = Wm.conclusion(20, S['caseA2']['A_min'], S['caseA2']['eps'])
    return {'ok': bool(c['ok'] and same(c, S['conclusion_x_to_0'])), 'g': c['g_lower'], 'k_eff2': c['k_eff2_max']}


def phases():
    """numerical illustrations of the proved phase steps: the parity identity, the Toeplitz factor, the triangle bound G (Lemma F4), and |a_n| >= 2 - n^2(2 - |a_1|)."""
    rng = np.random.default_rng(11); n_ok = 0; bad = []
    t = np.linspace(0, 2 * np.pi, 4097)[:-1]
    for _ in range(200 if FULL else 60):
        N = int(rng.integers(1, 8)); q = rng.uniform(0, 0.25, N); q *= min(1, 0.24 / q.sum()); ph = rng.uniform(0, 2 * np.pi, N)
        w, s = rng.normal(), 10 ** rng.uniform(-1, 1); n = np.arange(1, N + 1)[:, None]
        env = np.exp(-w * w / (2 * s * s)); h = 1 + env * np.sum(q[:, None] * np.cos(n * t - ph[:, None]), 0)
        ht = -env * np.sum(n * q[:, None] * np.sin(n * t - ph[:, None]), 0); hw = -(w / s ** 2) * (h - 1)
        if abs(np.mean(ht * hw / h)) > 1e-12 * max(1, abs(w) / s ** 2): bad.append('identity')
        tau = 10 ** rng.uniform(-3, 1.5); tt = np.linspace(-np.pi, np.pi, 8001)[:-1]
        th = 2 * np.pi * sum(np.exp(-(tt + 2 * np.pi * m) ** 2 / (2 * tau)) for m in range(-15, 16)) / np.sqrt(2 * np.pi * tau)
        htt = -np.sum(n * q[:, None] * np.sin(n * tt - ph[:, None]), 0); S2 = float(np.sum(n[:, 0] ** 2 * q * q))
        e = lambda k: math.exp(-k * k * tau / 2); dph = 2 * sum(e(k) for k in range(1, N)) + sum(e(k) for k in range(2, 2 * N + 1))
        if np.mean(th * htt * htt) > (1 + dph) * S2 / 2 * (1 + 1e-9) + 1e-15: bad.append('toeplitz')
        Q = rng.uniform(0.01, 0.25); X0 = Q + rng.uniform(1e-3, 0.7); L = 10 ** rng.uniform(-2, 2)
        z = np.linspace(-12, 12, 200001); g = np.sum(np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * (z[1] - z[0]) * np.maximum(X0 * np.abs(np.cos(L * z / (2 * math.sqrt(2)))) - Q, 0) ** 2)
        d = X0 - Q
        if g < d * d * math.sqrt(d / X0) / math.pi * (1 - 1e-6): bad.append('triangle')
        M = 12; pk = rng.uniform(0, 2 * np.pi); wd = 10 ** rng.uniform(-1.5, 0); f = np.exp((np.cos(t - pk) - 1) / wd ** 2); f = f / f.mean()
        ah = np.array([2 * np.mean(f * np.exp(-1j * k * t)) for k in range(1, M + 1)])
        if any(abs(ah[k - 1]) < 2 - k * k * (2 - abs(ah[0])) - 1e-9 for k in range(1, M + 1)): bad.append('remark')
        n_ok += 1
    return {'ok': not bad, 'cases': n_ok, 'bad': bad[:5]}


def rerun():
    """--full: the phase run behind results/bu_windows.json from scratch in a temporary folder, compared with the stored file (relative 1e-9)."""
    tmp = tempfile.mkdtemp(); out = _os.path.join(tmp, 'bu_windows.json'); t0 = time.time()
    env = dict(_os.environ, BR_PHASE='1', BR_CP=CP, BR_OUT=out, BR_DIR=tmp)
    p = subprocess.run([sys.executable, 'certificates/br_windows.py', '20'], capture_output=True, text=True, env=env)
    if p.returncode != 0: return {'ok': False, 'error': p.stderr[-300:]}
    new = json.load(open(out)); S = J('bu_windows.json')
    strip = lambda d: {k: strip(v) for k, v in d.items() if k != 'caseC_numeric'} if isinstance(d, dict) else d
    return {'ok': bool(same(strip(new), strip(S), 1e-9) and new[KEY]['caseC_numeric']['all_above_bound']), 'seconds': time.time() - t0}


if __name__ == '__main__':
    comp = [a for a in sys.argv[1:] if not a.startswith('--')][0]
    print(json.dumps(globals()[comp](), default=float))
