"""v1.3 checks for verification/run_all.py (section 9).  Each component runs in its own process (br_windows reads c' from the environment at import
and reconfigures module-level settings of bj_lemma), and prints one JSON line.

  python verification/bs_v13_checks.py COMPONENT [--full]

Components: F2i, F2ii, intervals, F6, F7, remark, tail, stress, rerun (rerun: --full only, every run behind results/br_windows.json in a temporary folder).
Nothing is written to results/.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, tempfile, subprocess, time
import numpy as np

np.seterr(all='ignore')
FULL = '--full' in sys.argv
J = lambda f: json.load(open('results/' + f))
JL = lambda f: [json.loads(l) for l in open('results/' + f) if l.strip()]
RT = 1e-9
KEYS = {6: 'N=6', 8: 'N=8', 12: "N=12, c'=0.0002", 16: "N=16, c'=0.00002", 20: 'N=20', 24: 'N=24'}
CPS = {6: '0.002', 8: '0.002', 12: '0.0002', 16: '0.00002', 20: '0.002', 24: '0.002'}
CELLF = {6: 'N6', 8: 'N8', 12: 'N12_cp0.0002', 16: 'N16_cp0.00002', 20: 'N20', 24: 'N24'}


def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-300)


def same(a, b, tol=RT):
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None: return a == b
    return rel(float(a), float(b)) <= tol


def W(cp):
    _os.environ['BR_CP'] = cp
    import br_windows
    return br_windows


def F2i():
    """Lemma F2(i): the cells at 1e-12 <= x <= 1 (N = 16, 20 by default; every N with --full), recomputed row by row (cell values; the in-cell audit with --full)."""
    import br_cells as C
    Ns = (6, 8, 12, 16, 20, 24) if FULL else (16, 20); dev = 0.0; ok = True; nrow = 0; stored = {}
    for N in Ns:
        rows = JL('br_cells_bk1_%s.jsonl' % CELLF[N]); pick = rows if FULL else rows[::6] + [rows[-1]]
        Wm = W(CPS[N])
        for d in pick:
            bs = np.geomspace(d['xa'] / 80, 1.0, 121); cells = []; okr = True; wm = np.inf
            for ba, bb in zip(bs[:-1], bs[1:]):
                r = C.cellG(N, d['xa'], d['xb'], ba, bb, sig=1e-24); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((d['xa'], d['xb'], ba, bb, r))
            if FULL:
                npt, nv = Wm.audit_row(N, cells[::4], False, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
            ok &= okr == d['ok']; dev = max(dev, rel(wm, d['margin'])); nrow += 1
        S = J('br_windows.json')[KEYS[N]]['bk1_x_ge_1e-12']
        ok &= S['x_cells'] == len(rows) == 120 and min(d['margin'] for d in rows) == S['worst_margin'] and S['audit_violations'] == 0
        stored[N] = {'cells': S['cells'], 'worst': S['worst_margin'], 'all_ok': S['all_ok']}
    closes = sorted(N for N in (6, 8, 12, 16, 20, 24) if J('br_windows.json')[KEYS[N]]['bk1_x_ge_1e-12']['all_ok'])
    return {'rows': nrow, 'ok': bool(ok and dev <= RT and closes == [6, 8, 12, 16, 20, 24]), 'dev': dev, 'N16': stored.get(16), 'N20': stored.get(20)}


def F2ii():
    """Lemma F2(ii): the case-B cells (x < 1e-12, d_n := 1, sigma_hat = 1e-24) for N = 16, c' = 2e-5; b-cells cover [c' 1e-24, 1]."""
    import br_cells as C
    Wm = W('0.00002'); rows = JL('br_cells_caseB_N16_cp0.00002.jsonl'); pick = rows if FULL else rows[::15] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        ba, bb = d['ba'], d['bb']; xhi = min(1e-12, 80 * bb); xs = np.concatenate([[0.0], np.geomspace(xhi * 1e-8, xhi, 41)]); wm = np.inf; okr = True; cells = []
        for xa, xb in zip(xs[:-1], xs[1:]):
            if xa / bb > 80: continue
            r = C.cellG(16, xa, xb, ba, bb, sig=1e-24, dec_one=True); okr &= r['ok']; wm = min(wm, r['margin']); cells.append((xa, xb, ba, bb, r))
        ok &= okr == d['ok'] and len(cells) == d['x_cells']; dev = max(dev, rel(wm, d['margin']))
        if FULL:
            npt, nv = Wm.audit_row(16, cells, True, 1e-24); ok &= npt == d['audit_points'] and nv == d['audit_violations'] == 0
    cover = abs(rows[0]['ba'] - 2e-29) < 1e-41 and abs(rows[-1]['bb'] - 1) < 1e-12 and all(abs(a['bb'] - b['ba']) <= 1e-12 * a['bb'] for a, b in zip(rows, rows[1:]))
    S = J('br_windows.json')[KEYS[16]]['caseB']
    return {'rows': len(pick), 'of': len(rows), 'ok': bool(ok and dev <= RT and cover and S['all_ok'] and S['audit_violations'] == 0), 'dev': dev, 'cover': cover,
            'cells': S['cells'], 'worst': S['worst_margin'], 'audit_points': S['audit_points']}


def intervals():
    """Lemmas F4, F5 and eps_R of Lemma F6: constants in outward-rounded interval arithmetic, recomputed and compared exactly (N = 6, 8, 12, 16; N = 20 attempt)."""
    out = {}; ok = True
    for N in ((6, 8, 12, 16) if FULL else (16,)):
        Wm = W(CPS[N]); S = J('br_windows.json')[KEYS[N]]
        c = Wm.caseC(N); a1 = Wm.caseA1(N); Ts = float(np.floor(a1['T_lower'] * 0.9)); a1['T_split'] = Ts; a1['ok'] = bool(Ts >= 1 and a1['T_lower'] > Ts)
        e = Wm.eps_R(N, Ts, a1['T_lower'])
        idn = json.loads(json.dumps([c, a1, e], default=list)) == json.loads(json.dumps([S['caseC'], S['caseA1'], S['eps_R']], default=list))
        ok &= idn and c['ok'] and a1['ok'] and e['ok']
        out[N] = {'identical': idn, 'a_prime_ratio': c['ratio_lower'], 'T': a1['T_lower'], 'eps_R': e['eps_R']}
        del _sys.modules['br_windows']
    return {'ok': bool(ok), **{'N%d' % k: v for k, v in out.items()}}


def F6():
    """Lemma F6: (i) N = 16 down to x -> 0 with eps = eps_in + eps_R + eps_Phi; (ii) N = 20 on [1e-12, 1]; recomputed (floating point, relative tolerance)."""
    S = J('br_windows.json'); r = {}
    Wm = W('0.00002'); A = S[KEYS[16]]['caseA2']; e = S[KEYS[16]]['eps_R']
    eps2 = Wm.EPS + e['eps_R'] + e['phi_remainder']; x = Wm.bj1(16, 0, eps2)
    r['i'] = {'A_min': x['A_min'], 'dev': max(rel(x['A_min'], A['A_min']), rel(x['step3a_interval']['margin_lower'], A['step3a_margin']), rel(eps2, A['eps'])),
              'ok': bool(x['ok'] and A['ok'] and x['x_cells'] == A['x_cells'] == 10001)}
    del _sys.modules['br_windows']; Wm = W('0.002'); B = S['N=20']['bj1_x_ge_1e-12']
    ER = Wm.E_R_sup(20); epsb = Wm.EPS + 2 * ER + 7.1e-14; y = Wm.bj1(20, 1e-12, epsb)
    r['ii'] = {'A_min': y['A_min'], 'dev': max(rel(ER, B['E_R_sup']), rel(y['A_min'], B['A_min']), rel(y['step2b']['a_lower_over_c'], B['step2i'])), 'ok': bool(y['ok'] and B['ok'])}
    n24 = S['N=24']['bj1_x_ge_1e-12']
    stops = (not n24['ok']) and n24['step2i'] < 1 and isinstance(S['N=20'].get('small_x_attempt', {}).get('caseA2'), str)
    return {'ok': bool(r['i']['ok'] and r['ii']['ok'] and max(r['i']['dev'], r['ii']['dev']) <= RT and stops), 'A_min_16': r['i']['A_min'], 'A_min_20': r['ii']['A_min'],
            'dev': max(r['i']['dev'], r['ii']['dev']), 'stops_recorded': stops}


def F7():
    """Lemma F7 and Theorems F8, F9: sup_x x f*(x) and the lower bounds of g, recomputed (floating point)."""
    S = J('br_windows.json'); Wm = W('0.002'); out = {}; ok = True
    for key, part, Nn in ((KEYS[16], 'conclusion_x_to_0', 16), ('N=20', 'conclusion_x_ge_1e-12', 20)):
        st = S[key][part]; A = S[key]['caseA2'] if part == 'conclusion_x_to_0' else S[key]['bj1_x_ge_1e-12']
        c = Wm.conclusion(Nn, A['A_min'], A['eps']); ok &= c['ok'] and same(c, st); out[Nn] = c['g_lower']
    xf = [Wm.conclusion(N, 0.9, 0.0)['sup_x_fstar'] for N in ((6, 8, 12, 16, 20, 24) if FULL else (6, 24))]
    ok &= max(xf) - min(xf) <= 1e-12 and max(xf) <= 1.1483
    return {'ok': bool(ok), 'g_16': out[16], 'g_20': out[20], 'sup_x_fstar': max(xf)}


def remark():
    """The remark of App. F, a_n >= 2 - n^2 (2 - a_1), on random non-negative profiles (numerical illustration of a proved inequality)."""
    rng = np.random.default_rng(7); t = np.linspace(0, 2 * np.pi, 4097)[:-1]; bad = 0; n = 2000 if FULL else 300
    for _ in range(n):
        f = np.zeros_like(t)
        for _k in range(rng.integers(1, 6)):
            c = rng.uniform(0, 2 * np.pi); w = 10 ** rng.uniform(-2, 0); f += np.exp(-np.angle(np.exp(1j * (t - c))) ** 2 / (2 * w * w))
        f = f / f.mean()
        a = np.array([2 * np.mean(f * np.cos(k * t)) for k in range(1, 30)])
        bad += int(np.sum(a[1:] < 2 - np.arange(2, 30) ** 2 * (2 - a[0]) - 1e-12))
    return {'ok': bad == 0, 'profiles': n, 'violations': bad}


def tail():
    """Theorem 11 for k0 >= 240 (bs_upper_tail.py): rerun into a temporary file and compare with the stored result."""
    tmp = tempfile.mkdtemp(); env = dict(_os.environ, BS_OUT=_os.path.join(tmp, 't.json'))
    p = subprocess.run([sys.executable, 'certificates/bs_upper_tail.py'], capture_output=True, text=True, env=env)
    new = json.load(open(env['BS_OUT'])); S = J('bs_upper_tail.json')
    return {'ok': bool(p.returncode == 0 and new['ok'] and S['ok'] and same(new, S)), 'identical_to_stored': same(new, S, 0.0),
            'R_U_hi': new['conclusion']['R_U_interval'][1], 'lhs_at_K0': new['conclusion']['lhs_at_K0']}


def stress():
    """The forms of every intermediate against a direct 300-digit evaluation (bs_upper_tail_stress.py; numerical)."""
    tmp = tempfile.mkdtemp(); env = dict(_os.environ, BS_OUT=_os.path.join(tmp, 's.json'))
    p = subprocess.run([sys.executable, 'certificates/bs_upper_tail_stress.py'], capture_output=True, text=True, env=env)
    new = json.load(open(env['BS_OUT'])); S = J('bs_upper_tail_stress.json')
    return {'ok': bool(p.returncode == 0 and new['ok'] and new['checks'] == S['checks'] and new['failures'] == 0), 'checks': new['checks'], 'failures': new['failures']}


def rerun():
    """--full: every run behind results/br_windows.json, from scratch in a temporary folder, compared with the stored file (relative 1e-9)."""
    tmp = tempfile.mkdtemp(); out = _os.path.join(tmp, 'br_windows.json'); t0 = time.time()
    runs = [('0.002', ['6', '8']), ('0.0002', ['12']), ('0.00002', ['16']), ('0.002', ['20', '24', '--large-x-only']), ('0.0000001', ['20', '--small-x-attempt'])]
    for cp, args in runs:
        env = dict(_os.environ, BR_CP=cp, BR_OUT=out, BR_DIR=tmp)
        p = subprocess.run([sys.executable, 'certificates/br_windows.py'] + args, capture_output=True, text=True, env=env)
        if p.returncode != 0: return {'ok': False, 'error': p.stderr[-300:]}
    new = json.load(open(out)); S = J('br_windows.json')
    def strip(d):
        if isinstance(d, dict): return {k: strip(v) for k, v in d.items() if k != 'caseC_numeric'}
        return d
    ok = same(strip(new), strip(S), 1e-9)
    cn = all(new[k]['caseC_numeric']['all_above_bound'] for k in new if 'caseC_numeric' in new[k])
    return {'ok': bool(ok and cn), 'seconds': time.time() - t0, 'keys': sorted(new)}


if __name__ == '__main__':
    comp = [a for a in sys.argv[1:] if not a.startswith('--')][0]
    print(json.dumps(globals()[comp](), default=float))
