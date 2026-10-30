"""v1.2 checks for verification/run_all.py (section 8).  Each component runs in its own process, because the certificates of App. E reconfigure
module-level settings of bj_lemma / bk_narrow_lemma (harmonic sums, x-cells), and prints one JSON line.

  python verification/bq_v12_checks.py COMPONENT [--full]

Components: E1i, E1ii, E1iii, E2, proofs, E5, BN1, E7, sigma, audit.  Nothing is written to results/.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, sys, tempfile
import numpy as np

np.seterr(all='ignore')
FULL = '--full' in sys.argv
J = lambda f: json.load(open('results/' + f))
JL = lambda f: [json.loads(l) for l in open('results/' + f) if l.strip()]
RT = 1e-9


def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-300)


def E1i():
    import bk_narrow_lemma as BK, bn_strip_x as BN
    BK.NN = np.array([1.0]); rows = JL('bn_bk1_strip.jsonl'); pick = rows if FULL else rows[::24] + [rows[-1]]; dev = 0.0; ok = True
    for d in pick:
        r = BN.bk1_strip_row(d['j'], d['xa'], d['xb']); ok &= r['ok'] == d['ok'] and r['ok']; dev = max(dev, rel(r['margin'], d['margin']))
    S = J('bn_strip_x.json')['bk1_strip']
    return {'rows': len(pick), 'of': len(rows), 'ok': bool(ok and dev <= RT and len(rows) == 480), 'dev': dev, 'cells': S['cells'], 'worst_margin': S['worst_margin'],
            'stored_ok': S['all_ok'] and min(d['margin'] for d in rows) == S['worst_margin']}


def E1ii():
    import bk_narrow_lemma as BK, bp_strip_p5 as P
    BK.NN = np.array([1.0]); rows = JL('bp_strip_caseB.jsonl'); bc = P.caseB_bcells()
    pick = rows if FULL else rows[::20] + [rows[-1]]; dev = 0.0; ok = True; same = True
    for d in pick:
        r = P.caseB_cell(d['j'], *bc[d['j']]); ok &= r['ok']; dev = max(dev, rel(r['margin'], d['margin']))
        same &= r['x_cells'] == d['x_cells'] and r['audit_points'] == d['audit_points'] and r['audit_violations'] == d['audit_violations'] == 0
    cover = abs(rows[0]['ba'] - 2e-27) < 1e-40 and abs(rows[-1]['bb'] - 1) < 1e-12 and all(abs(a['bb'] - b['ba']) <= 1e-12 * a['bb'] for a, b in zip(rows, rows[1:]))
    sig_hat = all(abs(d['sigma'] - 1e-24) < 1e-36 for d in rows)
    B = J('bp_strip_p5.json')['case_B']
    return {'rows': len(pick), 'of': len(rows), 'ok': bool(ok and same and dev <= RT and cover and sig_hat), 'dev': dev, 'cover': cover, 'sigma_hat_1e-24': sig_hat,
            'cells': B['cells'], 'worst_margin': B['worst_margin'], 'audit_points': B['audit_points']}


def E1iii():
    import bp_windowsN as W
    rows = JL('bp_windowsN_bk1.jsonl'); Ns = (2, 3, 4, 6, 8, 12, 20, 40) if FULL else (6,)
    pick = [d for d in rows if d['N'] in Ns]; pick = pick if FULL else pick[::12] + [pick[-1]]; dev = 0.0; same = True
    for d in pick:
        r = W.bk1N_row(d['N'], d['j'], d['xa'], d['xb']); same &= r['ok'] == d['ok'] and r['audit_violations'] == d['audit_violations'] == 0
        dev = max(dev, rel(r['margin'], d['margin']))
    S = J('bp_windowsN.json')
    closes = [int(k[2:]) for k, v in S.items() if v['closes']]
    return {'rows': len(pick), 'ok': bool(same and dev <= RT and closes == [2, 3, 4, 6] and S['N=6']['bk1_all_ok']), 'dev': dev, 'closes': closes,
            'N6_margin': S['N=6']['bk1_worst_margin'], 'N6_cells': S['N=6']['bk1_cells']}


def E2():
    import bq_proofs as Q
    r = Q.b_scaling(n=2000 if FULL else 300)
    return {'ok': bool(r['F_over_b_decreased'] == 0 and r['Lambda1_decreased'] == 0), **{k: int(v) for k, v in r.items()}}


def proofs():
    import bq_proofs as Q
    S = J('bq_proofs.json'); new = {'case_A1': Q.case_A1(), 'case_C': Q.case_C(), 'eps_R': Q.eps_R()}
    same = all(json.loads(json.dumps(new[k], default=str)) == json.loads(json.dumps(S[k], default=str)) for k in new)
    return {'ok': bool(same and S['all_ok'] and all(new[k]['ok'] for k in new)), 'identical_to_stored': same, 'T': new['case_A1']['T'][0],
            'a_prime_lower': new['case_C']['a_prime_lower'], 'ratio_C': new['case_C']['ratio_lower'], 'eps_R': new['eps_R']['eps_R_upper'],
            'g_strip': S['conclusion_strip']['g_lower'], 'g_N6': S['conclusion_N']['6']['g_lower']}


def E5():
    import bp_strip_p5 as P
    S = J('bp_strip_p5.json'); sigs = (24, 48, 100, 300) if FULL else (24,); dev = 0.0; ok = True
    for e in sigs:
        r, BL = P.case_A2(10.0 ** (-e)); s = S['case_A2_1e-%d' % e]; ok &= r['ok']
        dev = max(dev, rel(r['A_min'], s['A_min']), rel(r['step3a']['margin'], s['step3a']['margin']), rel(r['conclusion']['g_lower'], s['conclusion']['g_lower']))
    ia = BL.step3a_interval(); same_ia = json.loads(json.dumps(ia, default=float)) == S['case_A2_step3a_interval']
    a1 = P.case_A1(); c = P.case_C()
    return {'ok': bool(ok and dev <= RT and same_ia and ia['g_gt_rhs'] and S['all_ok'] and a1['ok'] and c['ok']), 'sigmas': list(sigs), 'dev': dev,
            'A_min': S['case_A2_1e-24']['A_min'], 'step3a_interval_margin': ia['margin_lower'], 'g_lower': S['case_A2_1e-24']['conclusion']['g_lower']}


def BN1():
    import bn_strip_x as BN
    S = J('bn_strip_x.json'); er = BN.E_R_sup(); r, BL = BN.strip_bj1(1e-24)          # the order of bn_strip_x.__main__ (E_R with the general eta: an upper bound)
    dev = max(rel(r['A_min'], S['bj1_strip_1e-24']['A_min']), rel(er, S['E_R_sup']))
    return {'ok': bool(r['ok'] and S['all_ok'] and dev <= RT and er <= 1e-6), 'dev': dev, 'A_min': r['A_min'], 'E_R_sup': er}


def E7():
    import bp_windowsN as W
    S = J('bp_windowsN.json'); Ns = (2, 3, 4, 6) if FULL else (6,); dev = 0.0; ok = True
    for N in Ns:
        r = W.bj1N(N); s = S['N=%d' % N]; ok &= r['ok'] and r['E_R_sup'] <= 1e-6 and r['conclusion']['ratio_to_0.134'] > 1
        dev = max(dev, rel(r['A_min'], s['A_min']), rel(r['E_R_sup'], s['E_R_sup']), rel(r['conclusion']['sup_x(1+Z2)'], s['conclusion']['sup_x(1+Z2)']))
    return {'ok': bool(ok and dev <= RT), 'N': list(Ns), 'dev': dev, 'A_min_6': S['N=6']['A_min'], 'E_R_6': S['N=6']['E_R_sup']}


def sigma():
    import bq_proofs as Q
    r = Q.sigma_lemma(n=400 if FULL else 60)
    bad = r['general']['not_closing'] + r['strip']['not_closing'] + r['caseB_points_below_sigma_hat']['not_closing'] + r['general']['margin_below_sigma_hat'] + r['strip']['margin_below_sigma_hat']
    return {'ok': bool(bad == 0), 'pairs': r['general']['cell_sigma_pairs'] + r['strip']['cell_sigma_pairs'], 'caseB_points': r['caseB_points_below_sigma_hat']['points']}


def audit():
    """--full: the whole audit into a temporary file, compared row by row with results/bp_audit.jsonl; default: the reduced audit (BP_AUDIT_QUICK)."""
    import subprocess
    tmp = tempfile.mkdtemp(); out = _os.path.join(tmp, 'audit.jsonl')
    env = {**_os.environ, 'BP_AUDIT_OUT': out, 'OMP_NUM_THREADS': '1'}
    if not FULL: env['BP_AUDIT_QUICK'] = '1'
    subprocess.run([sys.executable, 'certificates/bp_audit.py'], env=env, capture_output=True, text=True, check=True)
    A = {}
    for l in open(out):
        if l.strip(): d = json.loads(l); A[d['key']] = d
    old = [d for k, d in A.items() if 'v1.1 code' in k and d['violations']]
    other = [k for k, d in A.items() if d.get('violations') and 'v1.1 code' not in k]
    res = {'checks': len(A), 'cells': sum(d.get('cells', 0) for d in A.values()), 'points': sum(d.get('points', 0) for d in A.values()),
           'v1.1_code_reversals': sorted({k.split(':')[-1] for k in (d['key'] for d in old)}), 'other_reversals': other}
    if FULL:
        S = {}
        for l in open('results/bp_audit.jsonl'):
            if l.strip(): d = json.loads(l); S[d['key']] = d
        res['identical_counts'] = set(S) == set(A) and all((S[k].get('cells'), S[k].get('points'), S[k].get('violations')) ==
                                                           (A[k].get('cells'), A[k].get('points'), A[k].get('violations')) for k in S)
    res['ok'] = bool(not other and 'H1' in res['v1.1_code_reversals'] and res.get('identical_counts', True))
    return res


if __name__ == '__main__':
    comp = [a for a in sys.argv[1:] if not a.startswith('--')][0]
    print(json.dumps(globals()[comp](), default=float))
