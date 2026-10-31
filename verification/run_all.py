"""Single entry point: checks every claim of the paper that rests on stored results or on a computer-assisted certificate.

  python verification/run_all.py              # default (a few minutes on one core): hashes, stored results, every number of the manuscript,
                                              # and a recomputation of representative cells of every certificate
  python verification/run_all.py --recompute  # in addition, recompute every certificate in full and the whole audit (two to three hours on one
                                              # core, of which the audit is about 40 min; writes nothing to results/)

v1.1 adds section 7: Lemma BK1 (tau0 = 80), the theta_tau corrections of the window lemmas (App. D), the lower-side chain with every input lemma
perturbed by eps_in = 1e-6 (421 cells and the tail), Lemma BJ1 (the former (P8)), and Lemma BL7 (the upper side in theta_tau form).
v1.2 adds section 8: App. E (strips without (P5): Lemmas 29-33 and Theorem 34; at most six harmonics: Lemma 35 and Theorem 36), the corrected
certificate of Lemma 20 (section 7 now recomputes it), and the audit at interior points (App. B; default: the reduced audit, a few minutes;
--recompute: the whole audit, about 40 minutes).
v1.3 adds section 9: App. F (windows with at most N harmonics without (P5): Lemmas F1-F7, Theorems F8, F9, Table F2) and Theorem 11 beyond
k0 = 240 (bs_upper_tail.py).  Default: representative rows of the new cells, all interval constants for N = 16, Lemma F6 and F7 in full (a few
minutes); --recompute: every row with its in-cell audit, the interval constants for every N, and every run behind results/br_windows.json from
scratch in a temporary folder (about one hour).

Comparisons of recomputed floating-point values with stored ones use a relative tolerance (CPUs differ in the last bits); the interval-arithmetic
certificate (decimal, 60 digits) is compared exactly.  Each check prints which kind of comparison it made.  Exit code 0 iff every check passes."""
import os, sys, json, glob, ast, hashlib, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('src', 'certificates', 'experiments', 'verification'):
    sys.path.insert(0, os.path.join(R, d))
os.chdir(R)
os.environ.setdefault('OMP_NUM_THREADS', '1')
import warnings
import numpy as np
from vtools import check, finish
# the case-B margin (sigma_caseB_limit / sigma_hat) is printed as inf where sigma_hat is floored at 1e-300; it is a report value, not a condition
warnings.filterwarnings('ignore', message='overflow encountered in scalar divide')

FULL = '--recompute' in sys.argv
RTOL = 1e-9
t0 = time.time()
J = lambda f: json.load(open(os.path.join('results', f)))


def rel(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-300)


def same(a, b, tol=1e-9):
    """JSON-like objects equal up to a relative tolerance on floats (booleans and integers exactly)."""
    if isinstance(a, dict): return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str): return a == b
    return rel(float(a), float(b)) <= tol


# ---------------------------------------------------------------- 1. files
man = {}
for line in open('MANIFEST.sha256'):
    h, p = line.split(None, 1); man[p.strip()] = h
files = sorted(os.path.join('results', f) for f in os.listdir('results'))
bad = [p for p in files if man.get(p) != hashlib.sha256(open(p, 'rb').read()).hexdigest()]
check('MANIFEST.sha256: %d result files, every SHA-256 matches, no file missing or extra' % len(files), not bad and set(man) == set(files), ', '.join(bad[:5]))
pys = sorted(glob.glob('*/*.py'))
gbad = []
for f in pys:
    try: ast.parse(open(f).read(), feature_version=(3, 11))
    except SyntaxError: gbad.append(f)
check('Python 3.11 grammar for all %d scripts' % len(pys), not gbad, ', '.join(gbad))
big = [p for p in glob.glob('**/*', recursive=True) if os.path.isfile(p) and os.path.getsize(p) >= 25e6]
check('every file is below 25 MB', not big, ', '.join(big))

# ---------------------------------------------------------------- 2. lower side: Theorem 9 (cells), Lemma 6 (table), Lemma 7 (tail)
C = {}
for f in sorted(glob.glob('results/bh_cert_cells_part*.json')): C.update(json.load(open(f)))
cs = sorted(C.values(), key=lambda v: v['la']); L_LO, L_HI = 24 * np.log(10), 1e5 * np.log(10)
check('lower certificate (stored): %d cells cover ln(1/sigma) in [24 ln10, 1e5 ln10] without gaps; every condition holds on every cell' % len(cs),
      len(cs) == 421 and cs[0]['la'] <= L_LO and cs[-1]['lb'] >= L_HI and all(abs(b['la'] - a['lb']) < 1e-9 * a['lb'] for a, b in zip(cs, cs[1:])) and all(v['ok'] for v in cs))
check('Theorem 9 (stored): g sqrt(l)/(1 - eps_bar) >= 1 - 1.7 ln(l)/l on every cell (largest constant needed %.4f); worst point at the left end (d >= %.2f)' % (
      max(v['c1_needed'] for v in cs), min(v['d_sign'] for v in cs)), all(v['L_cell'] >= 1 - 1.7 * np.log(v['la']) / v['la'] for v in cs) and all(v['d_sign'] >= 0 for v in cs))
import bh_cert_cells as H
meta = J('bh_cert_cells_meta.json')
redo = sorted(C, key=int) if FULL else [str(i) for i in list(range(0, 421, 15)) + [420]]
worst = 0.0
for i in redo:
    r = H.cell(C[i]['la'], C[i]['lb'], meta['S1_P'], meta['S_i_at_1']); worst = max(worst, rel(r['L_cell'], C[i]['L_cell']))
    if not r['ok']: break
check('Theorem 9 recomputed on %d cells (%s): every condition holds; L_cell agrees with the stored value (relative difference %.1e <= %.0e; floating point)' % (
      len(redo), 'all' if FULL else 'every 15th cell', worst, RTOL), r['ok'] and worst <= RTOL)
MT = J('bi_monotone.json')
check('Lemma 6, monotonicity table (stored grid check): %d checks consistent with the declared directions, none non-monotone' % MT['n_checks'], MT['all_ok'] and MT['n_nonmonotone'] == 0)
if True:
    import subprocess
    out = subprocess.run([sys.executable, 'certificates/bi_monotone_table.py'], capture_output=True, text=True, env={**os.environ, 'RGTS_NO_WRITE': '1'}).stdout
    check('Lemma 6, monotonicity table recomputed: %s' % out.strip().split('\n')[-1], 'all consistent with the table: True' in out)
T = J('bh_tail.json')
check('Lemma 7 (tail, stored): the explicit bound increases from %.6f at l_end to 1; direct runs at l = 1e6, 1e7, 1e8 hold with K below the bound' % T['T_at_l_end'],
      T['T_increasing'] and T['T_at_1e300'] > 0.999999 and all(v['ok'] and v['K'] <= v['K_bound_from_tail'] for v in T['direct'].values()))
r6 = H.cell(1e6, 1e6 * 1.0000001, meta['S1_P'], meta['S_i_at_1'])
check('Lemma 7 recomputed at l = 1e6: all conditions hold, K = %.3f (stored %.3f; relative %.1e, floating point)' % (r6['K'], T['direct']['1e+06']['K'], rel(r6['K'], T['direct']['1e+06']['K'])),
      r6['ok'] and rel(r6['K'], T['direct']['1e+06']['K']) <= RTOL)
AB = J('bf_ay_bootstrap.json')
check('strips read upward (Remark 8, representative points): closes from F_0 = %s (1e-24), %s (1e-96)' % (AB['1e-24']['smallest_F0'], AB['1e-96']['smallest_F0']),
      AB['1e-24']['smallest_F0'] == 0.5 and AB['1e-96']['smallest_F0'] == 0.2)

# ---------------------------------------------------------------- 3. upper side: Theorem 11 (interval certificate), class membership
CU = J('bf_cert_upper.json'); BA = J('ba_cert.json')
check('Theorem 11 (stored): U_P sqrt(l) <= 1 + 6.0 lnln/ln on %d cells covering l in [%.2f, %.0f], largest bound of h %.3e; strip sequence S\' with 7.4: %d cells, %.3e' % (
      CU['cells'], CU['t_range_checked'][0], CU['t_range_checked'][1], CU['worst_upper_bound_of_h'], BA['cells'], BA['worst_upper_bound_of_h']),
      CU['all_cells_certified'] and CU['worst_upper_bound_of_h'] < 0 and CU['t_range_checked'][0] <= L_LO and CU['t_range_checked'][1] >= L_HI and BA['worst_upper_bound_of_h'] < 0)
from decimal import Decimal as D
from bf_cert_upper import cell_bound
k = D('3.43'); ks = []
while k < D('240.0'):
    b = min(k + D('0.02') * (1 + k / 10), D('240.0')); ks.append((k, b)); k = b
pick = ks if FULL else ks[::8] + [ks[-1]]
ubs = [cell_bound(a, b)[0] for a, b in pick]
check('Theorem 11 recomputed on %d cells (%s): h < 0 on each, largest upper bound %.4e (decimal interval arithmetic; exact)' % (
      len(pick), 'all' if FULL else 'spread over the range', float(max(ubs))), all(u < 0 for u in ubs) and len(ks) == CU['cells'])
from bg_members import quantities
from bh_members_fc import deficit
from ba_ivl import I, Dl
pickm = ks if FULL else pick
mq = [quantities(Dl(I(a, b), I(0))) for a, b in pickm]; md = [sum(x.hi for x in deficit(Dl(I(a, b), I(0)))) for a, b in pickm]
check('the comb sequence lies in the class on %d cells: Qbar <= %.4f, rho <= %.1e, x >= %.2f, (P8) deficit <= %.1e (interval arithmetic; exact)' % (
      len(pickm), float(max(q[0].hi for q in mq)), float(max(q[1].hi for q in mq)), float(min(q[2].lo for q in mq)), float(max(md))),
      all(q[0].hi <= D('0.25') and q[1].hi <= D('0.125') and q[2].lo >= 1 for q in mq) and max(md) <= D('0.848'))
MB = J('bg_members.json'); MF = J('bh_members_fc.json')
check('class membership on all 1464 cells (stored): (P3)-(P5) and (P8)', MB['P3'] and MB['P4'] and MB['P5'] and MB['cells'] == 1464 and MF['P8_on_S_C'] and MF['cells'] == 1464)

# ---------------------------------------------------------------- 4. lemma checks with paper E's evaluator (stored), finite-sigma numerics
FC = J('bf_fc_full.json'); UP = J('bf_upper_full.json'); A1 = J('bg_compare.json')['A1_deficit']
check('Lemma 5 against the evaluator: every factor above its bound on all %d states (full precision); at 1e-24 the deficits 1 - a/den are below their bounds' % len(FC),
      all(v['ok'] for v in FC.values()) and len(FC) == 8 and all(v['ok'] for v in A1.values()))
check('Lemma 10 against the evaluator on the comb sequence at 1e-6, 1e-12, 1e-18, 1e-24: (a), (a\'), g below their bounds; U_P/g - 1 = %s' % ', '.join(
      '%.1e' % (v['U_main'] / v['g_true'] - 1) for v in UP['true'].values()), all(v['bound_true_ok'] for v in UP['true'].values()) and len(UP['true']) == 4)
LO = J('be_lo.json'); LA = J('be_lambda.json'); LX = J('bf_lo_cross.json')
check('Proposition 14: the leading-order identity holds to %.1e at 2000 random parameter sets; comb LO = strip LO at sigma/256' % LO['identity_max_dev'],
      LO['identity_max_dev'] < 1e-10 and all(abs(v['comb_over_shifted'] - 1) < 1e-6 for k, v in LO['table'].items() if k != '1e-6'))
MIX = {s: J('bd_opt_comb_%s.json' % s)['g_min_full'] for s in ('0.003', '0.001', '0.0003', '0.0001')}
check('combs against the mixtures (0.884): %s; leading-order crossing %.2e' % (', '.join('%s: %.3f' % kv for kv in MIX.items()), LX['comb_LO_cross']),
      MIX['0.003'] > 0.884 > MIX['0.001'] and 200 < LX['ratio'] < 300)
check('combs have no deep valleys: Lambda* = %s finite' % ', '.join('%.2f' % LA['comb_%s' % s]['Lambda_star'] for s in ('1e-06', '1e-12', '1e-24')),
      all(np.isfinite(LA['comb_%s' % s]['Lambda_star']) for s in ('1e-06', '1e-12', '1e-24')))

# ---------------------------------------------------------------- 5. the manuscript
import bs_numbers
tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
N = bs_numbers.numbers(); miss = [x['text'] for x in N if x['text'] not in tex]
check('every number of the manuscript (v1.3) recomputed from results/ is found in paper/sections (%d numbers)' % len(N), not miss and len(N) >= 200, ', '.join(miss[:5]))

# ---------------------------------------------------------------- 6. the demo notebook (stored with its outputs)
NBK = json.load(open('demo/hidden_ripple_demo.ipynb'))
cc = [c for c in NBK['cells'] if c['cell_type'] == 'code']; outs = [o for c in cc for o in c['outputs']]
txt = ''.join(''.join(o.get('text', '')) for o in outs)
check('demo notebook: %d code cells executed in order, no error output, %d figures, the recomputed certificate cells print h < 0' % (
      len(cc), sum('image/png' in o.get('data', {}) for o in outs)),
      [c['execution_count'] for c in cc] == list(range(1, len(cc) + 1)) and not any(o['output_type'] == 'error' for o in outs)
      and sum('image/png' in o.get('data', {}) for o in outs) >= 3 and txt.count('h < 0: True') == 3 and 'h < 0: False' not in txt)

# ---------------------------------------------------------------- 7. v1.1: theta_tau forms, Lemma BK1, the perturbed chain, Lemma BJ1, Lemma BL7
t7 = time.time(); np.seterr(all='ignore')
N8 = J('bl_narrow80.json')
check('Lemma BK1 (tau0 = 80, stored): %d cells cover {1 <= x <= 80, x/80 <= b <= 1}; every cell closes, smallest margin %.3f' % (N8['cells'], N8['worst_margin']),
      N8['all_ok'] and N8['cells'] == 57600 and N8['worst_margin'] > 2.5)
import bk_narrow_lemma as BK
xs = np.geomspace(1.0, 80.0, 241); step = 1 if FULL else 12; wm = np.inf; allok = True; nc = 0
for xa, xb in zip(xs[:-1][::step], xs[1:][::step]):
    bs = np.geomspace(xa / 80.0, 1.0, 241)
    for ba, bb in zip(bs[:-1], bs[1:]):
        r = BK.cell(xa, xb, ba, bb); nc += 1; allok &= r['ok']; wm = min(wm, r['margin'])
check('Lemma BK1 recomputed on %d cells (%s): all close, smallest margin %.3f (floating point, relative %.0e)' % (nc, 'all' if FULL else 'every 12th x-cell', wm, RTOL),
      allok and wm >= N8['worst_margin'] * (1 - RTOL) and (not FULL or rel(wm, N8['worst_margin']) <= RTOL))
import bl_theta as BT
TH = J('bl_theta.json'); t80 = TH['eps_table_tau80']; e80 = BT.eps_table()
check('theta_tau corrections for tau >= 80 (Lemmas BL1-BL6): eps_theta = %.2e recomputed (stored %.2e; relative %.0e); far below eps_in = 1e-6' % (
      e80['eps_theta'], t80['eps_theta'], RTOL), rel(e80['eps_theta'], t80['eps_theta']) <= RTOL and t80['eps_theta'] < 1e-12)
CH = TH['checks']
check('theta_tau lemmas against direct quadrature (stored, numerical): exact identity of Lemma BL3 to %.0e; BL1, BL2, BL3, BL5 hold on %d states (tau 3 ... 25)' % (
      max(c['identity_rel_err'] for c in CH), len(CH)), TH['all_checks_ok'] and max(c['identity_rel_err'] for c in CH) < 1e-9)
pickst = CH if FULL else [c for c in CH if c['tau'] == 12.0 and c['x'] == 2.0 and c['sigma'] == 1e-24]
STATES = {(1.0,): ([0.25], 1e-3, 1e-4), (4.0,): ([0.12, 0.01], 3e-3, 2e-4), (2.0,): ([0.2], 0.3, 0.02)}
okst = True; maxdev = 0.0
for c in pickst:
    q0, L2, s2 = STATES[(c['x'],)]; r = BT.check_state(c['sigma'], c['b'], c['x'], q0, L2, s2)
    okst &= r['AX3_ok'] and r['Jww_ok'] and r['W_ok'] and r['keff_ok'] and r['BL7_ok'] and r['identity_rel_err'] < 1e-9
    maxdev = max(maxdev, rel(r['two_Nphi'], c['two_Nphi']))
check('theta_tau lemmas recomputed on %d state(s) by direct quadrature: all bounds hold, 2N_phi agrees (relative %.1e; numerical)' % (len(pickst), maxdev), okst and maxdev < 1e-6)
CC7 = {}
for f in sorted(glob.glob('results/bl_chain_cells_part*.jsonl')):
    for line in open(f):
        if line.strip(): d = json.loads(line); CC7[d['i']] = d
c1m = max(d['c1_needed'] for d in CC7.values()); mg = min(d['margin_c1_1.7'] for d in CC7.values())
check('Theorem 9 with every input lemma perturbed by eps_in = 1e-6 (stored): %d cells, every condition holds; largest constant needed %.4f, smallest margin at c1 = 1.7: %.2e' % (
      len(CC7), c1m, mg), sorted(CC7) == list(range(421)) and all(d['ok'] and d['eps_in'] == 1e-6 for d in CC7.values()) and c1m < 1.7 and mg > 0
      and max(d['absorb_ratio'] for d in CC7.values()) <= 1)
import bl_chain as BLC
redo7 = sorted(CC7) if FULL else list(range(0, 421, 30)) + [420]; w7 = 0.0; ok7 = True
for i in redo7:
    r = BLC.cell(CC7[i]['la'], CC7[i]['lb'], meta['S1_P'], meta['S_i_at_1']); ok7 &= r['ok']; w7 = max(w7, rel(r['L_cell'], CC7[i]['L_cell']))
check('perturbed chain recomputed on %d cells (%s): every condition holds; L_cell agrees (relative %.1e <= %.0e; floating point)' % (
      len(redo7), 'all' if FULL else 'every 30th cell', w7, RTOL), ok7 and w7 <= RTOL)
TL7 = J('bl_chain_tail.json'); r6 = BLC.cell(1e6, 1e6 * 1.0000001, meta['S1_P'], meta['S_i_at_1'])
check('Lemma 7 (tail) with perturbed inputs: bound increases from %.6f; direct cells at 1e6, 1e7, 1e8 hold (1e6 recomputed: K relative %.1e)' % (
      TL7['T_at_l_end'], rel(r6['K'], TL7['direct']['1e+06']['K'])), TL7['T_increasing'] and all(v['ok'] and v['K'] <= v['K_bound_from_tail'] for v in TL7['direct'].values())
      and r6['ok'] and rel(r6['K'], TL7['direct']['1e+06']['K']) <= RTOL)
import bl_bj1 as BJ1
BJ = J('bl_bj1.json'); sig_list = (24, 48, 100, 200, 300) if FULL else (24,); okb = True; devb = 0.0
for e in sig_list:
    rb = BJ1.constants(10.0 ** (-e)); okb &= rb['ok']; devb = max(devb, rel(rb['A_min'], BJ['1e-%d' % e]['A_min']), rel(rb['step3a']['margin'], BJ['1e-%d' % e]['step3a']['margin']))
check('Lemma BJ1 (the former (P8)) recomputed at sigma = %s: every step closes, A_min = %.6f, 3(i) margin %.4f (relative %.1e)' % (
      ', '.join('1e-%d' % e for e in sig_list), BJ['1e-24']['A_min'], BJ['1e-24']['step3a']['margin'], devb), okb and BJ['all_ok'] and devb <= RTOL)
IA = BJ1.step3a_interval()
check('Lemma BJ1 step 3(i) in outward-rounded interval arithmetic: g(L1) > right side (margin >= %.4f) and the unimodality test (decimal intervals; exact)' % IA['margin_lower'],
      IA['g_gt_rhs'] and IA['unimodal'] and IA['margin_lower'] > 1.2 and same(IA, BJ['step3a_interval']))
if FULL:
    mono = BJ1.mono_grid()
    check('Lemma BJ1 monotonicity table recomputed on the grid: %d rows, every declared direction confirmed (numerical)' % len(mono), all(m['ok_sigma'] and m['ok_c'] for m in mono))
else:
    check('Lemma BJ1 monotonicity table (stored grid check): %d rows confirmed (numerical)' % len(BJ['monotonicity']), all(m['ok_sigma'] and m['ok_c'] for m in BJ['monotonicity']))
import bm_bl7 as B7
M7 = J('bm_bl7.json'); c7 = B7.comb_corrections()
check('Lemma BL7 on the comb sequence: tau >= %.3e, every theta correction to h <= 10^(%.4e) << the margin 1.35e-4 (decimal intervals; exact)' % (
      c7['tau_min'], c7['log10_correction_bound']), c7['negligible'] and same(c7, M7['comb']))
check('Lemma BL7 against direct quadrature (stored, numerical): 2N_phi below the bound on %d states (largest ratio %.6f)' % (
      len(M7['checks']), max(c['BL7_ratio'] for c in M7['checks'])), M7['all_checks_ok'])
print('section 7 (v1.1): %.0f s' % (time.time() - t7))

# ---------------------------------------------------------------- 8. v1.2: App. E (P5), Lemma 20 (vi), the audit at interior points
t8 = time.time()
import subprocess
def v12(comp):
    out = subprocess.run([sys.executable, 'verification/bq_v12_checks.py', comp] + (['--full'] if FULL else []), capture_output=True, text=True)
    try: return json.loads(out.stdout.strip().split('\n')[-1])
    except Exception: return {'ok': False, 'error': (out.stderr or out.stdout)[-300:]}
how = 'all' if FULL else 'a subset'
r = v12('E1i')
check('Lemma 29(i) (strips, 1e-12 <= x <= 1): %d cells, all close, smallest ratio %.3f; recomputed on %s of the %d x-cells (relative %.1e; floating point)' % (
      r.get('cells', 0), r.get('worst_margin', 0), how, r.get('of', 0), r.get('dev', 1)), r['ok'] and r.get('stored_ok', False), r.get('error', ''))
r = v12('E1ii')
check('Lemma 29(ii) (strips, x < 1e-12, b >= max(c\'sigma, 2e-27), sigma_hat = 1e-24): %d cells, all close, smallest ratio %.3f; b-cells cover [2e-27, 1]; in-cell audit %d points, none on the wrong side; recomputed on %s (relative %.1e)' % (
      r.get('cells', 0), r.get('worst_margin', 0), r.get('audit_points', 0), how, r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v12('E1iii')
check('Lemma 29(iii) (at most six harmonics, 1e-12 <= x <= 1): %d cells for N = 6, smallest ratio %.3f; the certificate closes for N = %s; recomputed on %s (relative %.1e)' % (
      r.get('N6_cells', 0), r.get('N6_margin', 0), r.get('closes'), how, r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v12('E2')
check('Lemma 30 (scaling in b, proof): numerical illustration on %d random states, F/b never decreases and Lambda_1 never decreases as b decreases (numerical)' % r.get('states', 0),
      r['ok'], r.get('error', ''))
r = v12('proofs')
check('Lemmas 31, 32, 33 and Theorems 34, 36: constants in outward-rounded interval arithmetic, identical to the stored ones: T = %.2f > 250, (a\') >= %.4f (%.2f c_*), eps_R <= %.1e; g >= %.4f (strips), %.4f (N = 6) (decimal intervals; exact)' % (
      r.get('T', 0), r.get('a_prime_lower', 0), r.get('ratio_C', 0), r.get('eps_R', 1), r.get('g_strip', 0), r.get('g_N6', 0)), r['ok'], r.get('error', ''))
r = v12('E5')
check('Lemma 33 (Lemma 28 for strips, 250 sigma^2 < x <= 1) recomputed at sigma = %s: A_min = %.6f, step 3(i) margin >= %.4f in interval arithmetic, g >= %.4f (relative %.1e)' % (
      ', '.join('1e-%d' % e for e in r.get('sigmas', [])), r.get('A_min', 0), r.get('step3a_interval_margin', 0), r.get('g_lower', 0), r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v12('BN1')
check('Lemma 28 for strips on [1e-12, 1] and E_R there (the first version of App. E, stage BN) recomputed: A_min = %.6f, E_R <= %.1e (relative %.1e)' % (
      r.get('A_min', 0), r.get('E_R_sup', 1), r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v12('E7')
check('Lemma 35 (at most six harmonics) recomputed for N = %s: A_min = %.4f, E_R <= %.1e (relative %.1e)' % (r.get('N'), r.get('A_min_6', 0), r.get('E_R_6', 1), r.get('dev', 1)),
      r['ok'], r.get('error', ''))
r = v12('sigma')
check('Lemma 20 (vi) (every sigma <= 1e-24), numerical illustration: %d (cell, sigma) pairs at sigma = 1e-30 ... 1e-300 and %d points of Lemma 29(ii) below sigma_hat all close (numerical)' % (
      r.get('pairs', 0), r.get('caseB_points', 0)), r['ok'], r.get('error', ''))
AUD = {}
for line in open('results/bp_audit.jsonl'):
    if line.strip(): d = json.loads(line); AUD[d['key']] = d
oth = [k for k, d in AUD.items() if d.get('violations') and 'v1.1 code' not in k]
check('audit at interior points (stored, App. B): %d checks, %d cells, %d interior points; no quantity on the wrong side except H1 (and what is computed from it) in the v1.1 code of Lemma 20 (numerical)' % (
      len(AUD), sum(d.get('cells', 0) for d in AUD.values()), sum(d.get('points', 0) for d in AUD.values())),
      not oth and AUD['bk1(general_tau80, v1.1 code):H1']['violations'] > 0 and all(AUD[k]['violations'] == 0 for k in AUD if 'v1.2 code' in k), ', '.join(oth[:3]))
r = v12('audit')
check('audit %s: %d checks, %d interior points; the v1.1 code of Lemma 20 is found again (%s), nothing else on the wrong side%s (numerical)' % (
      'recomputed in full' if FULL else 'recomputed in reduced form', r.get('checks', 0), r.get('points', 0), ', '.join(r.get('v1.1_code_reversals', [])),
      '; counts identical to the stored audit' if FULL else ''), r['ok'], r.get('error', ''))
print('section 8 (v1.2): %.0f s' % (time.time() - t8))

# ---------------------------------------------------------------- 9. v1.3: App. F (at most N harmonics), Theorem 11 beyond k0 = 240
t9 = time.time()
def v13(comp):
    out = subprocess.run([sys.executable, 'verification/bs_v13_checks.py', comp] + (['--full'] if FULL else []), capture_output=True, text=True)
    try: return json.loads(out.stdout.strip().split('\n')[-1])
    except Exception: return {'ok': False, 'error': (out.stderr or out.stdout)[-300:]}
r = v13('tail')
check('Theorem 11 for k0 >= 240 (sigma < 1e-100000): U_P = (1 + R_U/k0^2)/(2 k0) with R_U <= %.6f, margin %.4f > 0 at k0 = 240 and increasing; members, BL7 tail; recomputed, identical to the stored result: %s (decimal intervals; exact)' % (
      r.get('R_U_hi', 9), r.get('lhs_at_K0', 0), r.get('identical_to_stored')), r['ok'], r.get('error', ''))
r = v13('stress')
check('the asymptotic forms of Theorem 11 against a direct 300-digit evaluation at k0 = 240 ... 1e8: %d comparisons, %d outside their enclosure (numerical)' % (
      r.get('checks', 0), r.get('failures', -1)), r['ok'], r.get('error', ''))
r = v13('F2i')
check('Lemma F2(i) (at most N harmonics, 1e-12 <= x <= 1, tau <= 80): %d cells for N = 16 (smallest ratio %.3f) and N = 20 (%.3f), all close; recomputed on %d rows%s (relative %.1e; floating point)' % (
      (r.get('N16') or {}).get('cells', 0), (r.get('N16') or {}).get('worst', 0), (r.get('N20') or {}).get('worst', 0), r.get('rows', 0),
      ' with the in-cell audit, every N' if FULL else '', r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v13('F2ii')
check('Lemma F2(ii) (N = 16, x < 1e-12, b >= c\' 1e-24, c\' = 2e-5): %d cells, smallest ratio %.3f, b-cells cover [2e-29, 1], in-cell audit %d points; recomputed on %d of %d b-cells (relative %.1e)' % (
      r.get('cells', 0), r.get('worst', 0), r.get('audit_points', 0), r.get('rows', 0), r.get('of', 0), r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v13('intervals')
check('Lemmas F4, F5 and eps_R of Lemma F6 (interval arithmetic) recomputed for N = %s, identical to the stored constants: (a\')/c_* >= %.3f, T_16 >= %.4f, eps_R <= %.4f (decimal intervals; exact)' % (
      '6, 8, 12, 16' if FULL else '16', (r.get('N16') or {}).get('a_prime_ratio', 0), (r.get('N16') or {}).get('T', 0), (r.get('N16') or {}).get('eps_R', 9)), r['ok'], r.get('error', ''))
r = v13('F6')
check('Lemma F6 (Lemma 28 with N harmonics) recomputed: A_min = %.5f (N = 16, x -> 0), %.5f (N = 20, x >= 1e-12); where it stops (N = 20 below 1e-12, N = 24) is recorded (relative %.1e)' % (
      r.get('A_min_16', 0), r.get('A_min_20', 0), r.get('dev', 1)), r['ok'], r.get('error', ''))
r = v13('F7')
check('Lemma F7 and Theorems F8, F9 recomputed: sup x f* = %.4f (the same for every N checked), g >= %.4f (N <= 16, 0 < x < 1), %.4f (N <= 20, x >= 1e-12) (floating point)' % (
      r.get('sup_x_fstar', 9), r.get('g_16', 0), r.get('g_20', 0)), r['ok'], r.get('error', ''))
r = v13('remark')
check('remark of App. F, a_n >= 2 - n^2 (2 - a_1) (proof): %d random non-negative profiles, %d violations (numerical)' % (r.get('profiles', 0), r.get('violations', -1)),
      r['ok'], r.get('error', ''))
if FULL:
    r = v13('rerun')
    check('every run behind results/br_windows.json recomputed from scratch in a temporary folder (%s; %.0f s): identical to the stored file up to relative 1e-9' % (
          ', '.join(r.get('keys', [])), r.get('seconds', 0)), r['ok'], r.get('error', ''))
print('section 9 (v1.3): %.0f s' % (time.time() - t9))
print('elapsed %.0f s (%s)' % (time.time() - t0, 'full recomputation' if FULL else 'default'))
finish()
