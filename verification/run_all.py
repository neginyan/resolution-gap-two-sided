"""Single entry point: checks every claim of the paper that rests on stored results or on a computer-assisted certificate.

  python verification/run_all.py              # default (a few minutes on one core): hashes, stored results, every number of the manuscript,
                                              # and a recomputation of representative cells of every certificate
  python verification/run_all.py --recompute  # in addition, recompute every certificate in full (about 20 minutes on one core; writes nothing to results/)

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
import bh_numbers
tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
N = bh_numbers.numbers(); miss = [x['text'] for x in N if x['text'] not in tex]
check('every number of the manuscript recomputed from results/ is found in paper/sections (%d numbers)' % len(N), not miss and len(N) >= 55, ', '.join(miss[:5]))

# ---------------------------------------------------------------- 6. the demo notebook (stored with its outputs)
NBK = json.load(open('demo/hidden_ripple_demo.ipynb'))
cc = [c for c in NBK['cells'] if c['cell_type'] == 'code']; outs = [o for c in cc for o in c['outputs']]
txt = ''.join(''.join(o.get('text', '')) for o in outs)
check('demo notebook: %d code cells executed in order, no error output, %d figures, the recomputed certificate cells print h < 0' % (
      len(cc), sum('image/png' in o.get('data', {}) for o in outs)),
      [c['execution_count'] for c in cc] == list(range(1, len(cc) + 1)) and not any(o['output_type'] == 'error' for o in outs)
      and sum('image/png' in o.get('data', {}) for o in outs) >= 3 and txt.count('h < 0: True') == 3 and 'h < 0: False' not in txt)
print('elapsed %.0f s (%s)' % (time.time() - t0, 'full recomputation' if FULL else 'default'))
finish()
