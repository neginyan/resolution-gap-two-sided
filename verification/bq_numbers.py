"""v1.2: every number of the manuscript re-derived from results/.  The v1.1 list (bm_numbers.py; its Lemma 20 margin is now recomputed with the
corrected certificate), plus the numbers of Lemma 20 (vi), Appendix E (strips without (P5), windows with at most six harmonics) and the audit of
Appendix B.  Bounds are rounded outward (ceil for upper bounds, floor for lower bounds).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, math
from bm_numbers import numbers as bm_numbers, up, dn, sci

R = lambda f: json.load(open('results/' + f))


def lo(v): return v[0] if isinstance(v, (list, tuple)) else v
def hi(v): return v[1] if isinstance(v, (list, tuple)) else v


def audit():
    A = {}
    for line in open('results/bp_audit.jsonl'):
        if line.strip(): d = json.loads(line); A[d['key']] = d
    return A


def numbers():
    n = bm_numbers()
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    N8 = R('bl_narrow80.json'); m = dn(N8['worst_margin'], 2)
    add('a margin $%.2f$ instead of $2.80$' % m, N8['worst_margin'], 'bl_narrow80:worst_margin (abstract)')
    add('the smallest margin is $%.2f$ instead of $2.80$' % m, N8['worst_margin'], 'bl_narrow80:worst_margin (introduction)')
    u_hat = 1e-24 * 80 / (4 * 0.134 / 0.9)                                     # u <= sigma/(4 A'), A' >= c_* b >= c_*/80 (kappa_w >= 1)
    assert u_hat < 1e-21
    add('cells $\\hat u<10^{-21}$', u_hat, 'Lemma 20 (vi): u_hat <= 1e-24 * 80/(4 c_*)')
    # Appendix E
    BN = R('bn_strip_x.json')['bk1_strip']
    add('$%d$ geometric $x$-cells on $[10^{-12},1]$ times $%d$ $b$-cells' % (BN['x_cells'], BN['b_cells_per_x']), BN['cells'], 'bn_strip_x:cells')
    add('$%d$ cells; every cell closes, smallest ratio $%.3f$' % (BN['cells'], dn(BN['worst_margin'], 3)), BN['worst_margin'], 'bn_strip_x:worst_margin')
    S = R('bp_strip_p5.json'); B = S['case_B']
    add('$%d$ geometric $b$-cells on $[2\\times10^{-27},1]$' % B['b_cells'], B['b_cells'], 'bp_strip_p5:case_B b_cells')
    add('$%d$ cells; every cell closes, smallest ratio $%.3f$' % (B['cells'], dn(B['worst_margin'], 3)), B['worst_margin'], 'bp_strip_p5:case_B worst_margin')
    add('($%d$ points for Lemma' % B['audit_points'], B['audit_points'], 'bp_strip_p5:case_B audit points')
    assert B['all_ok'] and B['audit_violations'] == 0
    W = R('bp_windowsN.json'); W6 = W['N=6']
    add('$120\\times120=%d$ cells; every cell closes, smallest ratio $%.3f$' % (W6['bk1_cells'], dn(W6['bk1_worst_margin'], 3)), W6['bk1_worst_margin'], 'bp_windowsN:N=6')
    add('$%d$ for each $N$' % W6['bk1_audit_points'], W6['bk1_audit_points'], 'bp_windowsN:audit points per N')
    Q = R('bq_proofs.json'); C = Q['case_C']; A1 = Q['case_A1']; ER = Q['eps_R']
    add('\\le%.4f$' % up(hi(C['kappa_w']), 4), C['kappa_w'], 'bq_proofs:case_C kappa_w')
    add('\\le%.3f$' % up(hi(C['V_max']), 3), C['V_max'], 'bq_proofs:case_C V_max')
    add('$\\ge%.7f$' % dn(lo(C['main']), 7), C['main'], 'bq_proofs:case_C main')
    alE = hi(C['E']) / math.sqrt(2) * (1 + 1e-12)
    add('$\\alpha E\\le%s$' % sci(alE, 1, up), alE, 'bq_proofs:case_C alpha E')
    add('\\ge%.4f>%.2f\\,c_* .' % (dn(C['a_prime_lower'], 4), dn(C['ratio_lower'], 2)), C['a_prime_lower'], 'bq_proofs:case_C (a\') lower')
    add('$(a\')\\ge%.4f>%.2f\\,c_*$' % (dn(C['a_prime_lower'], 4), dn(C['ratio_lower'], 2)), C['ratio_lower'], 'bq_proofs:case_C ratio (statement)')
    add('$W\\le x[(1+\\delta_\\tau)q^2g_s/(2(1-\\Qb))+g_1qe_1]/(1-g_1qe_1)\\le%.7f\\,x$' % up(hi(A1['W_coef']), 7), A1['W_coef'], 'bq_proofs:case_A1 W coefficient')
    add('$b+W\\le%.7f\\,x$' % up(hi(A1['b_plus_W_over_x']), 7), A1['b_plus_W_over_x'], 'bq_proofs:case_A1 b + W')
    add('\\ge%.2f$, and' % dn(lo(A1['T']), 2), A1['T'], 'bq_proofs:case_A1 T')
    add('$x>%.1f\\,\\sigma^2$' % dn(lo(A1['T']), 1), A1['T'], 'bq_proofs:case_A1 T (statement)')
    assert hi(A1['kappa_w']) - 1 <= 1e-17
    add('$A_0\\le%s$' % sci(hi(ER['A0']), 1, up), ER['A0'], 'bq_proofs:eps_R A0')
    add('$B_0\\le%s$' % sci(hi(ER['B0']), 1, up), ER['B0'], 'bq_proofs:eps_R B0')
    add('$C_0\\le%s$' % sci(hi(ER['C0']), 1, up), ER['C0'], 'bq_proofs:eps_R C0')
    add('\\varepsilon_R=5(A_0+B_0+C_0)\\le%s$' % sci(ER['eps_R_upper'], 1, up), ER['eps_R_upper'], 'bq_proofs:eps_R')
    add('is $\\le%s$ relative' % sci(ER['phi_remainder_upper'], 1, up), ER['phi_remainder_upper'], 'bq_proofs:Phi remainder')
    A2 = S['case_A2_1e-24']
    add('$A_{\\min}=%.5f$' % dn(A2['A_min'], 5), A2['A_min'], 'bp_strip_p5:case A2 A_min')
    add('A_{\\min}=%.4f$.' % dn(A2['A_min'], 4), A2['A_min'], 'bp_strip_p5:case A2 A_min (statement)')
    a2c = A2['step2b']['a_lower_over_c']; e = int(math.floor(math.log10(a2c)))
    add('$(a)/c\\ge%d\\times10^{%d}$' % (int(a2c / 10 ** e), e), a2c, 'bp_strip_p5:case A2 step 2(i)')
    add('step~3(i) margin $%.4f$ (interval' % dn(S['case_A2_step3a_interval']['margin_lower'], 4), S['case_A2_step3a_interval']['margin_lower'], 'bp_strip_p5:step 3(i)')
    r3 = A2['step3b']['W_lower_over_W_cap']; e = int(math.floor(math.log10(r3)))
    add('ratio $%.1f\\times10^{%d}$' % (dn(r3 / 10 ** e, 1), e), r3, 'bp_strip_p5:case A2 step 3(ii)')
    add('($%d$ cells)' % A2['x_cells'], A2['x_cells'], 'bp_strip_p5:case A2 x-cells')
    assert all(S['case_A2_1e-%d' % k]['ok'] for k in (24, 48, 100, 300))
    CS = Q['conclusion_strip']
    add('$\\kappa_2=%.5f$' % up(hi(CS['kappa2']), 5), CS['kappa2'], 'bq_proofs:kappa2')
    add('$\\keff^2\\le%.4f$' % up(hi(CS['k_eff2_max']), 4), CS['k_eff2_max'], 'bq_proofs:k_eff^2 (strip)')
    add('\\sqrt{%.4f})\\ge%.4f$' % (up(hi(CS['k_eff2_max']), 4), dn(CS['g_lower'], 4)), CS['g_lower'], 'bq_proofs:g lower (strip)')
    C6 = Q['conclusion_N']['6']
    add('at most\n$%s$ on the range' % sci(W6['E_R_sup'], 1, up), W6['E_R_sup'], 'bp_windowsN:E_R (N = 6)')
    add('$2E_R+\\epsilon_\\theta\\le%s<' % sci(C6['two_ER_plus_eps_theta_upper'], 1, up), C6['two_ER_plus_eps_theta_upper'], 'bq_proofs:2 E_R + eps_theta (N = 6)')
    add('$A_{\\min}=%.4f$, step~3(i) margin $%.4f$' % (dn(W6['A_min'], 4), dn(W6['step3a_margin'], 4)), W6['A_min'], 'bp_windowsN:A_min, 3(i) (N = 6)')
    add('$A_1A_2A_3\\ge%.4f$' % dn(W6['A_min'], 4), W6['A_min'], 'bp_windowsN:A_min (N = 6, statement)')
    xz = W6['conclusion']['sup_x(1+Z2)']
    add('x(1+Z_2(x))\\le%.3f$' % up(xz, 3), xz, 'bp_windowsN:sup x(1 + Z2) (N = 6)')
    add('$\\keff^2\\le%.3f$ and $g\\ge0.9\\times%.4f/(2\\sqrt{%.3f})\\ge%.4f$' % (up(hi(C6['k_eff2_max']), 3), dn(W6['A_min'], 4), up(hi(C6['k_eff2_max']), 3),
        dn(C6['g_lower'], 4)), C6['g_lower'], 'bq_proofs:g lower (N = 6)')
    rows = [('1 (strip)', R('bn_strip_x.json')['bk1_strip']['worst_margin'], A2['A_min'], CS['g_lower'], True)]
    for N in (2, 3, 4, 6, 8, 12):
        w = W['N=%d' % N]; rows.append((str(N), w['bk1_worst_margin'], w['A_min'], Q['conclusion_N'][str(N)]['g_lower'] if str(N) in Q['conclusion_N'] else None, w['closes']))
    for lab, mg, am, g, ok in rows:
        add('%s & %.3f & %.4f & %s & %s\\\\' % (lab, dn(mg, 3), dn(am, 4), '%.4f' % dn(g, 4) if g is not None else '--', 'yes' if ok else 'no'), mg, 'Table 10 row N = %s' % lab)
    # audit (App. B)
    AU = audit()
    nck = len(AU); ncell = sum(d.get('cells', 0) for d in AU.values()); npt = sum(d.get('points', 0) for d in AU.values())
    nl = sum(1 for k in AU if k.startswith('lcells')); nup = AU['upper:h(k) at interior points against the cell bound (bf_cert_upper)']['cells']
    mf = AU['bk1(strip_x_lt_1, v1.1 code):main_fac']['violations']
    add('in all $%d$ checks, $%d$ cells and $%d$ interior points' % (nck, ncell, npt), npt, 'bp_audit:totals')
    add('$%d$ cells of the lower chain' % nl, nl, 'bp_audit:lcells')
    add('$%d$ cells of the upper side' % nup, nup, 'bp_audit:upper cells')
    add('at $%d$ points of the' % mf, mf, 'bp_audit:main factor reversals (v1.1 code, strips)')
    add('($%.1f\\times10^6$ points)' % dn(npt / 1e6, 1), npt, 'bp_audit:points (Table 1)')
    bad = [k for k, d in AU.items() if d.get('violations') and 'v1.1 code' not in k]
    assert not bad, bad
    return n


if __name__ == '__main__':
    import glob
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d missing' % (len(N), len(miss)))
    for x in miss: print('  MISSING:', repr(x['text']), '<-', x['source'])
