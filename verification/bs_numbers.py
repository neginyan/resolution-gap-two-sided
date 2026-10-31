"""v1.3: every number of the manuscript re-derived from results/.  The v1.2 list (bq_numbers.py), plus the numbers of App. F (windows with at most
N harmonics without (P5); br_windows.json, bp_windowsN.json) and of Theorem 11 beyond k0 = 240 (App. B; bs_upper_tail.json, bs_upper_tail_stress.json).
Bounds are rounded outward (ceil for upper bounds, floor for lower bounds).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, math
from bm_numbers import up, dn, sci
from bq_numbers import numbers as bq_numbers

R = lambda f: json.load(open('results/' + f))


def lo(v): return v[0] if isinstance(v, (list, tuple)) else v
def hi(v): return v[1] if isinstance(v, (list, tuple)) else v


def numbers():
    n = bq_numbers()
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    B = R('br_windows.json'); K = {6: 'N=6', 8: 'N=8', 12: "N=12, c'=0.0002", 16: "N=16, c'=0.00002", 20: 'N=20', 24: 'N=24'}
    bk = {N: B[K[N]]['bk1_x_ge_1e-12'] for N in K}
    assert all(bk[N]['all_ok'] and bk[N]['audit_violations'] == 0 and bk[N]['cells'] == 14400 for N in K)
    assert dn(bk[6]['worst_margin'], 3) == dn(bk[8]['worst_margin'], 3)
    # Lemma F2
    add('$14400$ cells for each $N$', bk[16]['cells'], 'br_windows:bk1 cells')
    add('smallest ratio $%.3f$ ($N=6,8$),\n$%.3f$ ($N=12$), $%.3f$ ($N=16$), $%.3f$ ($N=20$)' % tuple(dn(bk[N]['worst_margin'], 3) for N in (6, 12, 16, 20)),
        bk[20]['worst_margin'], 'br_windows:bk1 worst margins')
    CB = B[K[16]]['caseB']; assert CB['all_ok'] and CB['audit_violations'] == 0
    add('$%d$ geometric $b$-cells on $[2\\times10^{-29},1]$' % CB['b_cells'], CB['b_cells'], 'br_windows:caseB b-cells (N = 16)')
    add('$%d$ cells; smallest ratio $%.3f$' % (CB['cells'], dn(CB['worst_margin'], 3)), CB['worst_margin'], 'br_windows:caseB (N = 16)')
    add('($%d$ points for (i) and $%d$ for (ii)' % (bk[16]['audit_points'], CB['audit_points']), CB['audit_points'], 'br_windows:audit points')
    # Lemma F4
    C = B[K[16]]['caseC']; assert C['ok']
    add('$(a\')\\ge%.4f>%.1f\\,c_*$' % (dn(C['a_prime_lower'], 4), dn(C['ratio_lower'], 1)), C['a_prime_lower'], 'br_windows:caseC (N = 16, statement)')
    add('$(W/b)_{\\max}\\le%.2f$ for $N=16$' % up(hi(C['W_over_b_max']), 2), C['W_over_b_max'], 'br_windows:caseC W/b')
    add('(1-\\Qb g_1)\\le%.4f$' % up(hi(C['kappa_w']), 4), C['kappa_w'], 'br_windows:caseC kappa_w')
    add('\\le%.2f$.\n\\emph{(2)' % up(hi(C['V_max']), 2), C['V_max'], 'br_windows:caseC V_max')
    add('X_{\\min}=e^{-\\alpha^2V_{\\max}/8}-\\Qb\\ge%.4f>0$' % dn(lo(C['X_min']), 4), C['X_min'], 'br_windows:caseC X_min')
    add('the first term is $\\ge%s$' % sci(lo(C['main']), 4, dn), C['main'], 'br_windows:caseC main')
    add('$\\alpha E\\le%s$' % sci(hi(C['err']), 1, up), C['err'], 'br_windows:caseC alpha E')
    add('\\big/\\big(2\\theta(1+(W/b)_{\\max})c\'\\big)\\ge%.4f$' % dn(C['a_prime_lower'], 4), C['a_prime_lower'], 'br_windows:caseC (a\') lower')
    # Lemma F5 and the split
    T = {N: B[K[N]]['caseA1'] for N in (6, 8, 12, 16)}; T20 = B['N=20']['small_x_attempt']['caseA1']
    add('$T_N\\ge%.2f$, $%.2f$, $%.2f$, $%.3f$ for $N=6$, $8$, $12$, $16$ ($%.2f$ for $N=20$)' % (
        dn(T[6]['T_lower'], 2), dn(T[8]['T_lower'], 2), dn(T[12]['T_lower'], 2), dn(T[16]['T_lower'], 3), dn(T20['T_lower'], 2)), T[16]['T_lower'], 'br_windows:caseA1 T')
    add('$W_c\\le%.5f$, $T_{16}\\ge%.4f$' % (up(hi(T[16]['W_coef']), 5), dn(T[16]['T_lower'], 4)), T[16]['W_coef'], 'br_windows:caseA1 (N = 16)')
    add('($%d$, $%d$, $%d$, $%d$ for $N=6,8,12,16$)' % tuple(int(T[N]['T_split']) for N in (6, 8, 12, 16)), T[16]['T_split'], 'br_windows:T_split')
    assert all(T[N]['ok'] for N in T)
    # Lemma F6
    A2 = B[K[16]]['caseA2']; E = B[K[16]]['eps_R']; L2 = B['N=20']['bj1_x_ge_1e-12']; assert A2['ok'] and L2['ok'] and E['ok']
    add('$A_{\\min}=%.4f$ for $N\\le16$' % dn(A2['A_min'], 4), A2['A_min'], 'br_windows:caseA2 A_min (statement)')
    add('$A_{\\min}=%.4f$ for $N\\le20$' % dn(L2['A_min'], 4), L2['A_min'], 'br_windows:bj1 N = 20 (statement)')
    add('($\\beta/\\tau\\le%.4f$ for $N=16$)' % up(E['beta_over_tau'], 4), E['beta_over_tau'], 'br_windows:eps_R beta/tau')
    add('$3/\\tau-\\tfrac12+1/\\sqrt{2T_s}\\le%.2f<0$' % up(E['slope'], 2), E['slope'], 'br_windows:eps_R slope')
    add('$A_0\\le%s$, $B_0\\le%s$,\n$C_0\\le%s$' % (sci(hi(E['A0']), 2, up), sci(hi(E['B0']), 2, up), sci(hi(E['C0']), 2, up)), E['A0'], 'br_windows:eps_R A0, B0, C0')
    e6, e8, e12 = (B[K[N]]['eps_R']['eps_R'] for N in (6, 8, 12))
    add('$\\varepsilon_R\\le%.5f$ ($N=16$;\n$%s$, $%s$, $%s$ for $N=6,8,12$)' % (up(E['eps_R'], 5), sci(e6, 1, up), sci(e8, 1, up), sci(e12, 1, up)), E['eps_R'], 'br_windows:eps_R')
    add('$2\\Qb(\\nu_0+\\tau\\nu_2)\\le%s$' % sci(E['phi_remainder'], 1, up), E['phi_remainder'], 'br_windows:Phi remainder')
    add('$\\sup E_R\\le%s$' % sci(L2['E_R_sup'], 2, up), L2['E_R_sup'], 'br_windows:E_R sup (N = 20)')
    add('\\le%s$.' % sci(L2['eps'], 2, up), L2['eps'], 'br_windows:eps (N = 20)')
    add('$A_{\\min}=%.5f$, step~2(i) $(a)/c\\ge%.2f$, step~3(i) margin\n$%.4f$ (interval arithmetic), step~3(ii) ratio $%.1f$' % (
        dn(A2['A_min'], 5), dn(A2['step2i'], 2), dn(A2['step3a_margin'], 4), dn(A2['step3ii'], 1)), A2['A_min'], 'br_windows:caseA2 (N = 16)')
    add('$A_{\\min}=%.5f$, step~2(i) $%.2f$, step~3(i) $%.4f$, step~3(ii) $%.2f$' % (dn(L2['A_min'], 5), dn(L2['step2i'], 2), dn(L2['step3a_margin'], 4), dn(L2['step3ii'], 2)),
        L2['A_min'], 'br_windows:bj1 (N = 20)')
    # Lemma F7, Theorems F8, F9
    c16 = B[K[16]]['conclusion_x_to_0']; c20 = B['N=20']['conclusion_x_ge_1e-12']
    xf = max(B[K[N]]['conclusion_x_ge_1e-12']['sup_x_fstar'] for N in K)
    add('$\\sup_{0<x\\le1}x f^*(x)\\le%.4f$' % up(xf, 4), xf, 'br_windows:sup x f*')
    W = R('bp_windowsN.json')
    add('is $%.3f$ for $N=6$ and $%.2f$ for $N=20$' % (up(W['N=6']['conclusion']['sup_x(1+Z2)'], 3), up(W['N=20']['conclusion']['sup_x(1+Z2)'], 2)),
        W['N=20']['conclusion']['sup_x(1+Z2)'], 'bp_windowsN:sup x(1 + Z2)')
    add('\\tfrac14\\big)\\le%.3f' % up(c16['k_eff2_max'], 3), c16['k_eff2_max'], 'br_windows:k_eff^2 (N = 16)')
    add('$g\\ge0.9\\times%.5f/(2\\sqrt{%.3f})\\ge%.4f$' % (dn(A2['A_min'], 5), up(c16['k_eff2_max'], 3), dn(c16['g_lower'], 4)), c16['g_lower'], 'br_windows:g (N = 16)')
    add('or $g\\ge%.4f$' % dn(c16['g_lower'], 4), c16['g_lower'], 'br_windows:g (N = 16, end)')
    add('$\\keff^2\\le%.3f$ (Lemma~\\ref{lem:F7}) and $g\\ge0.9\\times%.5f/(2\\sqrt{%.3f})\\ge%.4f$' % (up(c20['k_eff2_max'], 3), dn(L2['A_min'], 5), up(c20['k_eff2_max'], 3),
        dn(c20['g_lower'], 4)), c20['g_lower'], 'br_windows:g (N = 20)')
    # Table F2
    at = B['N=20']['small_x_attempt']; assert at['caseB']['all_ok'] and at['caseC']['ok'] and isinstance(at['caseA2'], str)
    for N, cp in ((6, '$2\\times10^{-3}$'), (8, '$2\\times10^{-3}$'), (12, '$2\\times10^{-4}$'), (16, '$2\\times10^{-5}$')):
        r = B[K[N]]; er = r['eps_R']['eps_R']
        add('%d & %s & %.3f & %.3f & %.2f & %.2f & %s\\\\' % (N, cp, dn(bk[N]['worst_margin'], 3), dn(r['caseB']['worst_margin'], 3), dn(r['caseC']['ratio_lower'], 2),
            dn(r['caseA1']['T_lower'], 2), '%.4f' % up(er, 4) if er > 1e-3 else '$%s$' % sci(er, 1, up)), r['caseC']['ratio_lower'], 'Table F2 row N = %d' % N)
    add('20 & $10^{-7}$ & %.3f & %.3f & %.2f & %.2f & %.2f (no)\\\\' % (dn(bk[20]['worst_margin'], 3), dn(at['caseB']['worst_margin'], 3), dn(at['caseC']['ratio_lower'], 2),
        dn(T20['T_lower'], 2), up(at['eps_R']['eps_R'], 2)), at['eps_R']['eps_R'], 'Table F2 row N = 20')
    add('24 & -- & %.3f & --' % dn(bk[24]['worst_margin'], 3), bk[24]['worst_margin'], 'Table F2 row N = 24')
    n24 = B['N=24']['bj1_x_ge_1e-12']; assert not n24['ok']
    add('gives $(a)/c\\ge%.2f$ only' % dn(n24['step2i'], 2), n24['step2i'], 'br_windows:N = 24 step 2(i)')
    add('$T_{20}\\ge%.2f$ only, so $\\beta/\\tau\\le%.3f$' % (dn(T20['T_lower'], 2), up(at['eps_R']['beta_over_tau'], 3)), T20['T_lower'], 'br_windows:N = 20 attempt')
    add('(ratio $%.3f$)' % dn(bk[24]['worst_margin'], 3), bk[24]['worst_margin'], 'br_windows:N = 24 cells')
    # Theorem 11 beyond k0 = 240 (App. B)
    U = R('bs_upper_tail.json'); assert U['ok']; C2 = U['conclusion']; M = U['members']; S = R('bs_upper_tail_stress.json'); assert S['ok']
    add('R_U\\in[%.5f,\\ %.5f]' % (dn(C2['R_U_interval'][0], 5), up(C2['R_U_interval'][1], 5)), C2['R_U_interval'], 'bs_upper_tail:R_U')
    add('$R_U\\le%.4f$ for all $k_0\\ge240$' % up(C2['R_U_interval'][1], 4), C2['R_U_interval'][1], 'bs_upper_tail:R_U (Sec. 5)')
    add('its value at $k_0=240$, $%.3f>0$' % dn(C2['lhs_at_K0'], 3), C2['lhs_at_K0'], 'bs_upper_tail:lhs at K0')
    add('$\\Qb\\le10^{%d}$, $\\varrho\\le10^{%d}$ and $x\\ge%d$' % (math.ceil(M['log10_Qbar_sup']), math.ceil(M['log10_varrho_sup']), int(M['x_min'])), M['x_min'], 'bs_upper_tail:members')
    add('$8.137/k_0^2$ by a factor smaller than $\\exp(-10^{%d})$' % math.floor(U['bl7_tail']['G_upper_log10_abs']), U['bl7_tail']['G_upper_log10_abs'], 'bs_upper_tail:BL7 tail')
    add('$%d$ comparisons' % S['checks'], S['checks'], 'bs_upper_tail_stress:checks')
    add('at $k_0=%s$ and\n$1000$' % ('$, $'.join(c['k'] for c in U['consistency'][:-1])), len(U['consistency']), 'bs_upper_tail:consistency points')
    return n


if __name__ == '__main__':
    import glob
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d missing' % (len(N), len(miss)))
    for x in miss: print('  MISSING:', repr(x['text']), '<-', x['source'])
