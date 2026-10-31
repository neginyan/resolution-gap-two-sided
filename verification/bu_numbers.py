"""v1.4: every number of the manuscript re-derived from results/.  The v1.3 list (bs_numbers.py) without the 34 numbers of App. F that v1.4 restates
for profiles with any phases (they are listed in SUPERSEDED by their source tag), plus the numbers of App. F (v1.4, any phases; bu_windows.json),
App. G (paper E's remainder along the comb; bt_eps.json), the theta_tau corrections with phases (bl_theta, phase=True) and the main text.
Bounds are rounded outward (ceil for upper bounds, floor for lower bounds).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, math
from bm_numbers import up, dn, sci
from bs_numbers import numbers as bs_numbers

R = lambda f: json.load(open('results/' + f))
SUPERSEDED = {'bl_theta:E_R', 'bl_theta:e3', 'br_windows:bk1 cells', 'br_windows:bk1 worst margins', 'br_windows:caseB b-cells (N = 16)', 'br_windows:caseB (N = 16)',
              'br_windows:audit points', 'br_windows:caseC (N = 16, statement)', 'br_windows:caseC W/b', 'br_windows:caseC V_max', 'br_windows:caseC X_min',
              'br_windows:caseC main', 'br_windows:caseC alpha E', "br_windows:caseC (a') lower", 'br_windows:caseA1 T', 'br_windows:caseA1 (N = 16)',
              'br_windows:T_split', 'br_windows:caseA2 A_min (statement)', 'br_windows:bj1 N = 20 (statement)', 'br_windows:eps_R beta/tau', 'br_windows:eps_R slope',
              'br_windows:eps_R A0, B0, C0', 'br_windows:eps_R', 'br_windows:Phi remainder', 'br_windows:E_R sup (N = 20)', 'br_windows:eps (N = 20)',
              'br_windows:caseA2 (N = 16)', 'br_windows:bj1 (N = 20)', 'br_windows:k_eff^2 (N = 16)', 'br_windows:g (N = 16)', 'br_windows:g (N = 16, end)',
              'br_windows:g (N = 20)', 'br_windows:N = 20 attempt', 'br_windows:N = 24 cells'}


def lo(v): return v[0] if isinstance(v, (list, tuple)) else v
def hi(v): return v[1] if isinstance(v, (list, tuple)) else v


def numbers():
    n = [x for x in bs_numbers() if x['source'] not in SUPERSEDED]
    assert len(SUPERSEDED) == 34
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    import numpy as np
    np.seterr(all='ignore')
    import bl_theta as T
    t = T.eps_table(80.0, phase=True)
    add('For $\\tau\\ge80$, $E_R\\le%s$' % sci(t['E_R'], 1, up), t['E_R'], 'bl_theta:E_R (phase)')
    add('with $\\epsilon_3\\le%s$.' % sci(t['e3_AX3'], 1, up), t['e3_AX3'], 'bl_theta:e3 (phase)')
    add('$\\epsilon_3\\le%s$, the factor' % sci(t['e3_AX3'], 1, up), t['e3_AX3'], 'bl_theta:e3 (phase, sizes)')
    assert t['eps_theta'] <= 7.1e-14
    # App. F v1.4
    B = R('bu_windows.json')["N=20, c'=0.000000001, phase"]
    bk, CB, C, A1, E, A2, cc = B['bk1_x_ge_1e-12'], B['caseB'], B['caseC'], B['caseA1'], B['eps_R'], B['caseA2'], B['conclusion_x_to_0']
    assert bk['all_ok'] and bk['audit_violations'] == 0 and CB['all_ok'] and CB['audit_violations'] == 0 and C['ok'] and A1['ok'] and E['ok'] and A2['ok'] and cc['ok']
    add('$14400$ cells; smallest ratio $%.3f$' % dn(bk['worst_margin'], 3), bk['worst_margin'], 'bu_windows:bk1')
    add('$%d$ geometric $b$-cells on $[10^{-33},1]$' % CB['b_cells'], CB['b_cells'], 'bu_windows:caseB b-cells')
    add('$%d$ cells; smallest ratio $%.3f$' % (CB['cells'], dn(CB['worst_margin'], 3)), CB['worst_margin'], 'bu_windows:caseB')
    add('($%d$ points for (i) and $%d$ for (ii))' % (bk['audit_points'], CB['audit_points']), CB['audit_points'], 'bu_windows:audit')
    add('$17614$ interior points for Lemma~\\ref{lem:F2}(i) and $%d$ for (ii)' % CB['audit_points'], CB['audit_points'], 'bu_windows:audit (App. B)')
    add("$(a')\\ge%.3f>%.1f\\,c_*$" % (dn(C['a_prime_lower'], 3), dn(C['ratio_lower'], 1)), C['a_prime_lower'], 'bu_windows:caseC (statement)')
    add('$(W/b)_{\\max}\\le%.2f$ for $N=20$' % up(hi(C['W_over_b_max']), 2), C['W_over_b_max'], 'bu_windows:caseC W/b')
    add('(1-\\Qb g_1)\\le%.4f$. Hence $V<V_{\\max}=(1+(W/b)_{\\max})c\\theta\\kappa_w(1+\\sigma^3)\\le%.2f$' % (up(hi(C['kappa_w']), 4), up(hi(C['V_max']), 2)), C['V_max'], 'bu_windows:caseC V_max')
    add('$d\\ge%.6f$, the first term is $\\ge%s$, $\\alpha E\\le%s$' % (dn(lo(C['X_min']), 6), sci(lo(C['main']), 3, dn), sci(hi(C['err']), 1, up)), C['main'], 'bu_windows:caseC main')
    add("c'\\big)\\ge%.3f$." % dn(C['a_prime_lower'], 3), C['a_prime_lower'], "bu_windows:caseC (a') lower")
    add('$T_{20}\\ge%.3f$ (any phases' % dn(A1['T_lower'], 3), A1['T_lower'], 'bu_windows:caseA1 T (statement)')
    add('$W_c\\le%.5f$, $T_{20}\\ge%.4f$' % (up(hi(A1['W_coef']), 5), dn(A1['T_lower'], 4)), A1['W_coef'], 'bu_windows:caseA1')
    add('($%d$ for $N=20$;' % int(A1['T_split']), A1['T_split'], 'bu_windows:T_split')
    add('$A_1A_2A_3\\ge A_{\\min}=%.4f$' % dn(A2['A_min'], 4), A2['A_min'], 'bu_windows:caseA2 (statement)')
    add('($r=\\beta/\\tau\\le%.4f$)' % up(E['beta_over_tau'], 4), E['beta_over_tau'], 'bu_windows:eps_R r')
    add('($3/80-(1-r)^2/2\\le%.2f$)' % up(E['slope'], 2), E['slope'], 'bu_windows:eps_R slope')
    add('$A_0\\le%.5f$, $B_0\\le%s$, $C_0\\le%s$' % (up(hi(E['A0']), 5), sci(hi(E['B0']), 2, up), sci(hi(E['C0']), 2, up)), E['A0'], 'bu_windows:eps_R A0, B0, C0')
    add('$\\varepsilon_R\\le%.5f$' % up(E['eps_R'], 5), E['eps_R'], 'bu_windows:eps_R')
    add('$2\\Qb(\\nu_0+\\tau\\nu_2)\\le%s$' % sci(E['phi_remainder'], 1, up), E['phi_remainder'], 'bu_windows:Phi remainder')
    add('$A_{\\min}=%.5f$, step~2(i) $(a)/c\\ge%.3f$, step~3(i) margin $%.4f$ (interval arithmetic), step~3(ii) ratio $%.2f$' % (
        dn(A2['A_min'], 5), dn(A2['step2i'], 3), dn(A2['step3a_margin'], 4), dn(A2['step3ii'], 2)), A2['A_min'], 'bu_windows:caseA2')
    add('\\tfrac14\\big)\\le%.3f' % up(cc['k_eff2_max'], 3), cc['k_eff2_max'], 'bu_windows:k_eff^2')
    add('$g\\ge0.9\\times%.5f/(2\\sqrt{%.3f})\\ge%.4f$' % (dn(A2['A_min'], 5), up(cc['k_eff2_max'], 3), dn(cc['g_lower'], 4)), cc['g_lower'], 'bu_windows:g')
    add('or $g\\ge%.4f$' % dn(cc['g_lower'], 4), cc['g_lower'], 'bu_windows:g (end)')
    add('20$^\\dagger$ & $10^{-9}$ & %.3f & %.3f & %.2f & %.2f & %.4f\\\\' % (dn(bk['worst_margin'], 3), dn(CB['worst_margin'], 3), dn(C['ratio_lower'], 2),
        dn(A1['T_lower'], 2), up(E['eps_R'], 4)), E['eps_R'], 'Table F2 row v1.4')
    O = R('br_windows.json')
    add('(ratio $%.3f$, even profiles)' % dn(O['N=24']['bk1_x_ge_1e-12']['worst_margin'], 3), O['N=24']['bk1_x_ge_1e-12']['worst_margin'], 'br_windows:N = 24 cells')
    add('gives $(a)/c\\ge%.2f$ only' % dn(O['N=24']['bj1_x_ge_1e-12']['step2i'], 2), O['N=24']['bj1_x_ge_1e-12']['step2i'], 'br_windows:N = 24 step 2(i) (v1.4 text)')
    O16 = O["N=16, c'=0.00002"]
    add("$(a')\\ge%.4f>%.1fc_*$ at $N=16$" % (dn(O16['caseC']['a_prime_lower'], 4), dn(O16['caseC']['ratio_lower'], 1)), O16['caseC']['a_prime_lower'], 'br_windows:caseC N = 16 (v1.3, quoted)')
    add('$A_{\\min}=%.5f$ for $N\\le16$ down to $x\\to0$, and $%.5f$ for $N\\le20$' % (dn(O16['caseA2']['A_min'], 5), dn(O['N=20']['bj1_x_ge_1e-12']['A_min'], 5)),
        O16['caseA2']['A_min'], 'br_windows:A_min (v1.3, quoted)')
    # App. G
    G = R('bt_eps.json'); assert G['ok']; f = G['form']
    add('276.4\\,k_0^{-2}E^{3/2}\\,(1+%.3f)' % up(f['R'][1], 3), f['R'][1], 'bt_eps:statement')
    add('$|\\varepsilon(S\'_C)|\\le%s$ on the whole range' % sci(G['sup_k_ge_3.43'], 2, up), G['sup_k_ge_3.43'], 'bt_eps:sup')
    add('\\le%.2f\\,k_0^{-2}E^{3/2}(1+R),' % up(f['c'][1], 2), f['c'][1], 'bt_eps:constant')
    add('R&\\in[%s,\\,%.4f]' % (sci(f['R'][0], 1, lambda v, d: -up(-v, d)), up(f['R'][1], 4)), f['R'], 'bt_eps:R')
    add('attained at $k_0=3.43$: $%s$' % sci(G['sup_k_ge_3.43'], 2, up), G['sup_k_ge_3.43'], 'bt_eps:sup (proof)')
    sd = G['side']
    add('$w_0\\le%s\\cdot3L$, $r_0^2\\le%s\\,c_2W_2/(4V)$, $M\\le%s$' % (sci(sd['w0_le_3L_sup'], 1, up), sci(sd['r0sq_over_main_sup'], 1, up), sci(sd['M_sup'], 1, up)),
        sd['M_sup'], 'bt_eps:side conditions')
    s562 = sci(G['sup_k_ge_3.43'], 2, up)
    for where in ('(P6) row', 'Theorem 12', 'Corollary 13', 'abstract', 'introduction'):
        n.append({'text': s562.replace('\\times', '\\times'), 'value': G['sup_k_ge_3.43'], 'source': 'bt_eps:sup (%s)' % where})
    return n


if __name__ == '__main__':
    import glob
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d missing' % (len(N), len(miss)))
    for x in miss: print('  MISSING:', repr(x['text']), '<-', x['source'])
