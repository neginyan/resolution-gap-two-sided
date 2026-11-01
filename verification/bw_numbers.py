"""v1.5: every number of the manuscript re-derived from results/.  The v1.4 list (bu_numbers.py), unchanged, plus the numbers of App. F for
K >= K0 (Lemmas F9, F10, Theorem F11 and the remark on K0 = 0.002; bw_k0_0.003.json, bw_k0_0.002.json).  Bounds are rounded outward
(ceil for upper bounds, floor for lower bounds).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
from bm_numbers import up, dn, sci
from bu_numbers import numbers as bu_numbers

R = lambda f: json.load(open('results/' + f))


def numbers():
    n = list(bu_numbers())
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    D = R('bw_k0_0.003.json'); s, C, A1, E, F6, f7, cc = D['sups'], D['caseC'], D['caseA1'], D['eps_R'], D['F6'], D['fstar'], D['conclusion']
    bk, CB = D['bk1'], D['caseB']
    assert C['ok'] and E['ok'] and F6['ok'] and cc['ok'] and bk['all_ok'] and CB['all_ok'] and bk['audit_violations'] == 0 and CB['audit_violations'] == 0
    assert F6['budget_audit_violations'] == 0 and f7['audit_violations'] == 0
    add('$N_1\\le%.3f$, $N_2\\le%.1f$, $S_2\\le%.3f$ (for at most $20$' % (up(s['N1'], 3), up(s['N2'], 1), up(s['S2'], 3)), s['N1'], 'bw_k0:sums (Lemma F9)')
    add("then $(a')\\ge%d$." % int(dn(C['a_prime_lower'], 0)), C['a_prime_lower'], 'bw_k0:caseC (statement)')
    add('with $T\\ge%.3f$.' % dn(A1['T_lower'], 3), A1['T_lower'], 'bw_k0:T (statement)')
    add('$A_1A_2A_3\\ge A_{\\min}=%.4f$.' % dn(F6['A_min'], 4), F6['A_min'], 'bw_k0:A_min (statement)')
    add('$\\sup_{x>0}xf^*(\\max(x,K_0))\\le%.4f$.' % up(f7['sup_x_fstar'], 4), f7['sup_x_fstar'], 'bw_k0:f* (statement)')
    add('$%d$ cells for (i), smallest ratio $%.3f$, and $%d$ cells for (ii), smallest ratio $%.3f$' % (bk['cells'], dn(bk['worst_margin'], 3), CB['cells'],
        dn(CB['worst_margin'], 3)), bk['worst_margin'], 'bw_k0:cells')
    add('the audit at $%d$ and $%d$ interior points' % (bk['audit_points'], CB['audit_points']), CB['audit_points'], 'bw_k0:cell audit')
    add('$S_2\\le%.3f$, $N_1\\le%.3f$, $N_2\\le%.1f$ (interval arithmetic): $(W/b)_{\\max}\\le%.2f$, $V_{\\max}\\le%.2f$, $d\\ge%.4f$,' % (
        up(s['S2'], 3), up(s['N1'], 3), up(s['N2'], 1), up(C['W_over_b_max'], 2), up(C['V_max'], 2), dn(C['X_min_lo'], 4)), C['V_max'], 'bw_k0:caseC (proof)')
    add("$(a')\\ge%d>%s\\,c_*$" % (int(dn(C['a_prime_lower'], 0)), sci(C['ratio_lower'], 1, dn)), C['ratio_lower'], 'bw_k0:caseC ratio')
    add('$W_c\\le%.4f$, $T\\ge%.3f$ (interval arithmetic)' % (up(A1['W_c'], 4), dn(A1['T_lower'], 3)), A1['W_c'], 'bw_k0:caseA1')
    add('$N_{\\mathrm{eff}}=%d$' % E['N_eff'], E['N_eff'], 'bw_k0:N_eff')
    add('$A_0\\le%s$, $B_0\\le%s$, $C_0\\le%s$, so $\\varepsilon_R\\le%.6f$' % (sci(E['A0'][1], 3, up), sci(E['B0'][1], 2, up), sci(E['C0'][1], 2, up), up(E['eps_R'], 6)),
        E['eps_R'], 'bw_k0:eps_R')
    add('$\\Gamma_*\\le%.3f$' % up(F6['Gamma_star_budget'], 3), F6['Gamma_star_budget'], 'bw_k0:Gamma* budget')
    add('$\\sup\\Gamma_1m$ with $\\eta_p$ is $%.2f$), so $e\'=%.5f$' % (F6['Gamma_star_eta'], dn(F6['e_prime'], 5)), F6['e_prime'], 'bw_k0:e prime')
    add('$\\gamma_*\\le%s$ and $(a)/c\\ge%.1f$' % (sci(F6['gamma_star_budget'], 3, up), dn(F6['step2i'], 1)), F6['step2i'], 'bw_k0:step 2(i)')
    add('by a factor $%.2f$' % dn(F6['step3b'], 2), F6['step3b'], 'bw_k0:step 3(ii)')
    add('margin $%.4f$ (interval arithmetic). On $%d$ $x$-cells' % (dn(F6['step3a_interval_margin'], 4), F6['budget_x_rows']), F6['step3a_interval_margin'], 'bw_k0:step 3(i)')
    add('an audit at $%d$ random interior points' % F6['budget_audit_points'], F6['budget_audit_points'], 'bw_k0:budget audit')
    add('$A_{\\min}=%.5f$.' % dn(F6['A_min'], 5), F6['A_min'], 'bw_k0:A_min (proof)')
    add('give $\\sup_xxf^*\\le%.4f$' % up(f7['sup_x_fstar'], 4), f7['sup_x_fstar'], 'bw_k0:f* (proof)')
    add('$\\varepsilon=10^{-6}+\\varepsilon_R+\\varepsilon_\\Phi\\le%.6f$' % up(cc['eps'], 6), cc['eps'], 'bw_k0:eps')
    add('\\ \\le\\ %.3f' % up(cc['k_eff2_max'], 3), cc['k_eff2_max'], 'bw_k0:k_eff^2')
    add('$g\\ge0.9\\times%.5f/(2\\sqrt{%.3f})\\ge%.4f$.' % (dn(F6['A_min'], 5), up(cc['k_eff2_max'], 3), dn(cc['g_lower'], 4)), cc['g_lower'], 'bw_k0:g')
    add('$\\infty^\\ddagger$ & $10^{-9}$ & %.3f & %.3f & %d & %.2f & %.4f\\\\' % (dn(bk['worst_margin'], 3), dn(CB['worst_margin'], 3), int(dn(C['ratio_lower'], 0)),
        dn(A1['T_lower'], 2), up(E['eps_R'], 4)), E['eps_R'], 'Table F2 row v1.5')
    D2 = R('bw_k0_0.002.json'); c2 = D2['conclusion']
    assert c2['ok'] and D2['bk1']['all_ok'] and D2['caseB']['all_ok'] and D2['bk1']['audit_violations'] == 0 and D2['caseB']['audit_violations'] == 0
    add('$\\varepsilon_R\\le%.4f$, $A_{\\min}=%.4f$, $g\\ge%.4f$' % (up(D2['eps_R']['eps_R'], 4), dn(D2['F6']['A_min'], 4), dn(c2['g_lower'], 4)), c2['g_lower'], 'bw_k0:K0 = 0.002')
    return n


if __name__ == '__main__':
    import glob
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d missing' % (len(N), len(miss)))
    for x in miss: print('  MISSING:', repr(x['text']), '<-', x['source'])
