"""v1.1: every number of the manuscript re-derived from results/.  The v1.0 list (bh_numbers.py) without the one value that v1.1 restates with
perturbed inputs, plus the numbers of the theta_tau appendix, the perturbed chain (bl_chain), Lemma BK1 (tau0 = 80), Lemma BJ1 and Lemma BL7.
Bounds are rounded outward (ceil for upper bounds, floor for lower bounds).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob, math
import numpy as np
from bh_numbers import numbers as bh_numbers

R = lambda f: json.load(open('results/' + f))
up = lambda v, d: math.ceil(v * 10 ** d) / 10 ** d
dn = lambda v, d: math.floor(v * 10 ** d) / 10 ** d


def sci(v, d, f):                       # mantissa x 10^e rounded with f (up or dn) to d decimals
    e = int(math.floor(math.log10(abs(v)))); m = f(v / 10 ** e, d)
    if m >= 10: m /= 10; e += 1
    return ('%%.%df\\times10^{%%d}' % d) % (m, e)


def chain_cells():
    C = {}
    for f in sorted(glob.glob('results/bl_chain_cells_part*.jsonl')):
        for line in open(f):
            if line.strip(): d = json.loads(line); C[d['i']] = d
    return C


def numbers():
    n = [x for x in bh_numbers() if x['source'] != 'bh_cert_cells:min margin at c1 = 1.7']
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    C = chain_cells(); OLD = {}
    for f in sorted(glob.glob('results/bh_cert_cells_part*.json')): OLD.update({int(k): v for k, v in json.load(open(f)).items()})
    m = min(v['margin_c1_1.7'] for v in C.values())
    add('$%.1f\\times10^{-5}$ (last cell; all' % dn(m * 1e5, 1), m, 'bl_chain:min margin at c1 = 1.7 (floor, 1 digit)')
    add('is $%.2f\\times10^{-5}$ (last cell)' % dn(m * 1e5, 2), m, 'bl_chain:min margin at c1 = 1.7 (floor, 2 digits)')
    add('at most $%.4f$' % up(max(v['c1_needed'] for v in C.values()), 4), max(v['c1_needed'] for v in C.values()), 'bl_chain:max c1_needed')
    dL = max(OLD[i]['L_cell'] - C[i]['L_cell'] for i in C); add('at most $%s$' % sci(dL, 1, up), dL, 'bl_chain:max change of L_cell')
    cb = min(v['caseB_margin'] for v in C.values()); add('$>%d\\times10^{%d}$ times' % (int(cb / 10 ** int(math.log10(cb))), int(math.log10(cb))), cb, 'bl_chain:min caseB margin')
    add('\\varphi_1\\le%.2f$ on every cell' % up(max(v['absorb_ratio'] for v in C.values()), 2), max(v['absorb_ratio'] for v in C.values()), 'bl_chain:max eta G theta/phi1')
    assert min(v['Lgt1_margin'] for v in C.values()) > 2700 and max(v['beta'] for v in C.values()) <= 0.251 and min(v['omega'] for v in C.values()) >= 0.99985 \
        and min(v['d_sign'] for v in C.values()) >= 8.9, 'the v1.0 margins quoted in App. B must also hold for the perturbed chain'
    TL = R('bl_chain_tail.json'); add('increasing from $%.6f$' % dn(TL['T_at_l_end'], 6), TL['T_at_l_end'], 'bl_chain_tail:T_at_l_end')
    N8 = R('bl_narrow80.json'); add('two sides is $%.2f$' % dn(N8['worst_margin'], 2), N8['worst_margin'], 'bl_narrow80:worst_margin')
    add('covered by $%d$ cells' % N8['cells'], N8['cells'], 'bl_narrow80:cells')
    TH = R('bl_theta.json'); t = TH['eps_table_tau80']
    add('$\\nu_0=%.2f\\times10^{-18}$' % (t['nu']['0'] * 1e18), t['nu']['0'], 'bl_theta:nu0 at 80')
    add('$E_R\\le%s$' % sci(t['E_R'], 1, up), t['E_R'], 'bl_theta:E_R')
    add('$\\epsilon_1\\le%s$' % sci(t['e1_Jww'], 1, up), t['e1_Jww'], 'bl_theta:e1')
    add('$\\le%s$ and its $b$-part' % sci(t['e2_W_rel'], 1, up), t['e2_W_rel'], 'bl_theta:e2 relative')
    add('$\\le%sb$' % sci(t['e2_W_add_coef_of_b'], 1, up), t['e2_W_add_coef_of_b'], 'bl_theta:e2 b-part')
    add('$\\epsilon_3\\le%s$' % sci(t['e3_AX3'], 1, up), t['e3_AX3'], 'bl_theta:e3')
    add('Lemma~\\ref{lem:BL5} $\\le%s$' % sci(t['e4_keff'], 1, up), t['e4_keff'], 'bl_theta:e4')
    add('$\\epsilon_5\\le%s$' % sci(t['e5_Lemma5'], 1, up), t['e5_Lemma5'], 'bl_theta:e5')
    add('$\\epsilon_\\theta:=\\max\\le%s$' % sci(t['eps_theta'], 1, up), t['eps_theta'], 'bl_theta:eps_theta')
    add('On $%d$ states' % len(TH['checks']), len(TH['checks']), 'bl_theta:number of checked states')
    add('to $%s$ for $\\tau$ from' % sci(max(c['identity_rel_err'] for c in TH['checks']), 0, up).replace('1\\times', ''), max(c['identity_rel_err'] for c in TH['checks']), 'bl_theta:identity error')
    B = R('bl_bj1.json')['1e-24']; s2a = B['step2a']
    add('A_{\\min}=%.3f$' % dn(B['A_min'], 3), B['A_min'], 'bl_bj1:A_min')
    add('=%.4f$; Lemma' % dn(B['step2b']['e_prime'], 4), B['step2b']['e_prime'], 'bl_bj1:e_prime')
    add('\\ge%d\\times10^%dc$' % (int(B['step2b']['a_lower_over_c'] / 10 ** int(math.log10(B['step2b']['a_lower_over_c']))), int(math.log10(B['step2b']['a_lower_over_c']))),
        B['step2b']['a_lower_over_c'], 'bl_bj1:(a)/c in 2(i)')
    add('=%.2f\\sigma$' % up(s2a['V_bar'] / 1e-24, 2), s2a['V_bar'], 'bl_bj1:V_bar/sigma')
    add('<%.4f$, hence' % up(s2a['one_minus_C_bar'], 4), s2a['one_minus_C_bar'], 'bl_bj1:1-C bound')
    add('$L^2<%.4f$' % up(s2a['L2_bar'], 4), s2a['L2_bar'], 'bl_bj1:L2 bound')
    add('$\\delta_H\\le%s$' % sci(s2a['delta_H'], 1, up), s2a['delta_H'], 'bl_bj1:delta_H')
    add('$r_C\\le%.4f$' % up(s2a['r_C'], 4), s2a['r_C'], 'bl_bj1:r_C'); add('$r_w\\le%.4f$' % up(s2a['r_w'], 4), s2a['r_w'], 'bl_bj1:r_w')
    add('$g(L_1)\\ge%.6f\\sigma>%.6f\\sigma' % (dn(B['step3a']['g_L1_over_sigma'], 6), up(B['step3a']['rhs_over_sigma'], 6)), B['step3a']['margin'], 'bl_bj1:3(i)')
    add('(margin $%.4f$; the step' % dn(B['step3a']['margin'], 4), B['step3a']['margin'], 'bl_bj1:3(i) margin')
    add('\\gamma_{\\mathrm{lo}}=%.3f$' % dn(B['step3b']['gamma_lo'], 3), B['step3b']['gamma_lo'], 'bl_bj1:gamma_lo')
    add('$x\\le%.2f$' % up(B['step3b']['admissible_x'][1], 2), B['step3b']['admissible_x'][1], 'bl_bj1:admissible x')
    add('by a factor $%s$: a contradiction' % sci(B['step3b']['W_lower_over_W_cap'], 1, dn), B['step3b']['W_lower_over_W_cap'], 'bl_bj1:3(ii) ratio')
    add('$\\Gamma_*=%.3f$' % up(B['Gamma_star'], 3), B['Gamma_star'], 'bl_bj1:Gamma*'); add('$W_{\\max}=%.4f$' % up(B['W_max'], 4), B['W_max'], 'bl_bj1:W_max')
    add('$S_\\xi=%.3f$' % up(B['S_xi'], 3), B['S_xi'], 'bl_bj1:S_xi')
    M7 = R('bm_bl7.json'); c = M7['comb']
    add('$\\tau\\ge%.2f\\times10^{22}$' % dn(c['tau_min'] / 1e22, 2), c['tau_min'], 'bm_bl7:tau_min')
    add('\\le%.4f$' % up(c['sigma_tau_max'], 4), c['sigma_tau_max'], 'bm_bl7:sigma tau')
    add('10^{-%.2f\\times10^{22}}' % dn(-c['log10_correction_bound'] / 1e22, 2), c['log10_correction_bound'], 'bm_bl7:log10 correction')
    add('On the $%d$ states of the numerical check' % len(M7['checks']), len(M7['checks']), 'bm_bl7:checks')
    return n


if __name__ == '__main__':
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d found in paper/sections' % (len(N), len(N) - len(miss)))
    for x in miss: print('NOT FOUND:', x['text'], x['source'])
