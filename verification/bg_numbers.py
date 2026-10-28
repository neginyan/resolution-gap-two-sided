"""Stage BG: every number of the paper-F draft re-derived from the stored results (paper E's form): each entry is (string as printed in the
manuscript, value recomputed from a results file, source).  run_all_BG checks that each string appears in paperF/paper and equals the
formatted value.  -> results/bg_numbers.json, paperF/paperF_numbers.md.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np

R = lambda f: json.load(open('results/' + f))


def numbers():
    C = R('bf_cert_p.json'); UP = R('bf_upper_full.json'); CU = R('bf_cert_upper.json'); BA = R('ba_cert.json'); AB = R('bf_ay_bootstrap.json')
    LO = R('be_lo.json')['table']; LX = R('bf_lo_cross.json'); LA = R('be_lambda.json'); MB = R('bg_members.json'); AY = R('ay_certificate.json')
    MIX = {s: R('bd_opt_comb_%s.json' % s)['g_min_full'] for s in ('0.003', '0.001', '0.0003', '0.0001', '1e-06', '1e-12', '1e-24')}
    n = []
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    for e, f in ((24, '%.4f'), (48, '%.4f'), (96, '%.4f'), (256, '%.4f'), (1024, '%.4f'), (16384, '%.4f')):
        v = C['eps0.01_1e-%d' % e]['g_sqrt_ln_over_1_minus_eps']; add(f % v, v, 'bf_cert_p.json:eps0.01_1e-%d.g_sqrt_ln_over_1_minus_eps' % e)
        x = C['eps0.01_1e-%d' % e]['X_minus_quarter_ln']; add('+%.2f' % x, x, 'bf_cert_p.json:eps0.01_1e-%d.X_minus_quarter_ln' % e)
    add('%.5f' % C['eps0.01_1e-24']['F_int_final'], C['eps0.01_1e-24']['F_int_final'], 'bf_cert_p.json:eps0.01_1e-24.F_int_final')
    add('%.3f' % C['eps0.01_1e-24']['steps'][0]['F_next'], C['eps0.01_1e-24']['steps'][0]['F_next'], 'bf_cert_p.json:eps0.01_1e-24.steps[0].F_next')
    add('%.2f' % C['S1_P'], C['S1_P'], 'bf_cert_p.json:S1_P')
    add('%.4f' % C['eps0.01_1e-24']['phi1'], C['eps0.01_1e-24']['phi1'], 'bf_cert_p.json:eps0.01_1e-24.phi1')
    add('%.2f' % C['eps0.01_1e-24']['W_p_forced_if_L_gt_1'], C['eps0.01_1e-24']['W_p_forced_if_L_gt_1'], 'bf_cert_p.json:W_p_forced_if_L_gt_1')
    add('%.1f' % C['eps0.01_1e-24']['x_lo'], C['eps0.01_1e-24']['x_lo'], 'bf_cert_p.json:eps0.01_1e-24.x_lo')
    add('F_0=%.1f' % AB['1e-24']['smallest_F0'], AB['1e-24']['smallest_F0'], 'bf_ay_bootstrap.json:1e-24.smallest_F0')
    add('F_0=%.1f' % AB['1e-96']['smallest_F0'], AB['1e-96']['smallest_F0'], 'bf_ay_bootstrap.json:1e-96.smallest_F0')
    for e, f in ((24, '%.3f'), (96, '%.3f'), (1024, '%.3f')):
        v = AY['eps0.01_1e-%d' % e]['g_bound_sqrt_ln'] / 0.99; add(f % v, v, 'ay_certificate.json:eps0.01_1e-%d.g_bound_sqrt_ln / 0.99' % e)
    add('$%d$ cells' % CU['cells'], CU['cells'], 'bf_cert_upper.json:cells')
    add('$%.2f$ to $%.0f$' % (CU['t_range_checked'][0], CU['t_range_checked'][1]), CU['t_range_checked'], 'bf_cert_upper.json:t_range_checked')
    add('$-%.2f\\times10^{-4}$' % (-CU['worst_upper_bound_of_h'] * 1e4), CU['worst_upper_bound_of_h'], 'bf_cert_upper.json:worst_upper_bound_of_h')
    add('$%d$ cells, $-%.2f\\times10^{-4}$' % (BA['cells'], -BA['worst_upper_bound_of_h'] * 1e4), BA['worst_upper_bound_of_h'], 'ba_cert.json')
    for k, v in UP['true'].items():
        add('%.4f & %.4f' % (v['g_true'], v['U_main']), [v['g_true'], v['U_main']], 'bf_upper.json:true.%s' % k)
    add('%.4f' % UP['true']['1e-24']['U_main'], UP['true']['1e-24']['U_main'], 'bf_upper.json:true.1e-24.U_main')
    add('%.2g' % (MB['Qbar_max']), MB['Qbar_max'], 'bg_members.json:Qbar_max')
    add('x\\ge%.2f' % MB['x_min'], MB['x_min'], 'bg_members.json:x_min')
    for k, f in (('1e-6', '%.4f & %.4f & %.3f'), ('1e-12', '%.4f & %.4f & %.3f'), ('1e-24', '%.4f & %.4f & %.3f')):
        v = LO[k]; add(f % (v['strip'], v['comb'], v['ratio']), [v['strip'], v['comb'], v['ratio']], 'be_lo.json:table.%s' % k)
    for k in ('1e-256', '1e-1024'):
        v = LO[k]; add('%.5f & %.5f & %.4f' % (v['strip'], v['comb'], v['ratio']), [v['strip'], v['comb'], v['ratio']], 'be_lo.json:table.%s' % k)
    for s, f in (('1e-06', '0.5429'), ('1e-12', '0.2852'), ('1e-24', '0.1690')):
        add('%.4f' % MIX[s], MIX[s], 'bd_opt_comb_%s.json:g_min_full' % s)
    add('$%.3f$, $%.3f$, $%.3f$ and $%.3f$' % tuple(MIX[s] for s in ('0.003', '0.001', '0.0003', '0.0001')), [MIX[s] for s in ('0.003', '0.001', '0.0003', '0.0001')], 'bd_opt_comb_*.json:g_min_full')
    sa, sb, ga, gb = 0.003, 0.001, MIX['0.003'], MIX['0.001']; lc = np.log10(sa) + (ga - 0.884) / (ga - gb) * (np.log10(sb) - np.log10(sa))
    add('1.4\\times10^{-3}', 10 ** lc, 'interpolation of bd_opt_comb_0.003 / 0.001 at 0.884')
    add('%.1f\\times10^{-3}' % (LX['comb_LO_cross'] * 1e3), LX['comb_LO_cross'], 'bf_lo_cross.json:comb_LO_cross')
    add('%.1f\\times10^{-6}' % (LX['strip_LO_cross'] * 1e6), LX['strip_LO_cross'], 'bf_lo_cross.json:strip_LO_cross')
    add('\\approx%d' % round(LX['ratio']), LX['ratio'], 'bf_lo_cross.json:ratio')
    add('$\\Lambda^*=%.2f$, $%.2f$, $%.2f$' % tuple(LA['comb_%s' % s]['Lambda_star'] for s in ('1e-06', '1e-12', '1e-24')), 0, 'be_lambda.json:comb_*.Lambda_star')
    add('%.2f$, $%.2f$, $%.2f' % tuple(LA['strip_%s' % s]['Lambda_star'] for s in ('1e-06', '1e-12', '1e-24')), 0, 'be_lambda.json:strip_*.Lambda_star')
    CP = R('bg_compare.json')['A1_deficit']
    for k in ('comb optimum 1e-24', 'strip optimum 1e-24'):
        v = CP[k]; add('%.2f\\times10^{-26}' % (v['actual'] * 1e26), v['actual'], 'bg_compare.json:A1_deficit.%s.actual' % k)
        add('%.2f\\times10^{-26}' % (v['bound'] * 1e26), v['bound'], 'bg_compare.json:A1_deficit.%s.bound' % k)
    return n


if __name__ == '__main__':
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d found in paper/sections' % (len(N), len(N) - len(miss)))
    for x in miss: print('NOT FOUND:', x['text'], x['source'])
