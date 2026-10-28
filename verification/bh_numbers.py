"""Stage BH: every number of the revised paper-F draft re-derived from results/ (paper E's form).  Stage BG's list (bg_numbers.py) without the
33-point certificate values, plus the cell certificate (bh_cert_cells), the tail (bh_tail) and the comb-sequence (P8) check (bh_members_fc).
-> results/bh_numbers.json, paperF/paperF_numbers.md.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
from bg_numbers import numbers as bg_numbers

R = lambda f: json.load(open('results/' + f))


def cells():
    C = {}
    for f in sorted(glob.glob('results/bh_cert_cells_part*.json')): C.update(json.load(open(f)))
    return C


def numbers():
    n = [x for x in bg_numbers() if not x['source'].startswith('bf_cert_p.json')]
    add = lambda txt, val, src: n.append({'text': txt, 'value': val, 'source': src})
    C = cells(); add('$%d$ cells of ratio' % len(C), len(C), 'bh_cert_cells_part*.json:count')
    c1n = max(v['c1_needed'] for v in C.values()); add('at most $%.3f$' % (np.ceil(c1n * 1000) / 1000), c1n, 'bh_cert_cells:max c1_needed')
    i = max(C, key=lambda k: C[k]['c1_needed']); add('\\ls\\approx%.1f' % C[i]['la'], C[i]['la'], 'bh_cert_cells:argmax c1_needed la')
    m = min(v['L_cell'] - (1 - 1.7 * np.log(v['la']) / v['la']) for v in C.values()); add('$%.1f\\times10^{-5}$ (last cell)' % (m * 1e5), m, 'bh_cert_cells:min margin at c1 = 1.7')
    add('$d\\ge%.1f$' % (np.floor(min(v['d_sign'] for v in C.values()) * 10) / 10), min(v['d_sign'] for v in C.values()), 'bh_cert_cells:min d_sign')
    add('$\\beta\\le%.3f$' % (np.ceil(max(v['beta'] for v in C.values()) * 1000) / 1000), max(v['beta'] for v in C.values()), 'bh_cert_cells:max beta')
    add('$\\omega\\ge%.5f$' % (np.floor(min(v['omega'] for v in C.values()) * 1e5) / 1e5), min(v['omega'] for v in C.values()), 'bh_cert_cells:min omega')
    add('factor $>%d$' % (np.floor(min(v['Lgt1_margin'] for v in C.values()) / 100) * 100), min(v['Lgt1_margin'] for v in C.values()), 'bh_cert_cells:min Lgt1_margin')
    c0 = C['0']; add('$F_{\\mathrm{int}}\\ge%.5f$' % (np.floor((1 - c0['Fdef']) * 1e5) / 1e5), 1 - c0['Fdef'], 'bh_cert_cells:cell 0 F_int'); add('%s steps reach' % {8: 'eight'}.get(c0['n_steps'], str(c0['n_steps'])), c0['n_steps'], 'bh_cert_cells:cell 0 n_steps')
    for e in (24, 48, 96, 256, 1024, 16384, 100000):
        l = e * np.log(10); v = [c for c in C.values() if c['la'] <= l <= c['lb'] + 1e-9][0]
        add('$%+.2f$ & ' % v['X_minus_quarter_l'], v['X_minus_quarter_l'], 'bh_cert_cells: cell containing 1e-%d X_minus_quarter_l' % e)
        add('& %.5f\\\\' % (np.floor(v['L_cell'] * 1e5) / 1e5), v['L_cell'], 'bh_cert_cells: cell containing 1e-%d L_cell' % e)
    add('\\ge%.4f$' % (np.floor(min(v['phi1'] for v in C.values()) * 1e4) / 1e4), min(v['phi1'] for v in C.values()), 'bh_cert_cells:min phi1')
    T = R('bh_tail.json'); add('from $%.5f$ to $1$' % T['T_at_l_end'], T['T_at_l_end'], 'bh_tail.json:T_at_l_end')
    MF = R('bh_members_fc.json'); add('(1-%.1f\\times10^{-13})' % (np.ceil(MF['D_max'] * 1e14) / 10), MF['D_max'], 'bh_members_fc.json:D_max')
    return n


if __name__ == '__main__':
    tex = ''.join(open(f).read() for f in sorted(glob.glob('paper/sections/*.tex')))
    N = numbers(); miss = [x for x in N if x['text'] not in tex]
    print('%d numbers, %d found in paper/sections' % (len(N), len(N) - len(miss)))
    for x in miss: print('NOT FOUND:', x['text'], x['source'])
