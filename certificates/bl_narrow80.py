"""Stage BL: Lemma BK1 with the boundary tau0 = 80 (computer-assisted proof; the same cells and bounds as bk_narrow_lemma.py, tau0 raised so that
above it delta_tau = 2 sum e^{-n^2 tau/2} <= 8.5e-18, small enough for every theta-remainder of Lemmas BL1-BL6 times any power of tau that occurs).
-> results/bl_narrow80.json.  Python 3.11."""
import json
import numpy as np
import bk_narrow_lemma as BK

TAU0 = 80.0

if __name__ == '__main__':
    xs = np.geomspace(1.0, TAU0, 241); res = {'tau0': TAU0, 'cells': 0, 'all_ok': True, 'worst_margin': np.inf}
    for xa, xb in zip(xs[:-1], xs[1:]):
        bs = np.geomspace(xa / TAU0, 1.0, 241)
        for ba, bb in zip(bs[:-1], bs[1:]):
            r = BK.cell(xa, xb, ba, bb); res['cells'] += 1; res['all_ok'] &= r['ok']
            if r['margin'] < res['worst_margin']: res['worst_margin'] = r['margin']; res['worst'] = {'x': [xa, xb], 'b': [ba, bb], **r}
    print('Lemma BK1 (tau <= 80): %d cells, all ok %s, smallest margin %.3f' % (res['cells'], res['all_ok'], res['worst_margin']))
    json.dump(res, open('results/bl_narrow80.json', 'w'), indent=1, default=float)
