"""Stage BI §1.1: size of the grid effect seen when the certificate's own x-grids (re-spaced over the current range) are used in the monotonicity check.
Every value computed on such a grid is an upper bound of the exact supremum, so the certificate stays valid; this script only measures the largest
relative decrease along l and F at the base points of bi_monotone_table.py.  -> results/bi_grid_wiggle.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
import bi_monotone_table as T
import bf_cert_p as B
B.cell_sup = T.B.cell_sup = __import__('be_cert_p').cell_sup        # the certificate's grid
if __name__ == '__main__':
    out = {}
    for bi, base in enumerate([dict(hs=np.exp(-60.0), G=0.13, eta=1 / 0.95, l=60.0, F=0.5)]):
        for var, grid in (('l', np.linspace(60.0, 78.0, 9)), ('F', np.linspace(0.45, 0.5, 9))):
            ser = [T.consts(**{**base, var: v}) for v in grid]
            for name, dirs in T.TABLE.items():
                v = np.array([s[name] for s in ser]); d = np.diff(v) * (dirs[T_idx] if (T_idx := ['hs', 'G', 'eta', 'l', 'F'].index(var)) is not None else 1)
                worst = float(max(0.0, -(d / np.maximum(np.abs(v[:-1]), 1e-300)).min())) if dirs[T_idx] != 0 else 0.0
                if worst > 0: out['%s/%s' % (var, name)] = worst
    out['max_relative_wiggle'] = max(out.values()) if out else 0.0
    print(out); json.dump(out, open('results/bi_grid_wiggle.json', 'w'), indent=1)
