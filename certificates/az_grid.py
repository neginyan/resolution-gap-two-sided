"""Stage AZ: dense check of U_main sqrt(ln(1/sigma)) along the sequence S' for ln(1/sigma) from ln(1e24) to ln(1e100000) (decimal arithmetic),
and the ratio (U sqrt(ln) - 1)/(ln ln / ln).  -> results/az_grid.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from az_upper import U_main_dec

if __name__ == '__main__':
    lns = np.exp(np.linspace(np.log(24 * np.log(10)), np.log(100000 * np.log(10)), 1500)); rows = []
    for L in lns:
        u, k0, qb = U_main_dec(L); rows.append((float(L), u, (u - 1) / (np.log(L) / L), k0, qb))
    R = np.array(rows)
    out = {'n': len(rows), 'ln_min': float(R[0, 0]), 'ln_max': float(R[-1, 0]), 'max_ratio': float(R[:, 2].max()), 'ratio_at_ends': [float(R[0, 2]), float(R[-1, 2])],
           'ratio_monotone_decreasing': bool(np.all(np.diff(R[:, 2]) < 0)), 'U_sqrt_ln_monotone_decreasing': bool(np.all(np.diff(R[:, 1]) < 0)),
           'U_sqrt_ln_ends': [float(R[0, 1]), float(R[-1, 1])], 'min_U_sqrt_ln_minus_1': float((R[:, 1] - 1).min()), 'samples': R[::100].tolist()}
    print('n %d: (U sqrt(ln) - 1)/(lnln/ln) max %.4f, from %.4f to %.4f (decreasing: %s); U sqrt(ln) from %.4f to %.6f (decreasing: %s, > 1: %s)' % (
        out['n'], out['max_ratio'], out['ratio_at_ends'][0], out['ratio_at_ends'][1], out['ratio_monotone_decreasing'], out['U_sqrt_ln_ends'][0], out['U_sqrt_ln_ends'][1],
        out['U_sqrt_ln_monotone_decreasing'], out['min_U_sqrt_ln_minus_1'] > 0))
    json.dump(out, open('results/az_grid.json', 'w'), indent=1)
