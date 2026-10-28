"""Stage BH: numerical consistency check of Lemma BH1 (monotonicity) — not part of the proof.  For sample cells, the certificate is re-run at
interior points l of the cell with the true sigma = e^{-l}, the true G(l), and |eps| in {0, 0.05, 0.1} (eta = 1/(1 - |eps|), bootstrap factor 1 - |eps|);
each point value of K must be <= the cell's K, of Fdef <= the cell's Fdef, and each point bound (divided by 1 - |eps|) >= the cell's L_cell.
-> results/bh_monotone_check.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, glob
import numpy as np
import bh_cert_cells as H

if __name__ == '__main__':
    R = {}
    for f in sorted(glob.glob('results/bh_cert_cells_part*.json')): R.update(json.load(open(f)))
    meta = json.load(open('results/bh_cert_cells_meta.json')); S1, Si1 = meta['S1_P'], meta['S_i_at_1']; out = []
    for i in (0, 5, 20, 60, 150, 300, 420):
        c = R[str(i)]
        for t in (0.0, 0.5, 1.0):
            l = c['la'] + t * (c['lb'] - c['la'])
            for eps in (0.0, 0.05, 0.1):
                H.ETA = 1 / (1 - eps)
                orig = H.B.chain
                r = H.cell(l, l, S1, Si1, sig_override=(np.exp(-l) if l < 690 else 1e-300))
                H.ETA = 1 / 0.9
                # the point run used the bootstrap factor 0.9; with smaller |eps| the true factor is larger, so this point value is itself conservative
                ok = bool(r['ok'] and r['K'] <= c['K'] + 1e-9 and r['Fdef'] <= c['Fdef'] * (1 + 1e-9) + 1e-300 and r['L_cell'] >= c['L_cell'] - 1e-12)
                out.append({'cell': i, 'l': l, 'eps': eps, 'K_point': r['K'], 'K_cell': c['K'], 'L_point': r['L_cell'], 'L_cell': c['L_cell'], 'ok': ok})
                print('cell %3d l %.4g eps %.2f: K %.4f <= %.4f, L %.6f >= %.6f  %s' % (i, l, eps, r['K'], c['K'], r['L_cell'], c['L_cell'], ok), flush=True)
    json.dump(out, open('results/bh_monotone_check.json', 'w'), indent=1, default=float)
