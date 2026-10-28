"""Stage AM, plan B (staggered double row): leading-order decomposition into the two-line problem (v) and the theta-function (lattice)
factor in w.  With q = exp(-2 pi^2 t/a^2), the blurred row profile is (1/a)[1 + 2 q cos(2 pi (w - o)/a) + O(q^4)], and the two rows have
opposite phases, so to leading order in q
  s_w ~ g(y_v) (2 pi/a) q sin(2 pi w/a),  H_ww ~ g (2 pi/a)^2 q cos(2 pi w/a),  H_vw ~ g' (2 pi/a) q sin(2 pi w/a),
with g the (signed) difference of the two rows' posterior weights.  Hence
  tR_ww = (2 pi/a)^2 t (1 + O(q)),   tR_vw = t E[g'^2]/E[g^2] (a Rayleigh quotient in v only),
  ln(1/tJ) = ln(1/E[g^2]) + 4 pi^2 t/a^2 - ln(2 (2 pi/a)^2 t) + O(q).
So the lattice factor contributes the same amount, 4 pi^2/a^2, to tR and to ln(1/tJ) (it costs nothing in Conjecture 1 with B' = 1), and
the rest is a one-dimensional lever-arm problem in v like the two-line family.  Checked here against the exact numbers.
-> results/am_theta.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np

if __name__ == '__main__':
    st = json.load(open('results/al_staggered.json')); tl = json.load(open('results/al_run.json')); lat = json.load(open('results/ak_lattice1d.json'))
    out = {}
    for k, v in st.items():
        D, a = v['D'], v['a']; o5 = v['taus']['0.5']; tlk = 'tl_%g_0.1_None_0.5' % D
        r = {'D': D, 'a': a, 'tR': v['tR'], 'tR_ww': o5['tR_ww'], 'tR_vw': o5['tR_vw'], 'ln1_tJ': v['ln1_tJ'],
             'pred_tR_ww': (2 * np.pi / a) ** 2, 'lattice_tR_exact': lat['%g' % a]['rows'][0]['tR'] if '%g' % a in lat else None,
             'twoline_tR_vw': tl[tlk]['tR_vw'] if tlk in tl else None, 'twoline_ln': tl[tlk]['ln1_tJ'] if tlk in tl else None}
        r['ln_minus_lattice_part'] = r['ln1_tJ'] - (4 * np.pi ** 2 / a ** 2 - np.log(2 * (2 * np.pi / a) ** 2))
        r['T2_margin_B1'] = r['tR'] - r['ln1_tJ']; r['tR_over_ln'] = r['tR'] / r['ln1_tJ']
        r['margin_vw_part'] = r['tR_vw'] - r['ln_minus_lattice_part']
        out[k] = r
        print(k, ' '.join('%s %.4g' % (q, r[q]) for q in ('tR_ww', 'pred_tR_ww', 'tR_vw', 'twoline_tR_vw', 'ln1_tJ', 'ln_minus_lattice_part', 'twoline_ln', 'T2_margin_B1', 'tR_over_ln') if r[q] is not None))
    json.dump(out, open('results/am_theta.json', 'w'), indent=1)
