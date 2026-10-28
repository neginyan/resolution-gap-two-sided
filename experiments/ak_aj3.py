"""Stage AK, secondary goal: Proposition AJ3 with explicit error bounds (two w-lines at v = 0, 2D, density 1 per unit w-length;
point (D, 0) of weight e^{-L}, L = D(z + D/2); t = 1).  Exact reduction (u = y_v - D, x = D u, y = y_w, kappa = e^{-Dz}/sqrt(2 pi), g = kappa e^{-y^2/2}):
  p = e^{-(D^2+u^2)/2} (2pi)^{-1/2} [2 cosh x + g],  pi_P = g/(2cosh x + g),  pi_2 - pi_1 = 2 sinh x/(2 cosh x + g),
  s_w = -pi_P y,  H_vw = -pi_P y D (pi_2 - pi_1),  H_ww = pi_P (1 - (1 - pi_P) y^2),
  J = P D^{-1} kappa^2 int y^2 e^{-y^2} F0 dy,   G_vw = P D kappa^2 int y^2 e^{-y^2} F2 dy,   P = e^{-D^2/2}(2pi)^{-1/2},
  F0 = int e^{-x^2/2D^2} dx/(2cosh x + g),  F2 = int e^{-x^2/2D^2} 4 sinh^2 x dx/(2cosh x + g)^3.
Elementary bounds (proof):  pi/2 - g/2 <= I0(g) := int dx/(2cosh x + g) <= pi/2;  pi/4 - g/2 <= I2(g) <= pi/4;
  I0 - pi^3/(16 D^2) <= F0 <= I0;  F2 >= I2 - (pi/8)(2 + pi^2/4)/D^2   (int x^2 (sech - sech^3) = pi + pi^3/8).
Consequences, with e := kappa (2/3)^{3/2}/pi + pi^2/(8 D^2) and C0 = (3/2) ln 2pi - ln(pi^{3/2}/4) (per unit line length; + ln 2 if J is
normalised by the total line density 2):
  D^2/2 - (2/3)^{3/2} kappa D^2/pi - (2 + pi^2/4)/4  <=  tR  <=  (D^2/2 + 7/2)/(1 - e)
  D^2/2 + 2Dz + ln D + C0 + ln(1 - e)  <=  ln(1/(tJ))  <=  ... (upper: J <= J0) ... i.e.  ln(1/tJ) >= D^2/2 + 2Dz + ln D + C0 ,  ln(1/tJ) <= that - ln(1 - e).
Along z_D = 2 ln D / D:  |tR - ln(1/tJ) + 5 ln D + C0| <= 4.4 for D >= 6 (<= 4.6 for D >= 4); the per-D window is computed below (and tR - B' ln(1/tJ) -> +inf for every B' < 1).
This script checks every bound numerically against the separate closed-form evaluator.  -> results/ak_aj3.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json
import numpy as np
from scipy.integrate import quad
from ai_twoline_exact import evaluate

sech = lambda x: 1 / np.cosh(x)
C0 = 1.5 * np.log(2 * np.pi) - np.log(np.pi ** 1.5 / 4)

if __name__ == '__main__':
    out = {'C0_per_unit_length': C0}
    out['int_x2_sech'] = quad(lambda x: x * x * sech(x), -80, 80)[0]; out['pi3_over_4'] = np.pi ** 3 / 4
    out['int_x2_sech3'] = quad(lambda x: x * x * sech(x) ** 3, -80, 80)[0]; out['pi3_over_8_minus_pi'] = np.pi ** 3 / 8 - np.pi
    gs = np.linspace(0, 0.4, 9)
    I0 = [quad(lambda x: 1 / (2 * np.cosh(x) + g), -60, 60)[0] for g in gs]; I2 = [quad(lambda x: 4 * np.sinh(x) ** 2 / (2 * np.cosh(x) + g) ** 3, -60, 60)[0] for g in gs]
    out['I0_bounds_ok'] = bool(all(np.pi / 2 - g / 2 - 1e-12 <= a <= np.pi / 2 + 1e-12 for g, a in zip(gs, I0)))
    out['I2_bounds_ok'] = bool(all(np.pi / 4 - g / 2 - 1e-12 <= a <= np.pi / 4 + 1e-12 for g, a in zip(gs, I2)))
    rows = []
    for D in (4.0, 6.0, 10.0, 20.0, 40.0):
        z = 2 * np.log(D) / D; kap = np.exp(-D * z) / np.sqrt(2 * np.pi); e = kap * (2 / 3) ** 1.5 / np.pi + np.pi ** 2 / (8 * D * D)
        r = evaluate(D, z); ln_unit = r['ln1_tJ'] - np.log(2.0)          # ai_twoline_exact normalises by the total line density 2
        lo_tR = D * D / 2 - (2 / 3) ** 1.5 * kap * D * D / np.pi - (2 + np.pi ** 2 / 4) / 4; hi_tR = (D * D / 2 + 3.5) / (1 - e)
        ln0 = D * D / 2 + 2 * D * z + np.log(D) + C0
        m = r['tR'] - ln_unit + 5 * np.log(D) + C0
        win_hi = e * D * D / (2 * (1 - e)) + 3.5 / (1 - e); win_lo = -((2 / 3) ** 1.5 * kap * D * D / np.pi + (2 + np.pi ** 2 / 4) / 4 - np.log(1 - e))
        rows.append({'D': D, 'z': z, 'e': e, 'tR': r['tR'], 'tR_lo': lo_tR, 'tR_hi': hi_tR, 'ln_unit': ln_unit, 'ln_lo': ln0, 'ln_hi': ln0 - np.log(1 - e),
                     'margin_plus_5lnD_plus_C0': m, 'window_lo': win_lo, 'window_hi': win_hi,
                     'ok': bool(lo_tR <= r['tR'] <= hi_tR and ln0 - 1e-9 <= ln_unit <= ln0 - np.log(1 - e) + 1e-9 and win_lo <= m <= win_hi and (D < 6 or max(-win_lo, win_hi) <= 4.4))})
        print(rows[-1], flush=True)
    out['rows'] = rows; out['all_ok'] = bool(all(r['ok'] for r in rows) and out['I0_bounds_ok'] and out['I2_bounds_ok'])
    print('int x^2 sech = %.10f (pi^3/4 = %.10f); int x^2 sech^3 = %.10f (pi^3/8 - pi = %.10f); all ok: %s' % (out['int_x2_sech'], out['pi3_over_4'], out['int_x2_sech3'], out['pi3_over_8_minus_pi'], out['all_ok']))
    json.dump(out, open('results/ak_aj3.json', 'w'), indent=1, default=float)
