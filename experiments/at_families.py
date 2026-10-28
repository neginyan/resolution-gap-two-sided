"""Stage AT: Theorem AT2 on paper E's state families (stage X1 jobs: mixtures, stored states, localised ripples, two clusters) and on a scan of
tilted two-point states (the search for a state where the window route loses).  -> results/at_families.json (restartable).  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paperE_release', 'code'))
from at_tools import sums_grid, analyse
from u_cgm import mixture_terms, ripple_terms, nodes


def x1_terms(job):
    from x1_states import grid_for_terms
    from h2_adversarial import unpack
    key, fam, a1, sig, x = job
    if fam == 'mixture':
        T = mixture_terms(*unpack(x, a1, sig)); vn, wn = grid_for_terms(T, sig)
    elif fam == 'mixture (stored)':
        T = mixture_terms(*a1); vn, wn = grid_for_terms(T, sig)
    elif fam == 'two clusters':
        from u2_chain import two_clusters
        T = mixture_terms(*two_clusters(a1, sig)); vn, wn = grid_for_terms(T, sig)
    elif fam.startswith('localised ripple'):
        o = json.load(open('results/' + a1['file'])); par = dict(o['best']['par']); par['q0'] = a1['par_units']['q0_over_sqrt_sigma'] * np.sqrt(sig)
        T = ripple_terms(sig, par['Lv'], par['Lw'], par['c'], par['k0'], par['Rv'], par['ell'], par['q0'])
        hv = min(0.2 * sig, 2 * np.pi * sig / par['k0'] / 20); hw = 0.35 * min(par['ell'] / 40, (2 * np.pi * sig / par['q0']) / 24 if par['q0'] > 0 else 1e9, 0.5 * np.sqrt(sig))
        vn = nodes(6.5 * par['Lv'], hv, 6.5 * par['Lv'], hv); wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
        while len(vn) * len(wn) > 8.0e6: hw *= 1.1; wn = nodes(6.5 * par['Lw'], hw, 6.5 * par['Lw'], hw)
    else:
        return None
    return T, vn, wn


def tilted_pair(D, ang, sig, wid=0.3):
    """two equal thin Gaussian points (width wid*sigma) at +-(D sigma/2)(cos ang, sin ang) in (v, w) around the top."""
    e = np.array([np.cos(ang), np.sin(ang)]) * D * sig / 2; S = np.eye(2) * (wid * sig) ** 2
    return [(0j, e, S), (0j, -e, S)]


if __name__ == '__main__':
    fn = 'results/at_families.json'; R = json.load(open(fn)) if os.path.exists(fn) else {}
    from x1_states import jobs
    for job in jobs():
        key = job[0]
        if key in R or job[1].startswith('strip') or job[1].startswith('multi'): continue
        out = x1_terms(job)
        if out is None: continue
        T, vn, wn = out; t0 = time.time(); r = analyse(sums_grid(T, job[3], vn, wn), job[3]); r.update({'sigma': job[3], 'family': job[1], 'seconds': time.time() - t0})
        R[key] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
        print('%-44s g %.4f Thm2 %.4f AT2 %.4f  C_w %.4f k_eff %.4f heis %.3f  %.0fs' % (key, r['g'], r['bound_ThmW'], r['bound_AT2'], r['C_w'], r['k_eff'], r['heisenberg_ratio'], r['seconds']), flush=True)
    sig = 1e-3
    for D in (2.0, 4.0, 6.0, 8.0):
        for ang_deg in (0, 10, 30, 60, 85):
            key = 'pair_D%g_ang%d' % (D, ang_deg)
            if key in R: continue
            T = tilted_pair(D, np.radians(ang_deg), sig); ex = D * sig / 2 + 8 * sig
            vn = np.linspace(-ex, ex, 1601); wn = np.linspace(-ex, ex, 1601)
            r = analyse(sums_grid(T, sig, vn, wn), sig); r.update({'sigma': sig, 'family': 'tilted pair', 'D': D, 'ang': ang_deg})
            R[key] = r; json.dump(R, open(fn, 'w'), indent=1, default=float)
            print('%-44s g %.4f Thm2 %.4f AT2 %.4f  C_w %.4f k_eff %.4f heis %.3f' % (key, r['g'], r['bound_ThmW'], r['bound_AT2'], r['C_w'], r['k_eff'], r['heisenberg_ratio']), flush=True)
