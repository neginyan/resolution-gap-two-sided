"""Stage BF §1.2: Theorem AZ1-P's final one-variable inequality on the comb sequence S'_C, certified cell by cell (as ba_cert.py for S').

Claim:  h(k) := U_P(k) sqrt(t(k)) - 1 - C ln t(k)/t(k) < 0  for every k in [k_lo, k_hi],   C = 6.0,
  t(k) = 4k^2 + 11 ln k - 8 ln 2 = ln(1/sigma) on S'_C (bf_upper.seqC), increasing in k; t(k_lo) <= 24 ln 10, t(k_hi) >= 1e5 ln 10.
  U_P = Lemma AZ1-P's closed-form bound (bf_upper.U_main_P) for the full comb a_n = 2 e^{-n^2 tau^2/2}, tau = 1/k^2 (all n).
Method (as stage BA):
 * lengths are exact monomials times O(1) mantissas: sigma = 256 k^-11 e^{-4k^2}, ell = 32 k^-6 e^{-2k^2}, Lw = 8 k^-4 e^{-k^2}, Lv = 32 k^-8 e^{-2k^2}
   (class S of ba_cert: exponents exact rationals, mantissas interval dual numbers, outward-rounded 60-digit decimal);
 * the fundamental: qbar_1 = a_1 e^{-K/2}, K = k^2/(1 + rho), a_1 = 2 e^{-1/(2k^4)} (in the mantissa);
 * the harmonics n >= 2 enter only through the sums  Q = qbar_1 (1 + E0), N1 = qbar_1 (1 + E1), N2 = qbar_1 (1 + E2),  sum q0_n^2 = q0_1^2 (1 + Z0),
   sum n^2 q0_n^2 = q0_1^2 (1 + Z2), with 0 <= E_p <= eta_p(K_min), 0 <= Z_p <= zeta_p(K_min) (a_n <= a_1, K >= K_min = 10.5 on the whole range);
   the lower sum in den_lo keeps n = 1 only.  E_p, Z_p are carried as interval parameters with zero k-derivative: for every fixed value of
   the parameters the mean-value form bounds h on the cell, so the interval bound covers the true h(k) (whose parameters lie in the box).
 * cells [a, b]: h <= h(m) + max|h'([a, b])| (b - a)/2; bisect if needed.
-> results/bf_cert_upper.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, time
from fractions import Fraction as Fr
from decimal import Decimal as D
from ba_ivl import I, Dl
from ba_cert import S, _Fu, _psi

CC = D('6.0'); KMIN = D('10.5')


def _series(p, K, sq):
    """sum_{m>=2} m^p e^{-(m^2-1) K/2} (sq: squared terms e^{-(m^2-1) K}), enclosed: m <= 12 summed, tail bounded by 1e-200."""
    s = I(0)
    for m in range(2, 13):
        s = s + I(D(m) ** p) * (I(-(D(m * m - 1)) * K / (1 if sq else 2))).exp()
    return I(D(0), (s.hi + D('1e-200')))


EP = {p: _series(p, KMIN, False) for p in (0, 1, 2)}; ZP = {p: _series(p, KMIN, True) for p in (0, 2)}


def h_of(k):
    one = lambda m: S(0, 0, m, k)
    par = lambda iv: one(Dl(I(1) + iv, I(0)))                 # 1 + [0, bound], zero k-derivative
    sig = S(-4, -11, 256, k); ell = S(-2, -6, 32, k); Lw = S(-1, -4, 8, k); Lv = S(-2, -8, 32, k)
    t = 4 * k.sq() + 11 * k.ln() - 8 * Dl(I(2).ln(), I(0))
    rho = (sig.sq() / Lv.sq()).plain()
    V = Lv.sq() * one(1 + rho); L2 = Lw.sq() + sig.sq()
    Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    a1 = 2 * (-(1 / (2 * k.sq().sq()))).exp()
    qb = S(Fr(-1, 2), 0, a1 * (k.sq() * rho / (2 * (1 + rho))).exp(), k)          # qbar_1 = a_1 exp(-k^2 Lv^2/(2V))
    q0 = qb * (Sb * L2 / (Lw.sq() * (Sb + sig.sq()))).sqrt(); s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2)
    x = S(0, 2, (1 / (1 + rho)).sq(), k)
    s = s2.sqrt(); L = L2.sqrt(); s2g = sig.sq(); b = s2g / V; E1 = Dl(I(1).exp(), I(0))
    Q = qb * par(EP[0]); N1 = qb * par(EP[1]); N2 = qb * par(EP[2]); Sq0 = q0.sq() * par(ZP[0]); Sq2 = q0.sq() * par(ZP[2])
    Qp = Q.plain(); oQ = one(1 - Qp); pQ = one(1 + Qp)
    g15 = (1 + 2 * L2 / s2).p15(); geo = s / (s2 + 2 * L2).sqrt()
    u0 = 4 * L2.sq() * Sq0 / (s2.sq() * g15)
    Jww_up = (1 + u0 / (8 * oQ)) / L2
    I2_lo = q0.sq() * geo; I2_up = Sq2 * geo
    den_lo = b + x * I2_lo / (2 * pQ); Jper_up = x * I2_up / (2 * oQ * s2g)
    bp = b.plain(); a2 = (1 - bp).sq() / 2; y = a2 * V.plain() / 2; e = (-y).exp(); ome = V * one(a2 / 2 * _psi(y))
    xi = sig * x.sqrt() * N1 / oQ; Xi = xi.sq() + s2g * s2g * Q.sq() / (one(E1) * s2 * oQ.sq())
    r = (3 * (xi.sq() + s2g * s2g / L2 + s2g * Q.sq() * (s2g / s2) / (one(E1) * oQ.sq()))).sqrt()
    gam = (x * (N2 * pQ + N1.sq()) + x.sqrt() * N1 * one(1 + 2 * Qp) / one(E1.sqrt())) / oQ.sq()
    v = L2 * s2 / (s2 + 2 * L2); st2w2 = Jper_up * v
    st2Rw = 3 * Xi * Jper_up + 3 * s2g * s2g * st2w2 / L2.sq()
    st2R2 = (st2w2.sqrt() + st2Rw.sqrt()).sq(); omC = (L + r).sq() / 4
    Esv2phi = ome * Jper_up + st2R2 / 4 + ome / V + omC / V + one(a2 * e) + gam * omC
    Uv = (2 * s2g * Jww_up / (sig * den_lo) + s2g / 2 * Esv2phi / (sig * den_lo)).plain()
    return Uv * t.sqrt() - 1 - CC * t.ln() / t, Uv, t


def cell_bound(a, b):
    m = (a + b) / 2; rad = (b - a) / 2
    hm = h_of(Dl(I(m), I(0)))[0].v; hd = h_of(Dl(I(a, b), I(1)))[0]
    return min(hm.hi + hd.d.mag() * rad, hd.v.hi), hm.hi


def t_of(k): return 4 * I(k).sq() + 11 * I(k).ln() - 8 * I(2).ln()


if __name__ == '__main__':
    import numpy as np
    from bf_upper import U_of
    ln10 = I(10).ln(); tlo, thi = 24 * ln10, 100000 * ln10
    k_lo, k_hi = D('3.43'), D('240.0')
    assert t_of(k_lo).hi <= tlo.lo and t_of(k_hi).lo >= thi.hi, (t_of(k_lo), tlo, t_of(k_hi), thi)
    # K >= KMIN on the whole range (K = k^2/(1 + rho) >= k_lo^2/(1 + rho(k_lo)), rho decreasing)
    k0i = Dl(I(k_lo), I(0)); rho_lo = (S(-4, -11, 256, k0i).sq() / S(-2, -8, 32, k0i).sq()).plain().v
    assert (I(k_lo).sq() / (1 + rho_lo)).lo >= KMIN
    agree = []
    for kk in ('3.43', '4', '5', '6', '7'):                 # float check of the formula (bf_upper.U_main_P on the same sequence, all teeth)
        _, Uv, t = h_of(Dl(I(D(kk)), I(0))); tt = float(t.v.mid()); uf = U_of(tt)[0]['U_main']
        agree.append(abs(float(Uv.v.mid()) / uf - 1))
    print('agreement with bf_upper.U_main_P (float, harmonics summed exactly): max rel %.1e (the interval version uses the harmonic upper bounds)' % max(agree), flush=True)
    t0 = time.time(); a = k_lo; ncell = 0; nbis = 0; worst = (D(-10), None); rec = []; step = D('0.02')
    while a < k_hi:
        b = min(a + step * (1 + a / 10), k_hi); stack = [(a, b)]
        while stack:
            u_, v_ = stack.pop(); ub, hm = cell_bound(u_, v_)
            if ub < 0:
                ncell += 1
                if ub > worst[0]: worst = (ub, float(u_))
            else:
                assert v_ - u_ > D('1e-9'), ('cannot certify', u_, v_, ub)
                w = (u_ + v_) / 2; stack += [(w, v_), (u_, w)]; nbis += 1
        if not rec or float(a) >= rec[-1][0] * 1.2: rec.append((float(a), float(hm), float(ub)))
        a = b
        if ncell % 500 == 0: print('k %.3f  cells %d  bisections %d  worst bound %.3e  %.0fs' % (float(a), ncell, nbis, worst[0], time.time() - t0), flush=True)
    out = {'claim': "h = U_P sqrt(ln) - 1 - 6.0 lnln/ln < 0 for 1e-100000 <= sigma <= 1e-24 on the comb sequence S'_C", 'k_range': [float(k_lo), float(k_hi)],
           't_range_checked': [float(t_of(k_lo).hi), float(t_of(k_hi).lo)], 'cells': ncell, 'bisections': nbis, 'worst_upper_bound_of_h': float(worst[0]),
           'worst_at_k': worst[1], 'samples_k_h_mid_h_upper': rec, 'precision_digits': 60, 'agreement_with_float': max(agree), 'K_min': float(KMIN),
           'eta_bounds': {p: float(EP[p].hi) for p in EP}, 'zeta_bounds': {p: float(ZP[p].hi) for p in ZP}, 'seconds': time.time() - t0, 'all_cells_certified': True}
    json.dump(out, open('results/bf_cert_upper.json', 'w'), indent=1)
    print('certified: %d cells, %d bisections, max upper bound of h = %.4e (at k = %.3f), %.0f s' % (ncell, nbis, worst[0], worst[1], time.time() - t0))
