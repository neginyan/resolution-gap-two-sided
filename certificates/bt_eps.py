"""Stage BT §2.1: paper E's remainder eps on the comb sequence S'_C, bounded for every k0 >= 3.43 (sigma <= 1e-24) — proof with constants in
outward-rounded interval arithmetic (asymptotic forms of bs_upper_tail.py, here with K0 = 3.43 and constant relative remainders).

Definitions (paper F, Sec. 2; paper E Theorem 2):  sigma den g = 2 sigma^2 J_ww + N = 2 A_w + N_phi + (B + R3),
  eps = -(B + R3)/D,  D := sigma a sqrt(P_w) = 2 sqrt(2 A_w N_phi),
  N = -sigma^-2 E[Cov(x_p, x_z - sin x_z | y)],  R3 := N - sigma^2 E[H_pz phi],  B := sigma^2 E[H_pz phi] - (sigma^2/2) E[(s_v^2 - s_w^2) phi].
Step 1 (exact).  Integrating by parts in v and w (E[H_vv F] = E[s_v^2 F] + E[s_v d_v F]):
  B = (sigma^2/2) E[s_v d_v phi - s_w d_w phi],  d_v phi = sin m_z (1/sqrt2 - sigma^2 H_vz),  so with s_p = (s_v - s_w)/sqrt2 and E[s_p sin m_z] = sigma^2 E[cos m_z H_pz]:
  B = (sigma^4/2) E[cos m_z H_pz] - (sigma^4/2) E[sin m_z (s_v H_vz - s_w H_wz)].
Step 2 (exact + Lagrange).  g(z) = z - sin z, Taylor around m_z = E[x_z|y] to third order (|g'''| <= 1); Tweedie: Cov(x_p, x_z|y) = sigma^4 d_pz ln p,
  Cov(x_p, (x_z - m_z)^2 | y) = sigma^6 d_pzz ln p (third posterior cumulant).  Hence
  R3 = T2 + R_L,  T2 = -(sigma^4/2) E[sin m_z d_pzz ln p],  |R_L| <= sigma^-2 (1/6) E_y[ sd(x_p|y) sqrt(E[(x_z - m_z)^6|y]) ]
       <= (sigma^-2/6) sqrt(E_y Var(x_p|y)) sqrt(E (x_z - m_z)^6) <= (sigma^2/6) sqrt(960):
  E_y Var(x_p|y) = sigma^2 - sigma^4 E[H_pp] <= sigma^2 (E[H] = E[s s^T]);  ||x_z - m_z||_6 <= ||x_z - y_z||_6 + ||m_z - y_z||_6 <= 2 ||x_z - y_z||_6 (Jensen),
  E (x_z - y_z)^6 = 15 sigma^6 (the blur is independent Gaussian noise) — a bound that does not depend on the state.
Step 3 (Cauchy estimates).  p = N_V(v) Gamma(w) h(k'v, w)/Z, h = 1 + u, u = sum_n q0_n e^{-w^2/2s^2} cos(n k' v), Z in [1 - Qbar, 1 + Qbar].  On the polydisc
  |v' - v| <= 1/k', |w' - w| <= s:  |u| <= eta(w) = e^{1/2} Q_c psi(w),  psi(w) = exp(-(|w| - s)_+^2/(2 s^2)),  Q_c = sum |q0_n| cosh n,
  so |ln h| <= c_M psi(w), c_M = e^{1/2} Q_c/(1 - e^{1/2} Q_c), and |d_v^a d_w^b ln h| <= a! b! k'^a s^-b c_M psi(w).  The Gaussian factors give the slow
  parts exactly (s_v = -v/V + ..., H_vv = 1/V + ..., H_ww = 1/L^2 + ..., H_vw = 0 + ...).
Step 4.  Every bound is a sum of terms coef |v|^a |w|^b psi^j; E_p[...] <= (1 + Qbar)/(1 - Qbar) * int N_V Gamma (...), with
  int N_V |v|^a = V^{a/2} mu_a,  int Gamma |w|^b = L^b mu_b,  int Gamma |w|^b psi^j <= (2 pi)^{-1/2} L^-1 I_j(b) s^{b+1} (j >= 1, Gamma <= 1/(sqrt(2 pi) L)).
Step 5 (denominator).  A_w >= sigma^2 J_ww/2 >= sigma^2/(2 E_p w^2) >= sigma^2 (1 - Qbar)/(2 L^2 (1 + Qbar)) (1 - phi/4 >= 1/2, Cramer-Rao).
  N_phi = (sigma^2/2) E[s_v^2 phi] restricted to R' = {|v| <= 3 sqrt V, w0 <= |w| <= 3L}, w0 = s(1 + sqrt8 k0) (psi <= E^4 there):
  phi >= (m_z^2/2)(1 - M^2/12),  (s_v m_z)^2 >= (v z/V)^2/2 - r0^2,  int_{R'} (v z/V)^2 N_V Gamma >= c2 W2/(2V),  c2 = int_{|x|<=3} x^2 dN(x).
All quantities are carried as asymptotic forms c k^beta E^gamma (1 + R k^-nu) valid for every k0 >= K0 = 3.43 (bs_upper_tail.AF with K0 = 3.43;
super-exponentially small relative terms are bounded by their value at K0, nu = 0).  Result: |eps(S'_C)| <= C k0^beta E^gamma for all k0 >= 3.43.
-> results/bt_eps.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, math
from fractions import Fraction as Fr
from decimal import Decimal as D
from ba_ivl import I
import bs_upper_tail as BU
from bs_upper_tail import AF, Q2, hull

K0 = D('3.43'); K0F = Fr(343, 100)
BU.K0 = K0F; BU.LN_K0 = I(K0).ln(); BU.NU_TINY = Fr(0)


def _tiny_sup(p, q, nu):
    p, q, nu = Fr(p), Fr(q), Fr(nu); assert q > 0 and p + nu < 2 * q * K0F * K0F, (p, q, nu)
    e = I(D((p + nu).numerator) / D((p + nu).denominator)) * BU.LN_K0 - I(D(q.numerator) / D(q.denominator)) * I(K0) * I(K0)
    return I(D(0), e.exp().hi)


BU.tiny_sup = _tiny_sup
PI = I('3.14159265358979323846264338327950288419716939937510582097494', '3.14159265358979323846264338327950288419716939937510582097495')
S2PI = (2 / PI).sqrt()                                     # sqrt(2/pi)
MU = {0: I(1), 1: S2PI, 2: I(1), 3: 2 * S2PI, 4: I(3)}     # E|X|^a, X ~ N(0, 1)
MHALF = {0: (PI / 2).sqrt(), 1: I(1), 2: (PI / 2).sqrt(), 3: I(2), 4: 3 * (PI / 2).sqrt()}   # int_0^inf x^i e^{-x^2/2} dx


def AFc(x): return AF(x if isinstance(x, I) else I(x))


def ipow(x, n):
    r = I(1)
    for _ in range(n): r = r * x
    return r


def apow(x, n):
    r = AF(1)
    for _ in range(n): r = r * x
    return r


def Ij(j, b):
    """int psi^j |w|^b dw = s^{b+1} * Ij(j, b)."""
    tot = I(1) / (b + 1)
    for i in range(b + 1):
        tot = tot + I(math.comb(b, i)) * MHALF[i] / ipow(I(j).sqrt(), i + 1)
    return 2 * tot


class Poly:
    """sum of coef |v|^a |w|^b psi^j (coef: AF > 0)."""
    def __init__(self, terms=None): self.t = dict(terms or {})
    @staticmethod
    def mono(c, a=0, b=0, j=0): return Poly({(a, b, j): c if isinstance(c, AF) else AF(c)})
    def __add__(x, y):
        out = dict(x.t)
        for k, c in y.t.items(): out[k] = out[k] + c if k in out else c
        return Poly(out)
    def __mul__(x, y):
        if not isinstance(y, Poly): y = Poly.mono(y)
        out = {}
        for (a1, b1, j1), c1 in x.t.items():
            for (a2, b2, j2), c2 in y.t.items():
                k = (a1 + a2, b1 + b2, j1 + j2); c = c1 * c2; out[k] = out[k] + c if k in out else c
        return Poly(out)
    __rmul__ = __mul__


def setup():
    sig = AF(256, -11, 4); Lw = AF(8, -4, 1); Lv = AF(32, -8, 2); ell = AF(32, -6, 2)
    rho = sig.sq() / Lv.sq(); V = Lv.sq() * (1 + rho); L2 = Lw.sq() + sig.sq(); Sb = 1 / (1 / Lw.sq() + 1 / ell.sq())
    s2 = 1 / (1 / (Sb + sig.sq()) - 1 / L2); s = s2.sqrt(); L = L2.sqrt()
    kp = AF(1, 1, 0) / sig                                  # k' = (k0/sigma) Lv^2/V <= k0/sigma
    E = lambda c, p, q: AF(I(c), p, q)
    # q0_n = C_w a_n e^{-n^2 K/2}, C_w <= 1, |a_n| <= 2, K = k^2/(1 + rho) >= k^2 (1 - rho):  e^{-K/2} <= E^{1/2} e^{k^2 rho/2} <= E^{1/2}(1 + 64 k^-4 E^4)
    one_t = BU.one_plus_tiny(I(64), -4, 4)
    cosh1 = (I(1).exp() + (-I(1)).exp()) / 2
    # n >= 2: sum 2 cosh(n) e^{-n^2 K/2} <= 2 cosh(1) e^{-K/2} * 2 (cosh 2/cosh 1) e^{-3K/2} (ratios <= 1/2), e^{-3K/2} <= E^{1.49}
    tail = BU.one_plus_tiny(4 * ((I(2).exp() + (-I(2)).exp()) / 2) / cosh1, 0, Fr(149, 100))
    Qc = AF(2 * cosh1, 0, Fr(1, 2)) * one_t * tail
    Qb = AF(2, 0, Fr(1, 2)) * one_t * BU.one_plus_tiny(I(4), 0, Fr(149, 100))     # Qbar = sum |q0_n| <= 2 e^{-K/2}(1 + 2 e^{-3K/2}...)
    eh = I('0.5').exp()
    cM = AF(eh) * Qc / (1 - AF(eh) * Qc)
    return dict(sig=sig, V=V, L2=L2, L=L, s=s, s2=s2, kp=kp, Qc=Qc, Qb=Qb, cM=cM, rho=rho)


def bound():
    q = setup(); sig, V, L2, L, s, kp, cM, Qb = q['sig'], q['V'], q['L2'], q['L'], q['s'], q['kp'], q['cM'], q['Qb']
    r2 = AFc(I(2).sqrt()); is2 = AFc(1 / I(2).sqrt())
    P = Poly.mono
    sv = P(1 / V, a=1) + P(kp * cM, j=1)                                     # |s_v|
    sw = P(1 / L2, b=1) + P(cM / s, j=1)                                     # |s_w|
    sz = (sv + sw) * is2                                                     # |s_z|
    s4 = sig.sq().sq(); s2g = sig.sq()
    mz = P(is2, a=1) + P(is2, b=1) + sz * s2g                                # |m_z| (>= |sin m_z|)
    Hpz = P(1 / (2 * V)) + P((kp.sq() + 1 / s.sq()) * cM, j=1)
    Hvz = (P(1 / V) + P((2 * kp.sq() + kp / s) * cM, j=1)) * is2
    Hwz = (P(1 / L2) + P((2 / s.sq() + kp / s) * cM, j=1)) * is2
    d3 = P((6 * kp.sq() * kp + 2 * kp.sq() / s + 2 * kp / s.sq() + 6 / (s.sq() * s)) * cM * AFc(1 / I(8).sqrt()), j=1)
    Ch = (1 + Qb) / (1 - Qb)
    def EB(poly):
        tot = None
        for (a, b, j), c in poly.t.items():
            va = apow(V.sqrt(), a)
            if j == 0: wb = apow(L, b) * AFc(MU[b])
            else: wb = apow(s, b + 1) * AFc(Ij(j, b) / (2 * PI).sqrt()) / L
            term = c * va * AFc(MU[a]) * wb
            tot = term if tot is None else tot + term
        return tot * Ch
    B1 = EB(Hpz) * (s4 / 2)
    B2 = EB(mz * (sv * Hvz + sw * Hwz)) * (s4 / 2)
    T2 = EB(mz * d3) * (s4 / 2)
    RL = s2g * AFc(I(960).sqrt() / 6)
    num = B1 + B2 + T2 + RL
    # denominator
    E4 = AF(1, 0, 4); k1 = AF(1, 1, 0)
    w0 = s * (1 + AFc(I(8).sqrt()) * k1)                                     # (w0 - s)^2/(2 s^2) = 8 k^2/2 = 4 k^2, so psi(w0) = E^4
    psi0 = E4
    F0 = kp * cM * psi0
    Sz = (AFc(3) / V.sqrt() + F0 + AFc(3) / L + cM * psi0 / s) * is2
    M = (AFc(3) * V.sqrt() + AFc(3) * L) * is2 + s2g * Sz
    r0 = AFc(3) * s2g * Sz / V.sqrt() + F0 * M
    c2 = I('0.9707091134651118') - I('1e-15')                                # int_{|x|<=3} x^2 dN(x) = erf(3/sqrt2) - 3 sqrt(2/pi) e^{-9/2} = 0.97070911...
    W2 = AFc(c2) * L2 - AFc(2 / (3 * (2 * PI).sqrt())) * w0.sq() * w0 / L
    inner = AFc(c2) * W2 / (4 * V) - r0.sq()
    Np = (s2g / 2) * ((1 - Qb) / (1 + Qb)) * ((1 - M.sq() / 12) / 2) * inner
    Aw = s2g * (1 - Qb) / (2 * L2 * (1 + Qb))
    Dlo = 2 * (2 * Aw * Np).sqrt()
    eps = num / Dlo
    parts = {'B1': B1 / Dlo, 'B2': B2 / Dlo, 'T2': T2 / Dlo, 'RL': RL / Dlo}
    return eps, parts, dict(num=num, Dlo=Dlo, Np=Np, Aw=Aw, r0_over=r0.sq() / (AFc(c2) * W2 / (4 * V)), M=M, w0_over_3L=w0 / (3 * L), q=q)


def sup_over_k(f, from_k=None):
    """sup over k >= K0 of f = c k^beta E^gamma (1 + R k^-nu) (gamma > 0: decreasing check inside tiny_sup)."""
    z = f.zrange()
    return f.civ() * _tiny_sup(f.beta, f.gamma, 0) * (1 + z) if f.gamma > 0 else None


def at(f, k):
    return f.val(D(k))


def check_k_check():
    """numbers for the report at single k (interval values)."""
    eps, parts, aux = bound(); out = {}
    for k in ('3.43', '5', '10', '240'):
        out[k] = {'eps_bound': float(at(eps, k).hi), **{n: float(at(v, k).hi) for n, v in parts.items()}}
    return out


if __name__ == '__main__':
    eps, parts, aux = bound()
    sup = sup_over_k(eps)
    print('eps <= %s k^%s E^%s (1 + R k^-%s), R = %s' % (eps.civ(), eps.beta, eps.gamma, eps.nu, eps.R))
    print('sup over k >= 3.43:', sup)
    for n, v in parts.items(): print('  part %s: k^%s E^%s  sup %.3e' % (n, v.beta, v.gamma, float(sup_over_k(v).hi)))
    side = {'w0_le_3L_sup': float(sup_over_k(aux['w0_over_3L']).hi), 'r0sq_over_main_sup': float(sup_over_k(aux['r0_over']).hi) if aux['r0_over'].gamma > 0 else float((aux['r0_over'].civ() * (1 + aux['r0_over'].zrange())).hi),
            'M_sup': float(sup_over_k(aux['M']).hi)}
    print(side)
    vals = {}
    for k in ('3.43', '3.4361775049423464', '5', '10', '240', '1000'):
        vals[k] = float(at(eps, k).hi); print('k0 = %s: |eps| <= %.3e' % (k, vals[k]))
    out = {'form': {'c': [float(eps.civ().lo), float(eps.civ().hi)], 'beta': str(eps.beta), 'gamma': str(eps.gamma), 'nu': str(eps.nu), 'R': [float(eps.R.lo), float(eps.R.hi)]},
           'sup_k_ge_3.43': float(sup.hi), 'parts_sup': {n: float(sup_over_k(v).hi) for n, v in parts.items()},
           'parts_form': {n: {'beta': str(v.beta), 'gamma': str(v.gamma)} for n, v in parts.items()}, 'side': side, 'values': vals,
           'ok': bool(sup.hi < D('0.1') and side['w0_le_3L_sup'] < 1 and side['r0sq_over_main_sup'] < 1 and side['M_sup'] < 1)}
    print('ok', out['ok'])
    json.dump(out, open(os.environ.get('BT_OUT', 'results/bt_eps.json'), 'w'), indent=1)
