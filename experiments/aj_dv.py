"""Stage AJ: the lever-arm / Donsker-Varadhan chain for (T2), measured step by step on the one-w-line + point family (t = 1).
Two-level representation (proved, stage AJ): split t = t1 + t2 (t1 = tau t), X1 = X + sqrt(t1) Z1, Y = X1 + sqrt(t2) Z2, sigma = grad ln rho_1.
  s_w(y) = E[sigma_w(X1) | y],      t2 d_j s_w(y) = Cov(X1_j, sigma_w(X1) | y)        (exact)
Chain:  t2^2 |grad s_w|^2  <=(CS)  tr Cov(X1|y) V(y)  <=  a(y) V(y),   V = Var(sigma_w(X1)|y),  a = E[|X1 - y|^2 | y]
        E_p[a V] = EV * E_mu[a]  (mu = p V / EV),  E_mu[a] <=(DV) (1/beta)[KL(mu||p) + ln E_p e^{beta a}],  beta = 1/(4 t2)
        KL(mu||p) <= ln(E_p V^2/EV^2) <= ln(3/(t1^2 EV^2)),  ln E_p e^{beta a} <= ln 2,   EV = J(t1) - J(t)   (law of total variance)
Final (Proposition AJ1):  tR <= (4/(1-tau)) eps [2 ln(1/(t1 J eps)) + ln 6],  eps = J(tau t)/J(t) - 1.
Posterior of X1 given y: two Gaussian components (line: N(((t1/t) y_v, y_w), diag(t1 t2/t, t2)); point: N((D,0) + (t1/t)(y - (D,0)), t1 t2/t I)),
weights = the blur-t posterior weights; inner integrals by 2-d Gauss-Hermite; everything in the log domain.  -> results/aj_dv.json.  Python 3.11."""
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('src', 'certificates', 'experiments', 'verification'):
    _p = _os.path.join(_R, _d)
    if _p not in _sys.path: _sys.path.insert(0, _p)
_os.chdir(_R)
import json, os, sys
import numpy as np
from scipy.special import logsumexp, log_expit

LN2PI = np.log(2 * np.pi)


def lse(a, axis=None, b=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = logsumexp(a, axis=axis, b=b, return_sign=b is not None)
    return r


def chain(D, z, tau=0.5, ngh=40, hv=None, hw=0.1, Wmax=7.0):
    t, t1 = 1.0, tau; t2 = t - t1; L = D * (z + D / 2)
    hv = hv or min(0.02, 0.25 / D)
    yv = np.arange(-8.0, D + max(z, 0) + 10.0, hv); yw = np.arange(-Wmax, Wmax + hw / 2, hw)
    gx, gwt = np.polynomial.hermite_e.hermegauss(ngh); lgw = np.log(gwt / np.sqrt(2 * np.pi))
    U1, U2 = np.meshgrid(gx, gx, indexing='ij'); U1 = U1.ravel(); U2 = U2.ravel(); LG = (lgw[:, None] + lgw[None, :]).ravel()
    acc = {k: [] for k in ('G0', 'CS', 'aV', 'V', 'V2', 'a_mgf', 'Vlnratio_num', 'idmax', 'sw_check')}
    for yw_ in yw:
        V_, W_ = yv, np.full_like(yv, yw_)
        # blur-t posterior weights and ln p
        aL = -V_ ** 2 / (2 * t) - 0.5 * (LN2PI + np.log(t)); aP = -L - ((V_ - D) ** 2 + W_ ** 2) / (2 * t) - (LN2PI + np.log(t))
        lp = np.logaddexp(aL, aP); lpiL, lpiP = aL - lp, aP - lp
        # components of X1 | y: means and std
        sv = np.sqrt(t1 * t2 / t)
        comps = [(lpiL, (t1 / t) * V_, W_, sv, np.sqrt(t2)), (lpiP, D + (t1 / t) * (V_ - D), (t1 / t) * W_, sv, sv)]
        # nodes: x = m + s * u ; weights lpi_c + LG
        X1v = np.concatenate([c[1][:, None] + c[3] * U1[None, :] for c in comps], 1)
        X1w = np.concatenate([c[2][:, None] + c[4] * U2[None, :] for c in comps], 1)
        LWt = np.concatenate([c[0][:, None] + LG[None, :] for c in comps], 1)          # log posterior weight of each node
        # score of rho_1 in w at X1: sigma_w = -pi1_P(X1) X1w / t1 ;  pi1_P logit
        l1 = -L - ((X1v - D) ** 2 + X1w ** 2) / (2 * t1) - (LN2PI + np.log(t1)) + X1v ** 2 / (2 * t1) + 0.5 * (LN2PI + np.log(t1))
        lpi1 = log_expit(l1)
        with np.errstate(divide='ignore'):
            lsig = lpi1 + np.log(np.abs(X1w)) - np.log(t1); ssig = -np.sign(X1w)
        # s_w(y) = E[sigma_w | y]
        ls, ss = lse(LWt + lsig, axis=1, b=ssig)
        sw_closed = -np.exp(lpiP) * W_ / t
        # centred quantities: d = sigma - s (signed log), V = E d^2
        dlog, dsg = lse(np.stack([lsig, np.broadcast_to(ls[:, None], lsig.shape)]), axis=0, b=np.stack([ssig, -np.broadcast_to(ss[:, None], ssig.shape)]))
        lV = lse(LWt + 2 * dlog, axis=1)
        # Cov(X1_j, sigma) = E[(X1_j - m_j)(sigma - s)]
        mv = np.exp(lse(LWt + np.log(np.abs(X1v) + 1e-300), axis=1, b=np.sign(X1v))[0]) * lse(LWt + np.log(np.abs(X1v) + 1e-300), axis=1, b=np.sign(X1v))[1]
        mw = np.exp(lse(LWt + np.log(np.abs(X1w) + 1e-300), axis=1, b=np.sign(X1w))[0]) * lse(LWt + np.log(np.abs(X1w) + 1e-300), axis=1, b=np.sign(X1w))[1]
        cov = []
        for Xc, m in ((X1v, mv), (X1w, mw)):
            dx = Xc - m[:, None]
            with np.errstate(divide='ignore'): l_, s_ = lse(LWt + np.log(np.abs(dx)) + dlog, axis=1, b=np.sign(dx) * dsg)
            cov.append((l_, s_))
        trC = np.exp(lse(LWt + np.log((X1v - mv[:, None]) ** 2 + (X1w - mw[:, None]) ** 2), axis=1))
        a = np.exp(lse(LWt + np.log((X1v - V_[:, None]) ** 2 + (X1w - W_[:, None]) ** 2), axis=1))
        # closed-form |grad s_w|^2 at blur t: H_vw = pi(1-pi) D y_w, H_ww = pi (1 - (1-pi) y_w^2)   (t = 1)
        lq = lpiL + lpiP
        with np.errstate(divide='ignore'):
            lG = np.logaddexp(2 * (lq + np.log(D) + np.log(abs(yw_) + 1e-300)), 2 * lpiP + 2 * np.log(np.abs(1 - np.exp(lpiL) * yw_ ** 2)))
            lCS = 2 * np.logaddexp(cov[0][0], cov[1][0]) * 0 + np.logaddexp(2 * cov[0][0], 2 * cov[1][0]) - 2 * np.log(t2)   # identity: |grad s_w|^2 = sum_j Cov_j^2 / t2^2
        lw = lp + np.log(hv * hw)
        acc['G0'].append(lse(lw + lG)); acc['idmax'].append(float(np.nanmax(np.abs(np.exp(lCS - lG) - 1)[lG > lG.max() - 30])))
        acc['CS'].append(lse(lw + np.log(trC) + lV - 2 * np.log(t2)))
        acc['aV'].append(lse(lw + np.log(a) + lV - 2 * np.log(t2)))
        acc['V'].append(lse(lw + lV)); acc['V2'].append(lse(lw + 2 * lV))
        acc['a_mgf'].append(lse(lw + a / (4 * t2)))
        acc['Vlnratio_num'].append((lw + lV, lV, a))
        m_ = np.abs(sw_closed) > 0
        acc['sw_check'].append(float(np.max(np.abs(ss * np.exp(ls) / np.where(m_, sw_closed, 1) - 1)[m_ & (lpiP > -600)])) if np.any(m_ & (lpiP > -600)) else 0.0)
    lG0 = lse(np.array(acc['G0'])); lV = lse(np.array(acc['V'])); lV2 = lse(np.array(acc['V2'])); lmgf = lse(np.array(acc['a_mgf']))
    # normalisation: the window of w-length 2 Wmax plays the role of the line's (envelope) length: M = total mass in the window,
    # so p/M is a probability measure on the window; E_p e^{beta a} <= 2 then holds as a true expectation
    lM = float(lse(np.concatenate([x[0] for x in acc['Vlnratio_num']]) - np.concatenate([x[1] for x in acc['Vlnratio_num']])))
    # KL(mu || p) with mu = p V / EV  (p normalised by M):  E_mu ln(V M / EV)
    allw = np.concatenate([x[0] for x in acc['Vlnratio_num']]); alllV = np.concatenate([x[1] for x in acc['Vlnratio_num']]); alla = np.concatenate([x[2] for x in acc['Vlnratio_num']])
    mu = np.exp(allw - lV); KL = float(np.sum(mu * (alllV + lM - lV))); Emu_a = float(np.sum(mu * alla))
    beta = 1 / (4 * t2); EVn = lV - lM                      # ln of normalised EV
    out = {'D': D, 'z': z, 'tau': tau}
    out['identity_max_rel'] = float(np.max(acc['idmax'])); out['tweedie_sw_max_rel'] = float(np.max(acc['sw_check']))
    G0 = 1.0
    out['loss_CS'] = float(np.exp(lse(np.array(acc['CS'])) - lG0)); out['loss_a'] = float(np.exp(lse(np.array(acc['aV'])) - lG0))
    dv_exact = np.exp(lV - 2 * np.log(t2) - lG0) * (KL + (lmgf - lM)) / beta
    out['loss_DV'] = float(dv_exact)
    out['KL'] = KL; out['KL_jensen'] = float(lV2 + lM - 2 * lV); out['KL_final'] = float(np.log(3 / t1 ** 2) - 2 * EVn); out['lnmgf'] = float(lmgf - lM)
    out['Emu_a_over_t2'] = Emu_a / t2
    out['loss_KLjensen_mgf2'] = float(np.exp(lV - 2 * np.log(t2) - lG0) * (out['KL_jensen'] + np.log(2)) / beta)
    out['loss_final'] = float(np.exp(lV - 2 * np.log(t2) - lG0) * (out['KL_final'] + np.log(2)) / beta)
    # tR, J, eps from the same quantities (J at blur t from the closed form, EV = J(t1) - J(t))
    return out, lG0, lV, lM


if __name__ == '__main__':
    from ai_lp_exact import evaluate
    fn = 'results/aj_dv.json'; res = json.load(open(fn)) if os.path.exists(fn) else {}
    cases = [(8.0, 0.0), (8.0, 4.0), (8.0, 8.0), (12.0, 0.0), (12.0, 6.0), (12.0, 12.0)] if len(sys.argv) < 2 else [tuple(map(float, sys.argv[1:3]))]
    for D, z in cases:
        for tau in (0.5,):
            k = '%g_%g_%g' % (D, z, tau)
            if k in res: continue
            out, lG0, lV, lM = chain(D, z, tau)
            ex = evaluate(D, z)                         # tR, ln(1/tJ) at blur t (separate code)
            lJ = np.logaddexp(0.0, -D * (z + D / 2)) - ex['ln1_tJ']        # raw (per unit line length) ln J from the separate code
            out['tR'] = ex['tR']; out['ln1_tJ_window'] = float(ex['ln1_tJ'] + lM - np.logaddexp(0.0, -D * (z + D / 2))); out['lnM_window'] = lM
            out['tR_from_chain_G0'] = float(np.exp(lG0 - lJ)); out['eps'] = float(np.exp(lV - lJ))
            out['bound_AJ1'] = (4 / (1 - tau)) * out['eps'] * (2 * (out['ln1_tJ_window'] + np.log(1 / tau) - np.log(out['eps'])) + np.log(6))
            out['bound_over_tR'] = out['bound_AJ1'] / out['tR']; out['tR_over_ln'] = out['tR'] / out['ln1_tJ_window']
            res[k] = out; json.dump(res, open(fn, 'w'), indent=1)
            print(k, json.dumps({q: (round(v, 4) if isinstance(v, float) else v) for q, v in out.items()}), flush=True)
