# resolution-gap-two-sided

**How fine can a hidden ripple be? A two-sided logarithmic law for the resolution gap of the arrow of time**

Code, stored results and computer-assisted proofs for the paper by Taishi Namba (2026).

**Paper:** T. Namba, *How fine can a hidden ripple be? A two-sided logarithmic law for the resolution gap of the arrow of time*, preprint, Zenodo (2026), [doi:10.5281/zenodo.23746972](https://doi.org/10.5281/zenodo.23746972).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23746555.svg)](https://doi.org/10.5281/zenodo.23746555)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neginyan/resolution-gap-two-sided/blob/main/demo/hidden_ripple_demo.ipynb)
[![verify](https://github.com/neginyan/resolution-gap-two-sided/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/resolution-gap-two-sided/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The result in brief

An observer who sees a pendulum's phase space through a Gaussian blur of width σ cannot always resolve the arrow of time near the hyperbolic top. The *resolution gap* g measures by how much. The companion paper ([paper E](https://doi.org/10.5281/zenodo.23617343)) showed that g is not bounded below uniformly in σ: a hidden ripple of the right fineness makes it shrink, like 1/√ln(1/σ) on the strips it searched.

This paper asks how fine such a hidden ripple can be, and answers it on the **periodic windows**: states that carry, inside a narrow window, any even periodic profile, from a single cosine (the strip) to a comb of thin teeth.

- **Two-sided theorem** [proof, computer-assisted]. Under explicit assumptions (P1)–(P8), for every 10⁻¹⁰⁰⁰⁰⁰ ≤ σ ≤ 10⁻²⁴,

  (1 − ε̄)(1 − 1.7 lnℓ/ℓ) ≤ g_min √ℓ ≤ (1 + |ε(S′_C)|)(1 + 6.0 lnℓ/ℓ),  ℓ = ln(1/σ).

  The lower side is certified on 421 cells covering the whole range and all |ε| ≤ ε̄ ≤ 0.1; the upper side on 1464 cells along an explicit comb sequence, in exact interval arithmetic.
- **The limit.** g_min √ln(1/σ) → 1: proved from below (up to 1 − ε̄), derived from above.
- **A sharper lens** [derivation]. At leading order, the comb at blur σ is the strip at blur σ/256. Combs fall below the best Gaussian mixtures already at σ ≈ 1.4 × 10⁻³ [numerical].
- **General states.** A decorrelation dichotomy and a three-case split are proved for every state. The remaining case is reduced to one conjectured inequality, so paper E's Conjecture 1 remains open.

Every statement in the paper carries its status: proof, computer-assisted proof, derivation, numerical, or conjecture.

## Demo

The [Colab notebook](demo/hidden_ripple_demo.ipynb) runs on a CPU in under a minute. It shows:
- the proved band closing towards 1, with the certified cell values and the evaluator's strips and combs inside it;
- the "256 times sharper lens": strip and comb windows side by side, with the leading-order identity recomputed;
- three cells of the upper certificate recomputed in interval arithmetic (h < 0).

## Verify

```bash
pip install -r requirements.txt
python verification/run_all.py              # about 2 minutes
python verification/run_all.py --recompute  # every certificate in full (about 20 minutes)
```

`run_all.py` runs 22 checks:
- the SHA-256 hashes in `MANIFEST.sha256`;
- the stored certificates, with representative cells recomputed (30 lower cells, 184 upper cells);
- the lemma checks against the evaluator;
- 62 numbers of the paper against the stored results;
- the demo notebook.

With `--recompute`, all 421 lower cells, all 1464 upper cells, the class membership and the monotonicity table are recomputed.

The upper certificate uses 60-digit decimal interval arithmetic with outward rounding, so it is independent of the CPU and is compared exactly. Floating-point checks are compared with a stated relative tolerance; each check prints which kind it is.

## Repository layout

| folder | content |
|---|---|
| `src/` | the evaluator of paper E and shared tools |
| `certificates/` | the computer-assisted proofs: lower side (`bh_cert_cells.py` with `bf_cert_p.py`, tail `bh_tail.py`), upper side (`bf_cert_upper.py` with `ba_ivl.py`, `ba_cert.py`), class membership (`bg_members.py`, `bh_members_fc.py`), monotonicity table (`bi_monotone_table.py`) |
| `experiments/` | scripts for the numerical results: comb searches, comparison with mixtures, lemma checks, leading-order formulas |
| `results/` | stored results; `MANIFEST.sha256` lists their SHA-256 |
| `verification/` | `run_all.py`, the single entry point |
| `paper/` | LaTeX source of the paper (REVTeX 4.2) |
| `demo/` | Colab notebook |

## Related work by the author

- T. Namba, *Is the resolution gap of the arrow of time uniform?*, Zenodo (2026), [doi:10.5281/zenodo.23617343](https://doi.org/10.5281/zenodo.23617343) (paper E; code: [resolution-gap-log-law](https://github.com/neginyan/resolution-gap-log-law)).
- T. Namba, *A resolution gap for the coarse-grained arrow of time in nonlinear Hamiltonian flows*, Zenodo (2026), [doi:10.5281/zenodo.23320603](https://doi.org/10.5281/zenodo.23320603).
- T. Namba, *Sharp criterion for the coarse-grained arrow of time in Hamiltonian dynamics*, v1.2, Zenodo (2026), [doi:10.5281/zenodo.23258135](https://doi.org/10.5281/zenodo.23258135).

## Use of AI assistants

The study was carried out with AI assistants: Anthropic Claude (planning, computations, certificates, code, verification and drafting), Google Gemini and OpenAI ChatGPT (review of the manuscript). Their roles are described in the acknowledgments of the paper. The author directed the research and takes full responsibility for it.

## Citation

Please cite the paper ([doi:10.5281/zenodo.23746972](https://doi.org/10.5281/zenodo.23746972)) and this repository; see `CITATION.cff`.
The code is archived at Zenodo: [doi:10.5281/zenodo.23746555](https://doi.org/10.5281/zenodo.23746555) (all versions; v1.0.0 is [10.5281/zenodo.23746556](https://doi.org/10.5281/zenodo.23746556)).

## License

MIT © 2026 Taishi Namba
