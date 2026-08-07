# Surrogate Testing

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what surrogate testing does
    and the relevant source files.

## What it computes

Surrogate testing answers the question: *"is this result (coherence, bispectral peak,
coupling strength...) actually meaningful, or could it have arisen by chance from
unrelated signals with similar basic statistical properties?"*

The method generates many **surrogate signals** — randomized versions of the original
that preserve some property (typically the power spectrum / autocorrelation structure)
but destroy the property being tested for (e.g. phase relationships between signals).
The analysis is then re-run on the surrogates, giving a distribution of "chance"
results to compare the real result against. If the real result falls well outside the
surrogate distribution, it's considered statistically significant.

## Source files

- `surrcalc.m` and the IAAFT (Iterative Amplitude-Adjusted Fourier Transform) family of
  functions generate surrogates that preserve a signal's amplitude distribution and
  power spectrum while randomizing phase relationships. IAAFT's convergence loop is
  inherently sequential (each iteration refines the previous surrogate) and is not a
  vectorization target.
- Exposed in the desktop app via each module's surrogate-testing controls, and via
  FastMODA's `/tests` diagnostics page and the `n_surrogates` parameter on
  `/analyze_bayesian` (see [REST API Reference](../api-and-ml/rest-api-reference.md#bayesian-inference)).

## Key parameters

- **Number of surrogates** — more surrogates give a more precise significance
  threshold at the cost of computation time; 19+ is a common minimum for a 5%
  significance level (giving a rank-based p-value).
- **Surrogate method** — which properties of the original signal are preserved (e.g.
  phase-randomized vs. IAAFT).

## Related pages

- [Wavelet Phase Coherence](wavelet-phase-coherence.md) and
  [Wavelet Bispectrum](wavelet-bispectrum.md) — the two analyses most commonly paired
  with surrogate testing in MODA.
- [Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md)
  — conceptual background on significance testing.
