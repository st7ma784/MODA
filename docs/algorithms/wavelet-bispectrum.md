# Wavelet Bispectrum

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what the algorithm computes,
    the relevant source files, and recent performance work.

## What it computes

The bispectrum is a higher-order spectral measure that detects **quadratic phase
coupling**: whether two frequency components f₁ and f₂ in a signal (or pair of
signals) combine nonlinearly to produce energy at their sum frequency f₃ = f₁ + f₂, in
a way that's phase-locked rather than coincidental. A strong bispectral peak at
(f₁, f₂) is evidence of a genuine nonlinear interaction between those two oscillatory
components — something an ordinary (linear) power spectrum cannot detect.

MODA computes the bispectrum from wavelet transforms rather than Fourier transforms,
so — like [time-frequency analysis](time-frequency-analysis.md) — it can track how
this coupling changes over time, and works well on signals with time-varying
frequencies (unlike classical FFT-based bispectrum estimators).

## Source files

- `allguis/guis/bispectrum/Functions/bispecWavNew.m` — main bispectrum computation
  (auto- and cross-bispectrum).
- `allguis/guis/bispectrum/Functions/wtAtf2.m` / `wtAtf2_batch.m` — evaluates the
  wavelet transform at an arbitrary (non-grid) frequency; `_batch` evaluates it at many
  frequencies at once, sharing the signal's FFT/preprocessing across them.
- `allguis/guis/bispectrum/Functions/python/bispecWavPython.m` — Python-packaging
  variant of the same algorithm.
- `allguis/guis/bispectrum/Bispectrum.m` — the app module.

## Performance notes

The bispectrum's O(n_freq²) double loop over frequency pairs (j, k) was recently
optimized: each row's frequencies are now batched into a single `wtAtf2_batch` call
instead of one call per (j, k) pair, since the expensive part (signal FFT/padding/
preprocessing) doesn't depend on which frequency is being evaluated. See
[Refactor Notes](../developer-guide/refactor-notes.md) for the verification
methodology used.

## Key parameters

- **Bispectrum type** (`111`/`112`/`122`/`222`) — which combination of signal(s) and
  auto- vs. cross-coupling to compute; see
  [REST API Reference](../api-and-ml/rest-api-reference.md#bispectrum-analysis) for the
  same typing used by FastMODA.
- **Frequency range and resolution** — inherited from the underlying wavelet
  transform.

## Related pages

- [Time-Frequency Analysis](time-frequency-analysis.md) — the transform the bispectrum
  is built on.
- [Surrogate Testing](surrogate-testing.md) — for assessing whether a bispectral peak
  is statistically significant.
