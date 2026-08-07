# Ridge Extraction & Filtering

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what the algorithm computes
    and the relevant source files.

## What it computes

Given a signal's [time-frequency representation](time-frequency-analysis.md), a
**ridge** is the curve of (time, frequency) points along which the representation's
amplitude is locally maximal — it tracks a signal's dominant instantaneous frequency
as it changes over time. **Filtering** then reconstructs a time-domain signal
corresponding to just the oscillation along that ridge (or within a frequency band
around it), separating it from other components/noise.

This is useful when a signal is a mixture of several oscillatory components at
different (possibly time-varying) frequencies — e.g. separating a respiratory
component from a cardiac component in a physiological recording.

## Source files

- `allguis/guis/filtering/Functions/ecurve.m` — ridge extraction (amplitude/index
  tracking, dynamic-programming path optimization).
- `allguis/guis/filtering/Functions/rectfr.m` — reconstruction of the time-domain
  signal from a ridge/band.
- `allguis/guis/filtering/Filtering.m` — the app module.

`ecurve.m`'s ridge tracing (`pathopt`/`onestepopt`) is a dynamic-programming
optimization — each step depends on the previous one, so it is not a batching/matrix
target (see [Refactor Notes](../developer-guide/refactor-notes.md)). Its
amplitude/index extraction at each fixed time step (`max` over frequency) **is**
vectorized.

## Key parameters

- **Ridge-tracking penalty** — controls how much the ridge is allowed to jump in
  frequency between adjacent time points (trades off following genuine frequency
  changes vs. resisting noise).
- **Band width** — for band-based (rather than pure-ridge) filtering, the frequency
  width around the ridge to include in reconstruction.

## Related pages

- [Time-Frequency Analysis](time-frequency-analysis.md) — the representation ridges
  are extracted from.
- [Dynamical Bayesian Inference](dynamical-bayesian-inference.md) — a common
  downstream use of filtered/reconstructed phase signals.
