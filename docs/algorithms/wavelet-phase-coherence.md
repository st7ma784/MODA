# Wavelet Phase Coherence

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what the algorithm computes
    and the relevant source files.

## What it computes

Given two signals, wavelet phase coherence measures how consistently the *phase
difference* between them is maintained over time, at each frequency — i.e. whether two
oscillators are synchronized, and at which frequencies/time intervals. A coherence
value near 1 means a stable phase relationship; a value near 0 means the phases drift
independently.

This differs from simple correlation: two signals can be strongly phase-coherent at
one frequency band while being essentially uncorrelated overall.

## Source files

- `allguis/guis/coherence/Functions/wphcoh.m` — the core coherence averaging routine.
- `allguis/guis/coherence/Functions/tlphcoh.m` — time-localized phase coherence.
- `allguis/guis/coherence/CoherenceMulti.m` — the app module, supporting up to several
  simultaneous signal pairs.

Both signals are first passed through the [wavelet transform](time-frequency-analysis.md)
to obtain their instantaneous phase at each frequency and time point; `wphcoh.m` then
averages the phase difference's complex exponential over a moving time window.

## Key parameters

- **Window length** — how many cycles/how much time to average phase difference over.
- **Frequency range** — inherited from the underlying wavelet transform.

## Statistical significance

Coherence values are compared against a surrogate distribution (see
[Surrogate Testing](surrogate-testing.md)) to determine whether an observed coherence
level is significant, rather than arising by chance from two independent oscillators.

## Related worked example

See [Maths Primer → Worked Example: Heartbeat Profiling](../maths-primer/worked-example-heartbeat.md)
for coherence applied to a cardiorespiratory-style signal pair.
