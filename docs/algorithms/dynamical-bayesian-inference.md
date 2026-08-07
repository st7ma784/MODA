# Dynamical Bayesian Inference

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what the algorithm computes
    and the relevant source files.

## What it computes

Given two signals believed to be coupled oscillators (e.g. cardiac and respiratory
rhythms), dynamical Bayesian inference estimates the **coupling functions** describing
how each oscillator's phase dynamics depend on the other's — including the *direction*
and *strength* of coupling, and how these change over time. Unlike coherence (which
only says *whether* two signals are synchronized), this method estimates the actual
functional form of the interaction.

## Source files

- `allguis/guis/bayesian/Functions/bayes_main.m` — main recursive Bayesian filtering
  loop (propagates a probability distribution over model parameters through time;
  inherently sequential, not a vectorization target).
- `allguis/guis/bayesian/Functions/CFprint.m` — coupling function evaluation on a grid,
  from inferred Fourier coefficients.
- `allguis/guis/bayesian/Functions/dirc.m` — extracts directionality/coupling-strength
  indices from the inferred parameters.
- `allguis/guis/bayesian/Bayesian.m` — the app module.

`CFprint.m` and `dirc.m`, along with `bayes_main.m`'s diagonal-matrix-construction
steps, were vectorized (matrix ops in place of nested loops) without changing the
underlying (necessarily sequential) recursive filter — see
[Refactor Notes](../developer-guide/refactor-notes.md).

## Key parameters

- **Frequency bands** for each signal — the oscillation of interest is first isolated
  via [ridge extraction & filtering](ridge-extraction-filtering.md) into a band around
  its dominant frequency.
- **Window length** — the time window over which coupling parameters are estimated
  before propagating to the next window.
- **Number of surrogates** — for significance testing of the inferred coupling (see
  [Surrogate Testing](surrogate-testing.md)).

## Related worked example

See [Maths Primer → Worked Example: Heartbeat Profiling](../maths-primer/worked-example-heartbeat.md)
and [Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md)
for the conceptual background.
