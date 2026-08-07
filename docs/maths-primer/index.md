# Maths Primer

This section builds up the mathematics behind MODA's algorithms from GCSE/A-level
foundations up to what's needed to understand eigenvalues, Fourier analysis, wavelets,
and Bayesian inference as MODA uses them. Each page assumes only the previous pages in
this section (plus standard GCSE maths) and links forward to the
[Algorithms](../algorithms/time-frequency-analysis.md) pages that use each concept.

!!! info "Stub section"
    This is a scaffold with a clear outline per page. Full content (worked examples,
    diagrams, practice problems) is planned as a substantial follow-up writing effort.

## Suggested reading order

1. [Foundations](foundations.md) — trigonometric/periodic functions, complex numbers,
   vectors. Start here if it's been a while since A-level maths.
2. [Fourier & Convolution](fourier-and-convolution.md) — how any signal can be built
   from sine waves, and what convolution means.
3. [Wavelets](wavelets.md) — why MODA mostly doesn't just use plain sine waves.
4. [Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md) — matrices,
   eigenvalues/eigenvectors, and where they show up in MODA's Bayesian inference code.
5. [Probability & Bayesian Inference](probability-and-bayesian-inference.md) —
   priors/posteriors, significance testing, and the logic behind surrogate testing.
6. [Worked Example: Heartbeat Profiling](worked-example-heartbeat.md) — all of the
   above applied end-to-end to a synthetic physiological signal.

## Who this is for

Written for readers comfortable with GCSE maths who want to understand *why* MODA's
algorithms work, not just how to click the buttons — e.g. a student or researcher using
MODA for the first time. If you just want to run an analysis, see
[Getting Started](../getting-started/installation.md) instead; you don't need any of
this section to use the software.
