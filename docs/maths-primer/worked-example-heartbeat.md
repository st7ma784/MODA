# Worked Example: Heartbeat Profiling

!!! info "Stub — outline of planned content"

A single running example applying every earlier page in the [Maths Primer](index.md)
to one realistic use case, end to end. Full walkthrough (with figures, and a runnable
MATLAB/Python script generating the example signal) is planned as a follow-up.

## Planned scenario

A synthetic signal resembling a cardiac/respiratory recording — e.g. a simulated pulse
waveform with a slowly time-varying heart rate, amplitude modulated by breathing, plus
measurement noise. Chosen because it's intuitive (everyone has felt their own heart
rate change with breathing or activity) and exercises every algorithm in the toolbox.

## Planned walkthrough structure

1. **The raw signal** — what it looks like in the time domain, and why eyeballing it
   doesn't reveal the heart rate's time-variation clearly.
2. **Time-Frequency Analysis** — apply the [wavelet transform](../algorithms/time-frequency-analysis.md)
   to reveal the dominant frequency (heart rate) as a function of time; connect the
   scalogram back to the [Wavelets](wavelets.md) page's explanation of time/frequency
   resolution trade-offs.
3. **Ridge Extraction** — extract the instantaneous heart-rate curve from the
   scalogram (see [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md)),
   and filter out the cardiac component from a combined cardiac+respiratory synthetic
   signal.
4. **Wavelet Phase Coherence** — pair the extracted cardiac oscillation with a
   simulated respiratory signal and show coherence appearing/disappearing as their
   frequencies drift in and out of a locked ratio (cardiorespiratory coupling is a
   well-studied real phenomenon this mirrors).
5. **Dynamical Bayesian Inference** — infer the coupling function between the two
   oscillators, and interpret the resulting coupling strength/direction using the
   [Bayesian inference](probability-and-bayesian-inference.md) framing from earlier in
   the primer.
6. **Significance** — run [surrogate testing](../algorithms/surrogate-testing.md) on
   the coherence/coupling results to confirm they're not artifacts of the synthetic
   noise.
7. **Eigenvalues in context** — a short callback to
   [Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md), pointing out
   where the Bayesian filter's covariance matrix and its eigenstructure appeared in
   step 5, now that the reader has seen the whole pipeline once.

## Related pages

This example deliberately touches every algorithm page under
[Algorithms](../algorithms/time-frequency-analysis.md) — see those pages for the
formal parameter reference for each step above.
