# Time-Frequency Analysis

!!! info "Stub"
    Full mathematical writeup planned. This page outlines what the algorithm computes,
    the relevant source files, and its parameters, so it's usable as a reference now.

## What it computes

Given a 1-D signal, Time-Frequency Analysis produces a 2-D representation showing how
the signal's frequency content changes over time — as opposed to a plain Fourier
transform, which only gives one frequency spectrum for the whole signal. MODA offers
two methods:

- **Continuous Wavelet Transform (CWT)** — `wt.m`. Uses dilated/translated copies of a
  mother wavelet. Good time resolution at high frequencies, good frequency resolution
  at low frequencies (matches how oscillatory biological/physical signals typically
  behave). See [Wavelets](../maths-primer/wavelets.md) for the underlying maths.
- **Windowed Fourier Transform (WFT / STFT)** — `wft.m`. Uses a fixed-width sliding
  window and a standard Fourier transform within it. Uniform time/frequency resolution
  across all frequencies.

## Source files

- `allguis/guis/tfa/Functions/wt.m` — CWT core transform.
- `allguis/guis/tfa/Functions/wft.m` — WFT/STFT core transform.
- `allguis/guis/tfa/TimeFrequencyAnalysis.m` — the app module wrapping these.

Both share one structure: build a kernel (wavelet or window) at each requested
scale/frequency, multiply against the signal's FFT, inverse-transform, and trim to the
signal's original length. See [Refactor Notes](../developer-guide/refactor-notes.md)
for the status of vectorizing this loop.

## Key parameters

- **Wavelet/window type** — built-in choices for `wt.m` (Lognorm, Morlet, Bump, Morse)
  and `wft.m` (Gaussian, Hann, Blackman, Exp, Rect, Kaiser), or a custom function
  handle.
- **Frequency range (`fmin`/`fmax`)** and **number of voices/frequencies** — resolution
  of the output in frequency.
- **Cut edges** — whether to mask out the cone-of-influence near the signal's start/end
  where results are unreliable.

## Worked example

A [heartbeat-profiling worked example](../maths-primer/worked-example-heartbeat.md) is
planned, walking a synthetic ECG-like signal through this transform end-to-end.

## Downstream uses

Every other algorithm in MODA builds on this transform: ridge extraction runs on its
output, coherence compares two signals' transforms, and the bispectrum evaluates it at
arbitrary (non-grid) frequencies via `wtAtf2.m`/`wtAtfMod.m`.
