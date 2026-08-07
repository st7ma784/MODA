# Wavelets

!!! info "Stub — outline of planned content"

Builds on [Fourier & Convolution](fourier-and-convolution.md). Directly underpins
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md).

## Planned content

### 1. The time-frequency resolution trade-off

- A plain Fourier transform localizes frequency perfectly but time not at all; a very
  short time-domain snapshot localizes time perfectly but frequency not at all
  (Heisenberg-style uncertainty trade-off, explained intuitively without requiring
  quantum mechanics background).
- The Windowed Fourier Transform (WFT/STFT) fixes a single window width for all
  frequencies — a compromise, but not the best possible one for signals with both
  slow and fast oscillations.

### 2. What a wavelet is

- A short, oscillating "wavelet" function that's dilated (stretched/squeezed) and
  translated (shifted in time) to analyze a signal at different scales.
- Why dilating instead of using a fixed window gives *scale-adaptive* resolution:
  narrow (good time resolution) at high frequencies, wide (good frequency resolution)
  at low frequencies — matching how physiological/physical oscillations typically
  behave (fast components are usually shorter-lived; slow components persist longer).

### 3. Common mother wavelets used in MODA

- Morlet — a Gaussian-windowed complex sinusoid, a common general-purpose choice.
- Lognorm, Bump, Morse — alternative shapes with different time/frequency trade-offs;
  brief comparison of when each is preferred.
- Link to [Time-Frequency Analysis → Key parameters](../algorithms/time-frequency-analysis.md#key-parameters)
  for how to actually select these in MODA.

### 4. From wavelet transform to scalogram

- Reading a scalogram (the wavelet equivalent of a spectrogram): time on one axis,
  frequency/scale on the other, amplitude as colour.
- The cone of influence: why results near the start/end of a signal are unreliable
  (the wavelet "runs off the edge" of the available data).

### 5. Ridges

- A ridge as the locus of local maxima in a scalogram — forward pointer to
  [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md).

## Next

[Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md) — the matrix maths
behind MODA's Bayesian inference module.
