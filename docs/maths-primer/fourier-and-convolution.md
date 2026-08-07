# Fourier & Convolution

!!! info "Stub — outline of planned content"

Builds on [Foundations](foundations.md). This is the mathematical basis for
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md).

## Planned content

### 1. The Fourier series idea

- Any periodic signal can be written as a sum of sine/cosine waves at multiples of its
  fundamental frequency.
- Intuition first (adding sine waves of increasing frequency to approximate a square
  wave), then the formal series.

### 2. The Fourier transform

- Extending the Fourier series idea to non-periodic signals: a continuum of
  frequencies instead of discrete multiples.
- Reading a spectrum: what the x-axis (frequency) and y-axis (amplitude/power) mean.
- The Discrete Fourier Transform (DFT) and FFT (Fast Fourier Transform) as the
  practical, computer-friendly version — this is what `fft()`/`ifft()` compute inside
  `wt.m`/`wft.m`.

### 3. Sampling and the Nyquist rate

- Why a signal must be sampled at more than twice its highest frequency component to
  be reconstructed without ambiguity (aliasing).
- Practical consequence for MODA: choosing a sensible `fmax` relative to the sampling
  frequency `fs`.

### 4. Convolution

- Convolution as "sliding and multiplying" one function against another.
- Why multiplying two signals' Fourier transforms together corresponds to convolving
  them in time (the convolution theorem) — this is *why* `wt.m`/`wft.m` compute the
  wavelet transform via `fft` → multiply → `ifft` rather than a direct time-domain
  convolution: it's mathematically identical but far faster to compute.

### 5. Limits of the plain Fourier transform

- A single Fourier transform gives one spectrum for an entire signal — no information
  about *when* a frequency was present.
- Motivates windowing (WFT) and wavelets — see [Wavelets](wavelets.md).

## Next

[Wavelets](wavelets.md) — trading time and frequency resolution to localize *when* a
frequency occurs.
