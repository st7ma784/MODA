# Fourier & Convolution

Builds on [Foundations](foundations.md). This is the mathematical basis for
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md).

## 1. The Fourier series idea

Start with a claim that sounds too strong: **any periodic signal, however jagged, is a
sum of sine waves.**

Take a square wave. Add a sine at its fundamental frequency $f$ — a rounded
approximation. Add a third of the amplitude at $3f$ — the corners sharpen. Add a fifth
at $5f$, a seventh at $7f$:

$$
\text{square}(t) \;\approx\; \frac{4}{\pi}\left(\sin\omega t + \frac{1}{3}\sin 3\omega t
+ \frac{1}{5}\sin 5\omega t + \cdots\right)
$$

Every term you add sharpens the edges further. In the limit you have the square wave
exactly. Formally, a periodic signal of period $T$ decomposes as

$$
x(t) = \sum_{n=-\infty}^{\infty} c_n e^{i n \omega_0 t}, \qquad \omega_0 = \frac{2\pi}{T}
$$

with each **coefficient** $c_n$ a complex number carrying that harmonic's amplitude and
phase — exactly the two-numbers-in-one from [Foundations](foundations.md).

The coefficients are recovered by a dot product of the signal against each candidate
frequency:

$$
c_n = \frac{1}{T}\int_0^T x(t)\, e^{-i n\omega_0 t}\, dt
$$

Read that integral as "how much does $x$ look like a wave at frequency $n\omega_0$?" —
the continuous version of the alignment measure from the previous page.

## 2. The Fourier transform

Real recordings are not periodic. Let the period grow without bound and the discrete
harmonics $n\omega_0$ crowd together into a continuum, giving the **Fourier transform**:

$$
\hat{x}(\omega) = \int_{-\infty}^{\infty} x(t)\, e^{-i\omega t}\, dt
$$

and its inverse, which rebuilds the signal from its spectrum:

$$
x(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{x}(\omega)\, e^{i\omega t}\, d\omega
$$

### Reading a spectrum

$\hat{x}(\omega)$ is complex. Plots almost always show:

- **Amplitude spectrum** $|\hat{x}(\omega)|$ — how much of each frequency is present.
- **Power spectrum** $|\hat{x}(\omega)|^2$ — the same thing in energy terms, which
  emphasises strong components and suppresses weak ones.

The phase $\arg\hat{x}(\omega)$ is usually not plotted — and discarding it is precisely
why a power spectrum cannot detect the phase coupling that the
[bispectrum](../algorithms/wavelet-bispectrum.md) exists to find.

### DFT and FFT

Computers hold $N$ samples, not a continuous function, so they compute the **Discrete
Fourier Transform**:

$$
X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i k n / N}
$$

Done directly this is $O(N^2)$. The **Fast Fourier Transform** is an algorithm computing
the identical result in $O(N \log N)$ by recursively exploiting symmetry. For a
1-million-sample record that is the difference between $10^{12}$ and $2\times10^7$
operations — roughly five orders of magnitude, and the reason spectral analysis is
routine rather than exotic. Every `fft()`/`ifft()` call inside `wt.m` and `wft.m` is
this algorithm.

## 3. Sampling and the Nyquist rate

Sampling at $f_s$ imposes a hard ceiling. The **Nyquist frequency** is

$$
f_{\text{Nyq}} = \frac{f_s}{2}
$$

and no frequency above it can be recovered. The reason is not a technical limitation
but an ambiguity: sampled at $f_s$, a sinusoid at $f_s/2 + \delta$ produces *exactly the
same sample values* as one at $f_s/2 - \delta$. The information distinguishing them is
not degraded, it is absent. This misidentification is **aliasing** — the effect that
makes wagon wheels appear to spin backwards in films.

Two practical consequences:

- **Choose `fmax` at or below $f_s/2$.** Requesting more does not fail loudly; it
  produces confident-looking output in a band where nothing is trustworthy.
- **Anti-alias before decimating.** Dropping every other sample halves $f_s$, so any
  content above the *new* Nyquist folds down into your band. FastMODA's
  [preprocessing page](../using-moda/web-app.md#preprocessing) handles this when
  decimating.

!!! warning "A worked trap"
    A 10 Hz recording has $f_{\text{Nyq}} = 5$ Hz. The changepoints page's
    `target_freq` field defaults to 10 Hz — above Nyquist for such a file. Fields like
    this are defaults for a *typical* recording, not for yours; check them against
    your own $f_s$.

## 4. Convolution

**Convolution** slides one function across another, multiplying and summing at every
offset:

$$
(x * g)(t) = \int_{-\infty}^{\infty} x(\tau)\, g(t - \tau)\, d\tau
$$

Every filter is a convolution: smoothing, bandpassing, and — importantly here — the
wavelet transform, which slides a wavelet along a signal and asks at each position how
well they match.

### The convolution theorem

The single most useful identity in the primer:

$$
\widehat{x * g}(\omega) = \hat{x}(\omega)\,\hat{g}(\omega)
$$

**Convolution in time is multiplication in frequency.** A sliding, overlapping
comparison at every offset becomes an element-by-element product.

This is why MODA computes transforms the way it does. A direct convolution costs
$O(N^2)$; the FFT route

$$
x * g \;=\; \mathcal{F}^{-1}\Big\{\hat{x}\cdot\hat{g}\Big\}
$$

costs $O(N\log N)$ — mathematically identical, dramatically cheaper. It is exactly the
structure of `wt.m`: **FFT the signal once**, multiply by the conjugated wavelet
spectrum at each scale, inverse-transform each row. The signal's FFT is computed once
and reused across every scale, which is also the optimisation exploited by
`wtAtf2_batch` in the [bispectrum](../algorithms/wavelet-bispectrum.md).

## 5. Limits of the plain Fourier transform

The Fourier transform integrates over **all time**. It tells you a 1.2 Hz component is
present in the recording; it cannot tell you it appeared at 40 s and vanished by 90 s.
That information is not lost — it is encoded in the phases, spread across every
frequency — but it is unreadable in the amplitude spectrum.

Consider two recordings: one where a 1 Hz and a 5 Hz tone sound together throughout,
another where 1 Hz plays for the first half and 5 Hz for the second. **Their amplitude
spectra are near-identical.** For anything whose frequency content changes over time —
which is essentially every physiological signal — that is disqualifying.

Two fixes exist. Chop the signal into short windows and transform each: the **Windowed
Fourier Transform**, which fixes one window width for all frequencies. Or let the
analysis window scale with frequency: **wavelets**. The next page explains why the
second choice suits signals spanning octaves.

## Next

[Wavelets](wavelets.md) — trading time and frequency resolution to localize *when* a
frequency occurs.
