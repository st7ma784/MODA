# Wavelets

Builds on [Fourier & Convolution](fourier-and-convolution.md). Directly underpins
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md).

## 1. The time-frequency resolution trade-off

To know a signal's frequency precisely you must observe many cycles — and observing
many cycles takes time, so you lose track of *when*. To know *when* something happened
you need a short observation — and a short observation cannot pin down frequency. These
pull against each other, and the tension is not an engineering shortcoming that a
cleverer algorithm might remove.

Quantitatively, for spreads $\Delta t$ in time and $\Delta f$ in frequency:

$$
\Delta t \cdot \Delta f \;\geq\; \frac{1}{4\pi}
$$

This is the same mathematics as Heisenberg's uncertainty principle — no quantum
mechanics required, just a property of the Fourier transform. **You may trade the two,
never beat their product.**

The extremes bracket the problem:

| Method | Time resolution | Frequency resolution |
|---|---|---|
| Fourier transform of the whole signal | none | best possible |
| Single sample | perfect | none |
| Windowed Fourier (WFT) | fixed, chosen by window width | fixed |
| Wavelet transform | **scales with frequency** | **scales with frequency** |

The WFT's compromise is reasonable but rigid: one window width serves every frequency.
Choose a 10-second window and you have 10 seconds' resolution both for a 0.1 Hz drift —
which completes just one cycle in that time, far too few to characterise — and for a
20 Hz oscillation, of which it contains 200 cycles, far more than needed. One setting
cannot suit both.

## 2. What a wavelet is

A **wavelet** is a short oscillating function, localized in time, with zero mean. From
one *mother wavelet* $\psi$ we generate a family by **dilating** (stretching) and
**translating** (shifting):

$$
\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}}\, \psi\!\left(\frac{t-\tau}{s}\right)
$$

Scale $s$ stretches the wavelet — large $s$ means a long, low-frequency wavelet; small
$s$ a brief, high-frequency one. The wavelet transform correlates the signal against
every member:

$$
W_x(s,\tau) = \int x(t)\, \psi^{*}_{s,\tau}(t)\, dt
$$

which is a [convolution](fourier-and-convolution.md#4-convolution), hence computed via
FFT.

### Why dilating is the key move

Because the wavelet stretches with scale, its analysis window is always a **fixed
number of cycles** rather than a fixed number of seconds. At 10 Hz, six cycles occupy
0.6 s; at 0.1 Hz, the same six cycles occupy 60 s. The resolution adapts:

- **High frequencies** — short window, sharp timing, coarse frequency.
- **Low frequencies** — long window, blurred timing, fine frequency.

This matches how real signals behave. Fast components tend to be transient (a heartbeat
lasts a fraction of a second); slow components persist (a respiratory rhythm continues
for minutes). Spending your time resolution where the events are brief, and your
frequency resolution where the oscillations are slow and sustained, is the right
allocation — and it is why the CWT's frequency axis is
[logarithmic](../algorithms/time-frequency-analysis.md#the-continuous-wavelet-transform),
each step a fixed *ratio* rather than a fixed increment.

## 3. Mother wavelets in MODA

Each is defined by its Fourier-domain shape $\hat\psi(\xi)$, tuned by the resolution
parameter $f_0$. Exact definitions live in
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md#the-wavelet-families);
the character of each:

- **Morlet** — a Gaussian-windowed complex sinusoid. The most intuitive: literally a
  sine wave faded in and out by a bell curve. Optimally concentrated under the
  uncertainty bound, and the usual general-purpose default.
- **Lognormal** — Gaussian in *log*-frequency, so it is symmetric on the logarithmic
  axis the CWT actually uses. MODA's default.
- **Bump** — strictly zero outside a finite frequency band, so it cannot leak into
  neighbouring bands; the price is slightly worse time localisation.
- **Morse** — a parameterised family whose asymmetry is adjustable, useful for
  analysing transients.

Raising $f_0$ makes the wavelet contain more cycles: sharper in frequency, blurrier in
time. It is the dial along the trade-off curve, not a quality setting — there is no
value that improves both.

## 4. From wavelet transform to scalogram

$W_x(s,\tau)$ is complex at every point. Plotting $|W|$ against time and frequency
gives a **scalogram** — the wavelet counterpart of a spectrogram. Time runs along one
axis, frequency (log-spaced) the other, amplitude as colour. A sustained oscillation
appears as a horizontal band; a drifting one as a sloping ridge; a click or artefact as
a vertical streak spanning all frequencies.

The [TFA screenshot](../using-moda/web-app.md#time-frequency-analysis) shows one: a
strong band near 1 Hz undulating slowly — a rhythm whose rate is itself being modulated.

### The cone of influence

At the record's start and end the wavelet overhangs the data and its coefficients are
computed partly from padding. Since low-frequency wavelets are long, the unreliable
region **widens as frequency falls**, carving a cone out of the corners. MODA's **Cut
Edges** option masks it.

!!! warning "Cutting edges is not free"
    Masking inserts NaN into the transform, and any downstream computation that sums
    over time will propagate those NaNs across the whole result. This is not
    hypothetical: it was a live bug in FastMODA's coherence path, where NaN-masked
    transforms turned every coherence value into NaN while the phase-difference plot —
    computed with a NaN-aware mean — still rendered. Mask at the point of *display*,
    not before an average.

## 5. Ridges

Following the brightest point of a scalogram through time traces the signal's dominant
instantaneous frequency — a **ridge**. For a heartbeat, that curve *is* the heart rate
over time.

Naively taking the loudest frequency at each instant produces a shattered curve, since
noise makes the peak hop between competing components. Real ridge extraction treats it
as an optimal-path problem penalising implausible jumps — see
[Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md).

## Next

[Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md) — the matrix maths
behind MODA's Bayesian inference module.
