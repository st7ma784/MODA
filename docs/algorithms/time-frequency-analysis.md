# Time-Frequency Analysis

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

## The mathematics

### The continuous wavelet transform

For a signal $x(t)$ and a mother wavelet $\psi$, the wavelet transform at scale $s$ and
time $t$ is the correlation of the signal against a scaled, shifted copy of $\psi$:

$$
W_x(s,t) \;=\; \int_{-\infty}^{\infty} x(u)\, \frac{1}{s}\,
\psi^{*}\!\left(\frac{u-t}{s}\right) du
$$

MODA never evaluates this integral directly. Because correlation is multiplication in
the frequency domain, `wt.m` computes it via the FFT:

$$
W_x(s,t) \;=\; \frac{1}{2\pi}\int_{0}^{\infty}
\hat{x}(\xi)\, \hat{\psi}^{*}(s\xi)\, e^{i\xi t}\, d\xi
$$

where $\hat{x}$ and $\hat{\psi}$ are the Fourier transforms of the signal and the
wavelet. This is the whole computational structure of the code: take the signal's FFT
once, multiply it by the conjugated wavelet spectrum $\hat{\psi}^{*}$ sampled at each
scale, and inverse-transform each row.

Scales map to frequencies through the wavelet's peak frequency $\omega_p$, so results
are reported directly in Hz as $f = \omega_p / (2\pi s)$. The frequency axis is
**logarithmic**, with successive frequencies separated by a fixed number of *voices*
per octave $n_v$:

$$
f_{k+1} = f_k \cdot 2^{1/n_v}
$$

This is why a CWT resolves low frequencies finely and high frequencies coarsely — the
grid itself is geometric, not linear.

### The wavelet families

Each wavelet is defined by its Fourier-domain form $\hat{\psi}(\xi)$, controlled by the
resolution parameter $f_0$. These are the exact definitions used in `wt.m`:

| Wavelet | $\hat{\psi}(\xi)$ | Parameter |
|---|---|---|
| **Lognormal** (default) | $\exp\!\left(-\tfrac{q^2}{2}\ln^2 \xi\right)$ | $q = 2\pi f_0$ |
| **Morlet** | $\exp\!\left(-\tfrac{1}{2}(\omega_0-\xi)^2\right) - \exp\!\left(-\tfrac{1}{2}(\omega_0^2+\xi^2)\right)$ | $\omega_0 = 2\pi f_0$ |
| **Bump** | $\exp\!\left(1 - \left\|\tfrac{1}{1-q^2(1-\xi)^2}\right\|\right)$ on $\|\xi-1\|<1/q$ | $q = 2.5 f_0$ |
| **Morse** | $\exp\!\left(-\xi^{a} + q\ln\xi + \tfrac{q}{a}\ln\tfrac{ea}{q}\right)$ | $q = 30 f_0^2/a$, $a=3$ |

The Morlet's second term is the admissibility correction — it subtracts the wavelet's
non-zero mean so that $\hat{\psi}(0) = 0$, which a strict wavelet requires. It is
numerically negligible once $f_0 \gtrsim 1$, which is why the time-domain form `twf` is
only supplied in that regime.

!!! note "Resolution is a genuine tradeoff, not a quality setting"
    $f_0$ trades time resolution against frequency resolution, and the product of the
    two is bounded below by the uncertainty principle. Raising $f_0$ separates
    close-together frequencies but smears events in time; lowering it does the
    opposite. There is no setting that improves both.

### The windowed Fourier transform

The WFT applies the same machinery with a window $g$ that does **not** scale with
frequency:

$$
G_x(\omega,t) \;=\; \int_{-\infty}^{\infty} x(u)\, g(u-t)\, e^{-i\omega u}\, du
$$

Because the window width is fixed, time and frequency resolution are constant across
the whole plane, and the natural frequency grid is linear rather than logarithmic. Use
the WFT when the components of interest sit in a narrow frequency range; use the CWT
when they span octaves.

### The cone of influence

Both transforms convolve the signal with a kernel of finite width, so near the start
and end of the recording the kernel overhangs the data. Those coefficients are computed
from padding rather than signal and are unreliable. The affected region widens as
frequency falls — at low frequencies the wavelet is long, so a proportionally larger
fraction of the record is contaminated. The **Cut Edges** option masks this region out;
`RelTol` (default $0.01$) sets the tolerance defining it.

## Source files

- `allguis/guis/tfa/Functions/wt.m` — CWT core transform.
- `allguis/guis/tfa/Functions/wft.m` — WFT/STFT core transform.
- `allguis/guis/tfa/TimeFrequencyAnalysis.m` — the app module wrapping these.

Both share one structure: build a kernel (wavelet or window) at each requested
scale/frequency, multiply against the signal's FFT, inverse-transform, and trim to the
signal's original length. See [Refactor Notes](../developer-guide/refactor-notes.md)
for the status of vectorizing this loop.

The implementation derives from Iatsenko, Stefanovska & McClintock's work on
time-frequency representations; see [Citations](../reference/citations.md).

## Key parameters

- **Wavelet/window type** — built-in choices for `wt.m` (Lognorm, Morlet, Bump, Morse)
  and `wft.m` (Gaussian, Hann, Blackman, Exp, Rect, Kaiser), or a custom function
  handle.
- **Frequency range (`fmin`/`fmax`)** and **number of voices/frequencies** — resolution
  of the output in frequency.
- **Resolution parameter (`f0`)** — the time/frequency resolution tradeoff above.
- **Cut edges** — whether to mask out the cone-of-influence near the signal's start/end
  where results are unreliable.

## In the web app

The [Time-Frequency Analysis page](../using-moda/web-app.md#time-frequency-analysis)
exposes CWT, WFT and STFT with these same parameters, and adds a **MODA-faithful
(legacy)** toggle — see
[Algorithmic Differences](../validation/algorithmic-differences.md) for what that
changes.

## Worked example

A [heartbeat-profiling worked example](../maths-primer/worked-example-heartbeat.md) is
planned, walking a synthetic ECG-like signal through this transform end-to-end.

## Downstream uses

Every other algorithm in MODA builds on this transform: ridge extraction runs on its
output, coherence compares two signals' transforms, and the bispectrum evaluates it at
arbitrary (non-grid) frequencies via `wtAtf2.m`/`wtAtfMod.m`.
