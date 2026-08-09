# Glossary

Terms as MODA uses them, with a pointer to the page that develops each in context.

## Signals and sampling

| Term | Definition | See also |
|---|---|---|
| **Time-series** | A sequence of values recorded at (usually regular) time intervals, with a known sampling frequency. | [Quickstart](../getting-started/quickstart.md) |
| **Sampling frequency ($f_s$)** | The rate, in Hz, at which a signal was recorded. Not stored in a bare array of samples — you must supply it, and every frequency axis depends on it. | [Foundations](../maths-primer/foundations.md#4-rates-of-change-and-sampling) |
| **Nyquist frequency** | $f_s/2$ — the highest frequency recoverable from a signal sampled at $f_s$. Content above it is not merely degraded but genuinely ambiguous. | [Fourier & Convolution](../maths-primer/fourier-and-convolution.md#3-sampling-and-the-nyquist-rate) |
| **Aliasing** | The misidentification of a frequency above Nyquist as a lower one, because both produce identical samples. | [Fourier & Convolution](../maths-primer/fourier-and-convolution.md#3-sampling-and-the-nyquist-rate) |
| **Detrending** | Removing a slow drift or offset before analysis, so it does not dominate the low-frequency end. | [The Web App](../using-moda/web-app.md#preprocessing) |
| **Decimation** | Reducing the sampling rate by keeping every $k$-th sample. Requires low-pass filtering first, or content above the new Nyquist folds into the band. | [The Web App](../using-moda/web-app.md#preprocessing) |

## Oscillations

| Term | Definition | See also |
|---|---|---|
| **Amplitude** | How far an oscillation swings — the $A$ in $A\sin(\omega t + \phi)$. | [Foundations](../maths-primer/foundations.md) |
| **Phase** | Where in its cycle an oscillation is at a given instant — the $\phi$ term. The quantity most of MODA's measures are built on. | [Foundations](../maths-primer/foundations.md#1-periodic-and-trigonometric-functions) |
| **Angular frequency ($\omega$)** | Frequency in radians per second, $\omega = 2\pi f$. The unit the maths and the source code use internally. | [Foundations](../maths-primer/foundations.md) |
| **Instantaneous frequency** | The rate at which phase advances at a given moment — a frequency that varies over time, rather than one number for the whole record. | [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md) |
| **Analytic signal** | The complex-valued representation of a real oscillation, carrying amplitude and phase together in one number. | [Foundations](../maths-primer/foundations.md#2-complex-numbers) |

## Transforms

| Term | Definition | See also |
|---|---|---|
| **Fourier transform** | Decomposition of a signal into the frequencies it contains, averaged over the entire record — no information about *when*. | [Fourier & Convolution](../maths-primer/fourier-and-convolution.md) |
| **FFT** | Fast Fourier Transform — an $O(N\log N)$ algorithm computing the DFT exactly. Every transform in MODA is built on it. | [Fourier & Convolution](../maths-primer/fourier-and-convolution.md#dft-and-fft) |
| **Convolution** | Sliding one function across another, multiplying and summing at each offset. Equivalent to multiplication in the frequency domain — the identity that makes the transforms fast. | [Fourier & Convolution](../maths-primer/fourier-and-convolution.md#4-convolution) |
| **Wavelet transform (WT/CWT)** | A time-frequency decomposition using dilated/translated copies of a mother wavelet, giving good time resolution at high frequencies and good frequency resolution at low frequencies. | [Time-Frequency Analysis](../algorithms/time-frequency-analysis.md), [Wavelets](../maths-primer/wavelets.md) |
| **Windowed Fourier transform (WFT / STFT)** | A time-frequency decomposition using a fixed-width sliding window, giving uniform time/frequency resolution across all frequencies. | [Time-Frequency Analysis](../algorithms/time-frequency-analysis.md) |
| **MODWT** | Maximal-overlap discrete wavelet transform — a shift-invariant dyadic decomposition into octave bands. | [The Web App](../using-moda/web-app.md#wavelet-transform-modwt) |
| **Mother wavelet** | The prototype function that is stretched and shifted to build the wavelet family. MODA offers Lognormal, Morlet, Bump and Morse. | [Wavelets](../maths-primer/wavelets.md#3-mother-wavelets-in-moda) |
| **Scalogram** | The plot of wavelet amplitude against time and frequency — the wavelet counterpart of a spectrogram. | [Wavelets](../maths-primer/wavelets.md#4-from-wavelet-transform-to-scalogram) |
| **Voices per octave ($n_v$)** | The number of frequency steps per doubling, setting the resolution of the logarithmic frequency grid. | [Time-Frequency Analysis](../algorithms/time-frequency-analysis.md) |
| **Resolution parameter ($f_0$)** | The dial trading time resolution against frequency resolution. Raising it sharpens frequency and blurs time; no setting improves both. | [Wavelets](../maths-primer/wavelets.md#1-the-time-frequency-resolution-trade-off) |
| **Cone of influence (COI)** | The region near a signal's start and end where the analysing wavelet overhangs the data, making results unreliable. Widens as frequency falls. | [Wavelets](../maths-primer/wavelets.md#the-cone-of-influence) |

## Coupling and higher-order measures

| Term | Definition | See also |
|---|---|---|
| **Ridge** | The curve of (time, frequency) points along which a time-frequency representation's amplitude is locally maximal — tracks a signal's dominant instantaneous frequency over time. | [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md) |
| **Phase coherence** | How consistently the phase difference between two signals is maintained over time at a given frequency. Measures consistency, not lag, and ignores amplitude entirely. | [Wavelet Phase Coherence](../algorithms/wavelet-phase-coherence.md) |
| **Time-localized coherence** | Coherence computed over a sliding window of fixed *cycle count*, giving coherence as a function of both time and frequency. | [Wavelet Phase Coherence](../algorithms/wavelet-phase-coherence.md#time-localized-coherence) |
| **Bispectrum** | A higher-order spectral measure detecting quadratic phase coupling — whether two frequency components combine to produce phase-locked energy at their sum frequency. | [Wavelet Bispectrum](../algorithms/wavelet-bispectrum.md) |
| **Biphase** | The phase combination $\phi_1 + \phi_2 - \phi_3$. Constant under genuine quadratic coupling; drifting otherwise. | [Wavelet Bispectrum](../algorithms/wavelet-bispectrum.md#why-the-triple-product-detects-nonlinearity) |
| **Bicoherence** | The bispectrum normalised by the biamplitude, bounded in $[0,1]$, isolating phase-locking from sheer signal power. | [Wavelet Bispectrum](../algorithms/wavelet-bispectrum.md#biamplitude-and-normalisation) |
| **Coupling function** | A function describing how one oscillator's phase dynamics depend on another's — the object inferred by dynamical Bayesian inference. | [Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) |
| **Directionality index** | A scalar in $[-1,1]$ summarising which oscillator predominantly drives the other. Coherence cannot supply this, being symmetric by construction. | [Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md#directionality) |
| **Respiratory sinus arrhythmia** | The physiological coupling whereby heart rate rises and falls with the breathing cycle — the worked example's subject. | [Worked Example](../maths-primer/worked-example-heartbeat.md) |

## Statistics

| Term | Definition | See also |
|---|---|---|
| **Surrogate data** | Randomized versions of a signal preserving some statistical properties (e.g. power spectrum) while destroying others (e.g. phase relationships), used to test significance. | [Surrogate Testing](../algorithms/surrogate-testing.md) |
| **IAAFT** | Iterative Amplitude-Adjusted Fourier Transform — surrogates preserving both the power spectrum and the amplitude distribution, for non-Gaussian signals. | [Surrogate Testing](../algorithms/surrogate-testing.md#preserving-the-amplitude-distribution) |
| **Null hypothesis** | The proposition that the effect being tested for is absent. In surrogate testing it is not assumed but constructed, by building data that satisfies it. | [Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md#4-hypothesis-testing-and-significance) |
| **p-value** | The probability of a result at least this extreme *if* the null holds. With $N$ surrogates the smallest attainable value is $1/(N+1)$. | [Surrogate Testing](../algorithms/surrogate-testing.md) |
| **Prior / posterior** | Belief about parameters before and after seeing data. In a Bayesian filter, each window's posterior becomes the next window's prior. | [Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md#2-bayes-theorem) |
| **Covariance matrix** | A description of how several variables vary together. Its eigenvectors give the directions of greatest joint variation; its eigenvalues, the spread along each. | [Linear Algebra & Eigenvalues](../maths-primer/linear-algebra-and-eigenvalues.md#3-covariance-matrices-and-their-eigenvalues) |
| **Eigenvalue / eigenvector** | For a matrix $A$, a vector $\mathbf{v}$ that $A$ only scales (by $\lambda$) rather than rotates: $A\mathbf{v} = \lambda\mathbf{v}$. | [Linear Algebra & Eigenvalues](../maths-primer/linear-algebra-and-eigenvalues.md) |
| **Propagation constant** | The parameter setting how quickly a Bayesian filter's uncertainty is allowed to grow between windows, and hence how fast inferred coupling may drift. | [Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) |
