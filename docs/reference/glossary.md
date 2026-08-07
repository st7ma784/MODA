# Glossary

!!! info "Stub"
    This glossary is a work in progress. Terms below link to the page where they're
    used in context; a fuller standalone definition is planned for each.

| Term | Short definition | See also |
|---|---|---|
| **Time-series** | A sequence of values recorded at (usually regular) time intervals, with a known sampling frequency. | [Quickstart](../getting-started/quickstart.md) |
| **Sampling frequency (fs)** | The rate, in Hz, at which a signal was recorded. | [Quickstart](../getting-started/quickstart.md) |
| **Wavelet transform (WT)** | A time-frequency decomposition using dilated/translated copies of a mother wavelet, giving good time resolution at high frequencies and good frequency resolution at low frequencies. | [Time-Frequency Analysis](../algorithms/time-frequency-analysis.md), [Wavelets](../maths-primer/wavelets.md) |
| **Windowed Fourier transform (WFT / STFT)** | A time-frequency decomposition using a fixed-width sliding window, giving uniform time/frequency resolution across all frequencies. | [Time-Frequency Analysis](../algorithms/time-frequency-analysis.md) |
| **Ridge** | The curve of (time, frequency) points along which a time-frequency representation's amplitude is locally maximal — tracks a signal's dominant instantaneous frequency over time. | [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md) |
| **Phase coherence** | A measure of how consistently the phase difference between two oscillating signals is maintained over time, at a given frequency. | [Wavelet Phase Coherence](../algorithms/wavelet-phase-coherence.md) |
| **Bispectrum** | A higher-order spectral measure that detects quadratic phase coupling — whether two frequency components combine to produce energy at their sum frequency. | [Wavelet Bispectrum](../algorithms/wavelet-bispectrum.md) |
| **Surrogate data** | Randomized versions of a signal that preserve some statistical properties (e.g. power spectrum) but destroy others (e.g. phase relationships), used to test whether a detected feature is statistically significant. | [Surrogate Testing](../algorithms/surrogate-testing.md) |
| **Coupling function** | A function describing how the dynamics of one oscillator influence another — the object inferred by dynamical Bayesian inference. | [Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) |
| **Eigenvalue / eigenvector** | For a matrix A, a vector v (eigenvector) that A only scales (by its eigenvalue λ) rather than rotates: Av = λv. | [Linear Algebra & Eigenvalues](../maths-primer/linear-algebra-and-eigenvalues.md) |
| **Cone of influence (COI)** | The region of a time-frequency plot near the start/end of a signal where edge effects make results unreliable. | [Quickstart](../getting-started/quickstart.md#truncating-signals) |
