# fastmoda

Python ports of the analyses in [MODA](https://github.com/luphysics/MODA) (Multiscale
Oscillatory Dynamics Analysis), the MATLAB toolbox from Lancaster University's Nonlinear
and Biomedical Physics group. Each port is checked against output from MODA itself:

- wavelet transform
- windowed Fourier transform
- wavelet phase coherence
- ridge extraction and reconstruction
- wavelet bispectrum and biphase
- Butterworth band-pass filtering
- dynamical Bayesian inference of phase coupling

The package also has fast spectrogram, changepoint and surrogate tools, and the
feature extraction behind the FastMODA web app's condition classifiers.

This package is the library only. The FastMODA web service (Flask, REST API, browser UI)
lives in the [repository](https://github.com/st7ma784/MODA/tree/main/FastMODA) and runs
with Docker.

## Install

```bash
pip install fastmoda
```

The core depends only on NumPy, SciPy and ruptures. Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `fastmoda[ml]` | scikit-learn, joblib | `fastmoda.condition_models` (classifiers) |
| `fastmoda[mat73]` | h5py | reading MATLAB v7.3 `.mat` files in `load_signal` |
| `fastmoda[gpu]` | torch | the `*_gpu` modules on a GPU (they fall back to NumPy without it) |
| `fastmoda[audio]` | sounddevice | `fastmoda.audio_capture` |
| `fastmoda[server]` | Flask, Plotly, gunicorn, … | running the web service from a checkout |

For CUDA, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
first, then `pip install fastmoda[gpu]`.

## Using it

Arrays are NumPy. Transforms have shape `(n_freq, n_samples)`, with frequencies as rows.
`fs` is the sampling rate in Hz.

```python
import numpy as np
from fastmoda.legacy_moda import wt_legacy

fs = 40.0
t = np.arange(0, 120, 1 / fs)
x = np.cos(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(t.size)

# MODA's wt.m. f0 is MODA's frequency-resolution parameter and has no default.
WT, freq = wt_legacy(x, fs, fmin=0.5, fmax=5, f0=1.0)
power = np.nanmean(np.abs(WT) ** 2, axis=1)       # time-averaged, NaN outside the cone of influence
```

| You want | Call | MODA original |
|---|---|---|
| Wavelet transform | `legacy_moda.wt_legacy(x, fs, fmin, fmax, wavelet="Lognorm", f0=…)` | `wt.m` |
| Windowed Fourier transform | `legacy_moda.wft_legacy(x, fs, fmin, fmax, window="Gaussian", f0=…)` | `wft.m` |
| Phase coherence | `legacy_moda.wphcoh_legacy(W1, W2)` on two `wt_legacy(..., cut_edges=False)` transforms | `wphcoh.m` |
| Time-localised coherence | `ridge_gpu.time_localized_coherence(W1, W2, freq, fs, numcycles=10)` | `tlphcoh.m` |
| Ridge + reconstruction | `legacy_ridge.extract_ridge_legacy(WT, freq, fs, *legacy_ridge.wavelet_constants("Lognorm", f0))` | `ecurve.m`, `rectfr.m` |
| Wavelet bispectrum | `legacy_moda.bispec_wav_legacy(x1, x2, fs, fmin=…, fmax=…, f0=…)` | `bispecWavNew.m` |
| Biamplitude and biphase | `legacy_moda.biphase_wav_legacy(x1, x2, fs, f1, f2, fmin=…, fmax=…, f0=…)` | `biphaseWavNew.m` |
| Band-pass filter | `legacy_bayes.loop_butter_bayes(x, (flo, fhi), fs)` | `loop_butter.m` |
| Bayesian coupling | `legacy_bayes.full_bayesian(x1, x2, band1, band2, fs, win, pr, ovr, bn, ns, signif)` | `full_bayesian.m` |
| Changepoints | `changepoint.changepoints_at_frequency(x, f, fs=fs)`, `changepoints_logbinned_power(x, fs=fs)` | — |
| Surrogates | `surrogates.iaaft_surrogate(x)`, `phase_randomization_surrogate(x)`, `legacy_bayes.surrogate_cpp(phase, n)` | `surrcalc.m` |
| ML feature vector | `pipeline.compute_feature_vector(x, fs)` | — |

More examples:

```python
from fastmoda.legacy_moda import wt_legacy, wphcoh_legacy
from fastmoda.legacy_ridge import extract_ridge_legacy, wavelet_constants
from fastmoda.legacy_bayes import full_bayesian

# Phase coherence between two signals x1 and x2
W1, freq = wt_legacy(x1, fs, fmin=0.5, fmax=5, f0=1.0, cut_edges=False)
W2, _    = wt_legacy(x2, fs, fmin=0.5, fmax=5, f0=1.0, cut_edges=False)
phcoh, phdiff = wphcoh_legacy(W1, W2)

# Instantaneous frequency, amplitude and phase of the dominant component
ridge = extract_ridge_legacy(WT, freq, fs, *wavelet_constants("Lognorm", 1.0))
ridge["ifreq"], ridge["iamp"], ridge["iphi"], ridge["recon"]

# Bayesian coupling between the 0.8–1.5 Hz phase of x1 and the 0.1–0.4 Hz phase of x2:
# 40 s windows, propagation 0.2, overlap 1, Fourier order 2, 19 surrogates at 95 %
fb = full_bayesian(x1, x2, (0.8, 1.5), (0.1, 0.4), fs, 40, 0.2, 1, 2, 19, 95)
fb["tm"], fb["cpl1"], fb["cpl2"], fb["surr_cpl1"]
```

Every function has a docstring with its parameters and return values; `help()` shows them.

## How close to MODA?

Against output from MODA running in MATLAB:

Maximum relative error against output from MODA running in MATLAB:

| Port | Agrees to |
|---|---|
| Wavelet transform | 2.3e-9 |
| Windowed Fourier transform | 1e-14 |
| Phase coherence (time-averaged and time-localised) | 1e-15 |
| Ridge | 1e-13 |
| Bispectrum | 1e-12 |
| Biphase | 4e-15 |
| Butterworth design | 6e-14 |
| Bayesian inference, given the same phases | 2e-8 |

The Bayesian filter stage is the exception. MODA raises the filter order until the output
blows up, then steps back one, which leaves it on the edge of instability. There a one-ulp
change in one coefficient moves MATLAB's own output by 3e-3 to 3e-2. On one reference
band the port settles one order higher than MATLAB does.

The faster functions outside `legacy_*` (`sliding_fft`, `analysis_gpu.cwt_gpu`,
`filtering.wft`) are re-implementations and do *not* match MODA. The
[validation pages](https://st7ma784.github.io/MODA/validation/algorithmic-differences/)
list every difference and every number.

## Documentation

- [Python library guide](https://st7ma784.github.io/MODA/api-and-ml/python-library/)
- [Algorithms](https://st7ma784.github.io/MODA/algorithms/time-frequency-analysis/)
- [Validation against MODA](https://st7ma784.github.io/MODA/validation/changelog-vs-moda/)
- [Citations](https://st7ma784.github.io/MODA/reference/citations/)

## Citing

If you use this in research, cite MODA and the papers behind the method you used (see
[Citations](https://st7ma784.github.io/MODA/reference/citations/)), starting with:

J Newman, G Lancaster and A Stefanovska, "Multiscale Oscillatory Dynamics Analysis",
v1.01, User Manual, 2018.

## Licence

GPL-3.0-only, the same as MODA.
