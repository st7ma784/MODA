# The Python Library

The analyses behind FastMODA come as an ordinary Python package, `fastmoda`, so you
can use them in scripts and notebooks without the web service. The `legacy_*` modules
are ports of MODA's MATLAB code, checked against MODA's own output (see
[how close](#how-close-to-moda)). The other modules are FastMODA's faster
re-implementations.

!!! tip "Example notebook"
    [`FastMODA/examples/fastmoda_pipeline.ipynb`](https://github.com/st7ma784/MODA/blob/main/FastMODA/examples/fastmoda_pipeline.ipynb)
    installs the package from PyPI and runs every analysis below on MODA's example heart
    and respiration recordings, with figures. It takes about a minute.
    [Open it in Colab](https://colab.research.google.com/github/st7ma784/MODA/blob/main/FastMODA/examples/fastmoda_pipeline.ipynb).

## Installing

=== "From PyPI"

    ```bash
    pip install fastmoda
    ```

=== "From GitHub"

    ```bash
    pip install "git+https://github.com/st7ma784/MODA.git#subdirectory=FastMODA"
    ```

=== "From a checkout"

    ```bash
    bash FastMODA/scripts/build_package.sh      # builds FastMODA/dist/ and tests the wheel
    pip install FastMODA/dist/fastmoda-*-py3-none-any.whl
    # or, to edit the sources in place:
    pip install -e FastMODA
    ```

Python 3.10 or newer. The core needs only NumPy, SciPy and ruptures. Add extras in
brackets, for example `pip install "fastmoda[ml,mat73]"`:

| Extra | Installs | For |
|---|---|---|
| `ml` | scikit-learn, joblib | `condition_models`: the condition classifiers |
| `mat73` | h5py | `load_signal` on MATLAB v7.3 `.mat` files |
| `gpu` | torch | running the `*_gpu` modules on a GPU. Without it they run on NumPy |
| `audio` | sounddevice | `audio_capture`: live input |
| `server` | Flask, Plotly, gunicorn, redis, … | the web service: `LocalServer` and `fastmoda-server` |
| `test` | pytest | the test suite |

For CUDA, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
before `fastmoda[gpu]`, so pip doesn't pull in the CPU build.

Check the installation:

```python
import fastmoda
print(fastmoda.__version__)       # e.g. 2026.10.15
```

The package includes the web service (`fastmoda.web`: the Flask app, its templates and
static files). It does not include the Docker files or `scripts/` (classifier
training); for those, use a checkout.

## Running the web service

`LocalServer` runs the web service, the same app the Docker images serve, inside your
Python process. It needs the `server` extra.

```python
from fastmoda import LocalServer

server = LocalServer().start()   # background thread; returns once /health answers
server.url                       # 'http://127.0.0.1:5000'
server.open_browser()            # the browser UI
server.stop()
```

As a context manager it stops on exit. `port=0` picks a free port, which is useful in
tests and when port 5000 is already in use:

```python
import json, urllib.request

with LocalServer(port=0) as server:
    with urllib.request.urlopen(server.url + '/health') as r:
        print(json.load(r))
```

| Argument | Default | |
|---|---|---|
| `host` | `'127.0.0.1'` | `'0.0.0.0'` serves the local network |
| `port` | `5000` | `0` picks a free port; `server.port` gives the one chosen |
| `data_dir` | `MODA_DATA_DIR`, else `~/.fastmoda` | SQLite database, recordings, trained models |
| `upload_dir` | `FASTMODA_UPLOAD_DIR`, else a new temporary directory | uploaded files, deleted after an hour |
| `log_requests` | `False` | print a line per HTTP request (errors are always printed) |

The app is a module-level singleton, so every `LocalServer` in one process serves the same
app. The same server runs from a shell with `fastmoda-server --port 5000 --open`
(`--help` lists the options), and blocks until Ctrl+C. `serve_forever()` does the same from a script.

This is Werkzeug's threaded server, meant for one person on their own machine. For a
shared deployment, use `gunicorn fastmoda.web.app:app` or the Docker images.


## Conventions

- Signals are 1-D NumPy arrays. `fs` is the sampling rate in Hz.
- Transforms are complex arrays of shape `(n_freq, n_samples)`: one row per frequency,
  one column per sample. `freq` gives the row frequencies.
- **`f0` has no default** on the wavelet functions. It is MODA's frequency-resolution
  parameter: larger `f0` gives sharper frequency and blurrier time. It also sets the
  number of frequency bins. MODA's GUI starts at 1.
- With `cut_edges=True` (MODA's default for the wavelet transform), cells outside the
  cone of influence are `NaN`. Use `np.nanmean` and similar.
- To get parameters for a call, read them off the MODA GUI. The names follow MODA's
  (`fmin`, `fmax`, `f0`, `padding`, `preprocess`, `cut_edges`, `nv`). `help(function)`
  lists them all.

## Loading data

```python
from fastmoda import load_signal

x, fs = load_signal("recording.mat")                  # largest numeric array; fs if the file has one
x, fs = load_signal("recording.mat", varname="ECG")   # a specific variable
x, _  = load_signal("signal.csv"); fs = 40.0          # .npy / .csv / .txt carry no rate: fs comes back 1.0
```

The examples below use this test signal:

```python
import numpy as np
fs = 40.0
t = np.arange(0, 300, 1 / fs)
x1 = np.cos(2 * np.pi * 1.1 * t + 0.4 * np.sin(2 * np.pi * 0.2 * t)) + 0.3 * np.random.randn(t.size)
x2 = np.sin(2 * np.pi * 0.2 * t) + 0.3 * np.random.randn(t.size)
```

## Time-frequency analysis

```python
from fastmoda.legacy_moda import wt_legacy, wft_legacy

WT,  freq  = wt_legacy(x1, fs, fmin=0.05, fmax=5, f0=1.0)                     # wt.m, Lognorm
WTm, freqm = wt_legacy(x1, fs, fmin=0.05, fmax=5, f0=1.0, wavelet="Morlet")
WFT, freqf = wft_legacy(x1, fs, fmin=0.05, fmax=5, f0=1.0, window="Gaussian")  # wft.m

amplitude = np.abs(WT)
power     = np.nanmean(np.abs(WT) ** 2, axis=1)      # time-averaged, as MODA's average plot
```

| Argument | Values | MODA default |
|---|---|---|
| `wavelet` | `Lognorm`, `Morlet`, `Bump` | `Lognorm` |
| `window` (WFT) | `Gaussian`, `Hann`, `Blackman`, `Exp`, `Rect`, `Kaiser-a` | `Gaussian` |
| `padding` | `predictive`, `zero`, `symmetric`, `periodic` | `predictive` |
| `preprocess` | detrend with a cubic, then band-pass to `[fmin, fmax]` | `True` |
| `cut_edges` | `NaN` outside the cone of influence | `True` (WT), `False` (WFT) |
| `nv` | voices per octave, or `"auto"` | `"auto"` |

The frequency grid follows MODA: logarithmic, $2^{k/nv}$, for the wavelet transform,
and linear for the WFT. If you leave out `fmin`, it comes from the signal length as in
MODA.

## Phase coherence

```python
from fastmoda.legacy_moda import wt_legacy, wphcoh_legacy
from fastmoda.ridge_gpu import time_localized_coherence

W1, freq = wt_legacy(x1, fs, fmin=0.05, fmax=5, f0=1.0, cut_edges=False)
W2, _    = wt_legacy(x2, fs, fmin=0.05, fmax=5, f0=1.0, cut_edges=False)

phcoh, phdiff = wphcoh_legacy(W1, W2)                            # wphcoh.m, one value per frequency
tpc = time_localized_coherence(W1, W2, freq, fs, numcycles=10)   # tlphcoh.m, (n_freq, n_samples)
```

Pass the transforms with `cut_edges=False`. Coherence handles the edges itself, and
`NaN` would spread through its running averages.

For a significance threshold, compute the coherence of `x1` against surrogates of `x2`
and take the maximum, or MODA's percentile, per frequency:

```python
from fastmoda.surrogates import iaaft_surrogate

surr = [wphcoh_legacy(W1, wt_legacy(iaaft_surrogate(x2, seed=k), fs, fmin=0.05, fmax=5,
                                    f0=1.0, cut_edges=False)[0])[0] for k in range(19)]
threshold = np.max(surr, axis=0)
significant = phcoh > threshold
```

## Ridge extraction and reconstruction

```python
from fastmoda.legacy_ridge import extract_ridge_legacy, wavelet_constants

C, D = wavelet_constants("Lognorm", 1.0)          # the wavelet and f0 used for WT
r = extract_ridge_legacy(WT, freq, fs, C, D)      # ecurve.m, then rectfr.m 'direct'

r["ifreq"]    # instantaneous frequency (Hz), per sample
r["iamp"]     # instantaneous amplitude
r["iphi"]     # instantaneous phase (rad)
r["recon"]    # iamp * cos(iphi): the reconstructed component
r["tfsupp"]   # (3, n_samples): ridge frequency and the band around it
```

Restrict the band first if you want a particular component. For example, compute `WT`
with `fmin`/`fmax` around it, as you would in MODA. For a WFT, pass `omg` instead of
`D`. `ecurve_legacy` and `rectfr_legacy` are available separately.

## Wavelet bispectrum and biphase

```python
from fastmoda.legacy_moda import bispec_wav_legacy, biphase_wav_legacy

Bisp, freq, opt, W1, W2 = bispec_wav_legacy(x1, x2, fs, fmin=0.1, fmax=3, f0=1.0)   # bispecWavNew.m
biamp, biphase = biphase_wav_legacy(x1, x2, fs, 1.1, 0.2, fmin=0.1, fmax=3, f0=1.0) # biphaseWavNew.m
```

`Bisp[j, k]` is the bispectrum at `(freq[j], freq[k])`. Cells MODA does not compute are
`NaN`: those whose sum frequency is off the grid, and those below the diagonal when
the two signals are the same. The bispectrum costs about $\tfrac12 N_f^2$ extra transforms, as MODA's
does, so it takes seconds rather than milliseconds. Narrow `fmin`/`fmax` to speed it
up. MODA's four-way bispectrum is this function called with the pairs
`(x1, x1)`, `(x2, x2)`, `(x1, x2)` and `(x2, x1)`.

## Filtering

```python
from fastmoda.legacy_bayes import loop_butter_bayes, loop_butter_filtering, bandpass_butter

y, order = loop_butter_bayes(x1, (0.8, 1.5), fs)       # loop_butter.m as MODA runs it
y = bandpass_butter(x1, 4, 0.8, 1.5, fs)                # a fixed order, zero phase (filtfilt)
```

MODA has two `loop_butter.m` files. The one on MATLAB's path first is the Bayesian
module's, so that one runs in practice, and `loop_butter_bayes` is its port.
`loop_butter_filtering` ports the other copy. Both choose the order MODA's way: the
last order before the filter blows up. That choice is sensitive to rounding (see
[below](#how-close-to-moda)).

## Dynamical Bayesian inference

```python
from fastmoda.legacy_bayes import full_bayesian

fb = full_bayesian(x1, x2,
                   (0.8, 1.5), (0.1, 0.4),  # band of each signal's oscillation
                   fs,
                   40,     # window (s)
                   0.2,    # propagation constant
                   1,      # window step as a fraction of the window (1 = no overlap)
                   2,      # Fourier order of the coupling basis
                   19,     # surrogates (0 to skip)
                   95)     # significance (%)

fb["tm"]                       # window centres (s)
fb["cpl1"], fb["cpl2"]         # coupling strength 2→1 and 1→2 per window
fb["surr_cpl1"]                # surrogate threshold for cpl1 (likewise surr_cpl2)
fb["cf1"], fb["cf2"]           # coupling functions per window (MODA's names: q21, q12)
fb["mcf1"], fb["mcf2"]         # their time averages
fb["order1"], fb["order2"]     # filter orders loop_butter chose
```

`bayes_main`, `bayes_phs`, `dirc`, `cf_print` and `surrogate_cpp` are the separate
stages, so you can feed in phases from another source. For repeatable surrogates pass
`rng=np.random.default_rng(seed)`.

## Changepoints, spectrograms and surrogates

These have no MODA counterpart.

```python
from fastmoda import sliding_fft, detect_changepoints
from fastmoda.changepoint import changepoints_at_frequency, changepoints_logbinned_power
from fastmoda.surrogates import iaaft_surrogate, phase_randomization_surrogate

freqs, times, Sxx = sliding_fft(x1, fs=fs, win_s=10.0)            # magnitude spectrogram
cp  = changepoints_at_frequency(x1, 1.1, fs=fs, win_s=10.0)       # power at one frequency
cpb = changepoints_logbinned_power(x1, fs=fs, win_s=10.0, n_bins=12)
cp["changepoint_times"]                                           # seconds
```

## Classifier features

```python
from fastmoda.pipeline import compute_feature_vector, FEATURE_SET

vec, names = compute_feature_vector(x1, fs)    # wavelet features from wt_legacy
```

`condition_models` scores a feature vector against trained models. It needs the `ml`
extra. No models come with the package. Train them with
`scripts/train_condition_classifiers.py` from a checkout (see
[Training a Classifier](training-a-classifier.md)), then tell the library where they
are:

```python
from fastmoda.condition_models import classify
scores = classify(vec, names, model_dir="/path/to/models")
```

Alternatively, set `MODA_DATA_DIR`, which must contain a `models/` folder. `classify`
returns one probability and its top contributing features per condition. If it finds
no models, it returns an empty dict. Models trained on a different `FEATURE_SET` are
refused, because their features would mean something else.

## How close to MODA?

Maximum relative difference from MODA's output in MATLAB R2026a. The full method is in
[Changelog: Legacy vs MODA](../validation/changelog-vs-moda.md).

| Port | Agrees to |
|---|---|
| `wt_legacy` (all wavelets and paddings, cone of influence) | $2.3\times10^{-9}$ |
| `wft_legacy` (all six windows) | $10^{-14}$ |
| `wphcoh_legacy`, `time_localized_coherence` | $10^{-15}$ |
| `extract_ridge_legacy` | $10^{-13}$ |
| `bispec_wav_legacy` | $10^{-12}$ |
| `biphase_wav_legacy` | $4\times10^{-15}$ |
| `butter_matlab` (filter coefficients) | $6\times10^{-14}$ |
| `bayes_main`, from the same phases | $2\times10^{-8}$ |

The exception is `loop_butter`. MODA settles on the last filter order before the
filter goes unstable. At that order, a one-ulp change in one coefficient moves MATLAB's
own output by $3\times10^{-3}$ to $3\times10^{-2}$, and on one reference band the port
picks one order higher than MATLAB. Anything downstream, such as `full_bayesian`,
inherits this. For exact agreement, filter both sides at a fixed order with
`bandpass_butter`.

The fast functions are not MODA: `sliding_fft`, `analysis_gpu.cwt_gpu`,
`ridge_gpu.cwt_complex`, `filtering.wft` and `filtering.butterworth_bandpass`.
[Algorithmic Differences](../validation/algorithmic-differences.md) lists how each one
differs.

## Citing

Cite MODA and the paper for the method you used. They are listed under
[Citations](../reference/citations.md).
