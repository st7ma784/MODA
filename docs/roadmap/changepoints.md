# Changepoint Detection Library

A reusable changepoint library with two focused modes, implemented in **both**
FastMODA (Python) and MODA (MATLAB) so the desktop and web apps detect the same
changes.

> **Status:** library ✅ · FastMODA UI page `/changepoints` + test-harness ✅ ·
> MODA `Changepoints` tab ✅ (MATLAB not run in the build env — verify on load).

## The two modes

### 1 · Single-frequency changepoints

Track the power (or amplitude) at **one chosen frequency** over time and find
changepoints in that single series. Use it when you care about a specific
oscillation — e.g. "when does the 5 Hz component start/stop or change strength?"

The target frequency is snapped to the nearest spectrogram bin; the series is the
time course of that bin's magnitude (squared to power by default).

### 2 · Log-binned full-signal power

Split the **whole** signal power into logarithmic frequency bins (linear is also
available), forming a `time × n_bins` matrix, and find changepoints **jointly**
across all bins. A change in *any* band — a frequency shift, a band appearing or
disappearing — shows up as one changepoint. This is the "what changed anywhere in
the spectrum, and when?" view.

Log binning matches human frequency perception and the
[binned-density overlay](../validation/algorithmic-differences.md), and reuses the
same `uniform_edges` so the two features stay at parity.

## API

### FastMODA (Python) — `fastmoda.changepoint`

```python
from fastmoda.changepoint import (
    changepoints_at_frequency, changepoints_logbinned_power)

# From a raw signal (spectrogram computed internally):
r1 = changepoints_at_frequency(x, target_freq=5.0, fs=40, win_s=1.0)
r2 = changepoints_logbinned_power(x, fs=40, win_s=1.0, n_bins=12, scale="log")

# Or from a precomputed spectrogram (freqs, times, Sxx) — no recompute:
r1 = changepoints_at_frequency(freqs, 5.0, times=times, Sxx=Sxx, fs=40)

r1["changepoint_times"]   # → e.g. [20.5]  (seconds)
r2["band_power"]          # → (T × n_bins) band-energy matrix
```

Both accept `pen='auto'` (an adaptive BIC-like penalty scaled by feature
dimensionality and variability) or an explicit float. `use_power=False` uses
amplitude instead of power.

### REST — `POST /analyze_changepoints`

```
file=<signal>  fs=40  win=1.0
mode=freq|binned|both      target_freq=5   (for freq mode)
n_bins=12  scale=log|linear  pen=auto  use_power=true
```

Returns Plotly figures (`freq_plot`, `binned_plot`) with the changepoints
overlaid as dashed lines, plus the changepoint times. Exposed in the web UI at
**`/changepoints`** (sidebar → 📐 Changepoints) and in the **All-Endpoints Test
Harness**.

### MODA (MATLAB) — `allguis/codes/Universal/`

```matlab
[freqs, times, Sxx] = ...        % from wt.m / wft.m magnitude
r1 = changepointsAtFrequency(freqs, times, Sxx, 5.0);
r2 = changepointsLogBinnedPower(freqs, times, Sxx, 'NBins', 12, 'Scale', 'log');
```

The MATLAB side uses `findchangepts` (Signal Processing Toolbox) as the
PELT-equivalent backend, sharing `uniformEdges` / `binPowerOverTime` with the
binned-density feature. It is surfaced as a dedicated **Changepoints** tab
(`allguis/guis/changepoints/Changepoints.m`, registered in `MODAApp`), which
computes the wavelet spectrogram with `wt.m` and drives both modes.

## Parity note

Detection is **behavioural** parity — both implementations flag the same changes
on the same signals. The *penalty scale* differs (ruptures' `pen` vs
`findchangepts` `MinThreshold` operate on different residual scales), so the two
are not expected to share a numeric penalty value, only to segment equivalently
at their respective auto settings. This is the same Tier-3 stance as the wavelet
transforms.

## Tests

`tests/parity/test_changepoint.py` (8 tests) pins both modes with ground-truth
signals that change at a known time (an amplitude step at 5 Hz; a 3 Hz→8 Hz
switch at t=20 s), verifies steady signals produce **no** changepoints, and
checks the band-power matrix carries the energy shift into the right bins.
