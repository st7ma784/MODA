# Changepoint Detection

## What it computes

Every other algorithm here describes *what* a signal contains. Changepoint detection
answers a different question: **when did it change?** Given a recording whose character
shifts partway through — an oscillation starting, a band collapsing, a rhythm moving
frequency — it returns the times at which that happened, without being told in advance
how many changes to expect.

Implemented in **both** FastMODA (Python) and MODA (MATLAB) so the desktop and web apps
detect the same changes.

![Changepoint detection results](../images/webapp/webapp-changepoints.png)

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

## The mathematics

### Changepoints as an optimal segmentation

Both modes reduce the signal to a feature series $y_1 \ldots y_T$ — one value per time
frame for single-frequency mode, a $d$-dimensional vector per frame for the binned mode
— and then ask for the segmentation that best explains it.

Formally, choose the number of changepoints $K$ and their positions
$\tau_1 < \cdots < \tau_K$ minimising

$$
\sum_{k=0}^{K} c\!\left(y_{\tau_k : \tau_{k+1}}\right) \;+\; \beta K
$$

The first term is the **cost** of describing each segment by a constant model; the
second charges a penalty $\beta$ per changepoint. Without that penalty the minimum is
trivially attained by putting a changepoint everywhere — every segment becomes one
point and fits perfectly. **The penalty is what makes the problem meaningful**, and it
is the parameter that matters most in practice.

The cost is the `l2` model: squared deviation from the segment mean,

$$
c\!\left(y_{a:b}\right) = \sum_{t=a}^{b} \left\| y_t - \bar{y}_{a:b} \right\|^2
$$

so a "change" means **a shift in the mean level of the features** — band power moving
up or down — rather than a change in variance or slope.

### Why PELT

Searching all possible segmentations is exponential. The implementation uses
**PELT** (Pruned Exact Linear Time) via `ruptures`, which is dynamic programming plus a
pruning rule: a candidate changepoint that can never become optimal later is discarded
permanently. The result is **exact** — not a heuristic approximation — at roughly $O(T)$
cost when changepoints are reasonably spread through the signal.

This is the same optimal-substructure idea as the DP used for
[ridge extraction](ridge-extraction-filtering.md), applied to segment boundaries rather
than a frequency path.

### Choosing the penalty

`pen='auto'` computes an adaptive, BIC-like penalty:

$$
\beta = \underbrace{d \ln T}_{\text{BIC-ish floor}} \times \left(1 + \bar{\sigma}\right),
\qquad
\bar{\sigma} = \frac{1}{d}\sum_{j=1}^{d} \operatorname{sd}\!\left(y_{\cdot j}\right)
$$

Both factors earn their place. The $d \ln T$ floor is the standard model-selection
trade-off — more channels and longer records need stronger evidence per changepoint.
The $(1 + \bar\sigma)$ term scales with how variable the standardised features actually
are, so a noisy 12-bin spectrum doesn't fragment into dozens of spurious segments the
way a fixed penalty would.

Raising `pen` yields fewer, more confident changepoints; lowering it yields more.
There is no universally correct value — it encodes how much change you consider worth
reporting.

### Standardisation and NaN handling

Features are z-scored per channel before detection:

$$
z_{tj} = \frac{y_{tj} - \bar{y}_{\cdot j}}{\operatorname{sd}(y_{\cdot j}) + \epsilon}
$$

Without this a single high-power band would dominate the `l2` cost and the other bins
would contribute nothing. Standardising puts every band on equal footing, so a change in
a weak band counts as much as one in a strong band.

Non-finite values — notably the NaNs left by
[cone-of-influence masking](../maths-primer/wavelets.md#the-cone-of-influence) — are then
replaced by 0, which after standardisation is the neutral, mean-valued fill. The MATLAB
engine's `local_sanitize` does the same, so both apps behave identically at the edges.

!!! note "Changepoints are reported in frames, then converted"
    Detection runs on the spectrogram's time axis, whose resolution is set by `win_s`.
    A 1-second window cannot localise a change to better than about a second, however
    finely the signal was sampled. Shrinking `win_s` sharpens timing but makes each
    frame's spectrum noisier — the same trade-off as everywhere else in the toolbox.

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

## Key parameters

- **Mode** — single-frequency, log-binned, or both.
- **Window (`win_s`)** — the spectrogram frame length, which sets the finest
  resolvable timing of a change.
- **Target frequency** — for single-frequency mode; snapped to the nearest bin. Keep it
  below Nyquist, or the mode has nothing to track.
- **Number of bins / scale** — for binned mode; log matches the
  [binned-density overlay](../validation/algorithmic-differences.md).
- **Penalty (`pen`)** — `auto` for the adaptive BIC-like value above, or a float.
  Higher means fewer changepoints.
- **Power vs amplitude (`use_power`)** — whether the tracked series is squared.

## Related pages

- [Time-Frequency Analysis](time-frequency-analysis.md) — supplies the spectrogram both
  modes are computed from.
- [Ridge Extraction & Filtering](ridge-extraction-filtering.md) — the other
  dynamic-programming optimisation in the toolbox, over a frequency path rather than
  segment boundaries.
- [The Web App](../using-moda/web-app.md#changepoints) — the `/changepoints` page shown
  above.
