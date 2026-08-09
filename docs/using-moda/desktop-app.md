# The Desktop App

MODA's MATLAB desktop app is a single windowed application (`MODAApp`) launched by
`MODA.m`. It presents one tab per module, each built lazily the first time you select
it — so startup is fast and a module you never open costs nothing.

```matlab
>> MODA
```

## The tabs

The tab order follows a workflow rather than an alphabet: prepare the signal, look for
structural changes, then analyse.

| Tab | Module | What it does |
|---|---|---|
| **Preprocessing** | — | Trim, detrend and resample before analysis |
| **Changepoints** | — | Detect times where spectral content shifts |
| **Time-Frequency Analysis** | `TimeFrequencyAnalysis.m` | [CWT / WFT](../algorithms/time-frequency-analysis.md) |
| **Coherence** | `CoherenceMulti.m` | [Wavelet phase coherence](../algorithms/wavelet-phase-coherence.md) |
| **Filtering** | `Filtering.m` | [Ridge extraction & filtering](../algorithms/ridge-extraction-filtering.md) |
| **Bispectral** | `Bispectrum.m` | [Wavelet bispectrum](../algorithms/wavelet-bispectrum.md) |
| **Bayesian** | `Bayesian.m` | [Dynamical Bayesian inference](../algorithms/dynamical-bayesian-inference.md) |

Each analysis tab follows the same three-region layout: **imported data** plotted along
the top, **results panels** filling the centre, and a **control panel** down the right
holding the data-import, plot-type and parameter fields. A status bar runs along the
bottom, and an Exit button sits top-right beside the MODA branding.

## Loading data

Each module's control panel has a "Load time series" action. Supported inputs:

- A single `.mat` or `.csv` file containing a 2-D array (see
  [Quickstart](../getting-started/quickstart.md#importing-time-series) for the
  row-wise/column-wise convention).
- Multiple files selected at once — one signal per file, stacked as rows. This is
  useful for modules that need several separately-recorded signals (e.g. Wavelet
  Bispectrum, Dynamical Bayesian Inference) rather than one file containing several
  channels.

`MODAread.m` handles the import and decides orientation: the **longer** dimension is
taken as time, so an $N \times 2$ and a $2 \times N$ file both load as two signals. A
recording with fewer samples than channels would be transposed incorrectly — rare in
practice, but worth knowing.

Sampling frequency is entered by hand and is not inferred from the file. Every
frequency axis in every module depends on it.

## Working with results

- **Plot Type / Calc Type** selectors switch between amplitude and power, and between
  the transform variants, without recomputing where possible.
- **Linked axes** — zooming or panning one plot moves the others with it, so a feature
  in the time series stays aligned with the same instant in the time-frequency plane.
- **Graph limits** (`Xlim`, `Ylim`) restrict the plotted range; the *Length* field
  reports the loaded record's duration.
- **Export / Open View** — a unified menu present on every module, replacing the older
  per-plot "Save X Plot" items. Open View detaches a plot into its own figure window,
  where the full MATLAB figure toolbar (data cursor, axis editing, export) is
  available.

## Per-module notes

- **Filtering** presents its results across Time-Frequency, Bands and Fourier tabs — the
  extracted ridge, the reconstructed band-limited components, and their spectra.
- **Bayesian** shows the inferred coupling-function surfaces alongside the phase time
  series; interpret these together with the directionality index rather than in
  isolation.
- **Coherence** supports several simultaneous signal pairs, and is normally run
  alongside [surrogate testing](../algorithms/surrogate-testing.md) — a raw coherence
  value is biased upward and means little on its own.

## Screenshots

!!! note "Screenshots pending for the current UI"
    The images under `docs/images/` were captured from MODA **v1.01**, which used a
    separate window per module rather than the current single tabbed shell. They are
    left in place for reference but are not shown here, because the layout they depict
    no longer matches the app. Regenerating them needs a MATLAB session with a display
    attached.

## Related pages

- [Quickstart](../getting-started/quickstart.md) — general loading/truncating/saving
  workflow.
- [The Web App](web-app.md) — the same algorithms in a browser, with screenshots.
- [Algorithms](../algorithms/time-frequency-analysis.md) — what each module actually
  computes.
