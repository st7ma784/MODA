# The Desktop App

MODA's MATLAB desktop app (`MODA.m`) is a single windowed application (`MODAApp`) with
one tab per analysis module: **Time-Frequency Analysis**, **Wavelet Phase Coherence**,
**Ridge Extraction & Filtering**, **Wavelet Bispectrum**, and **Dynamical Bayesian
Inference**. Each tab loads lazily the first time it's selected.

!!! info "This page is a stub"
    Full walkthrough content — with screenshots per module — is planned. For now, the
    [Quickstart](../getting-started/quickstart.md) covers the general loading/plotting/
    saving workflow that applies to every tab; this page covers what's specific to the
    desktop shell and each module.

## Outline of planned content

- **App shell** — the `MainTabGroup`, lazy module loading, the top-right Exit button,
  and the MODA/university branding anchors (top-right and bottom-right respectively).
- **Loading data** — each module's data-loading controls in the left-hand control
  panel; single-file vs multi-file selection; CSV vs `.mat`; row-wise vs column-wise
  orientation. See [Quickstart → Importing time-series](../getting-started/quickstart.md#importing-time-series)
  for the general rules, and [`MODAread.m`](https://github.com/luphysics/MODA/blob/master/allguis/codes/Universal/MODAread.m)
  for the supported input shapes in detail.
- **Per-module control panels** — Plot Type / Calc Type selectors, parameter fields,
  and how they map onto the algorithm parameters described in
  [Algorithms](../algorithms/time-frequency-analysis.md).
- **Results panels** — reading the plots for each module (e.g. Filtering's
  Time-Frequency/Bands/Fourier result tabs, Bayesian's coupling-function and
  time-series panels), and the linked-axes zoom/pan behaviour.
- **Exporting** — the unified `Export`/`Open View` menu present on every module,
  replacing the older per-plot "Save X Plot" menu items.

## Loading data

Each module's control panel has a "Load time series" action. Supported inputs:

- A single `.mat` or `.csv` file containing a 2-D array (see
  [Quickstart](../getting-started/quickstart.md#importing-time-series) for the
  row-wise/column-wise convention).
- Multiple files selected at once — one signal per file, stacked as rows. This is
  useful for modules that need several separately-recorded signals (e.g. Wavelet
  Bispectrum, Dynamical Bayesian Inference) rather than one file containing several
  channels.

## Related pages

- [Quickstart](../getting-started/quickstart.md) — general loading/truncating/saving
  workflow.
- [Algorithms](../algorithms/time-frequency-analysis.md) — what each module actually
  computes.
