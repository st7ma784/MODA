# Quickstart

This walks through running MODA's desktop app for the first time. The
[User Manual](https://github.com/luphysics/MODA/blob/master/User%20Manual.pdf) has a
more in-depth explanation of MODA's functionality; this page gets you to a first result
quickly.

## Running MODA

In your file explorer, double-click `MODA.m` inside the `MODA` folder to open it with
MATLAB. After the MATLAB window opens, press `F5` or click the "Run" button to start
MODA.

!!! note
    You may need to click inside the section displaying the contents of `MODA.m` for
    the "Run" button to appear.

If a dialog appears stating that `MODA.m` is not in the current MATLAB path, click
"Change Folder".

The MODA app window will then open, showing tabs for each analysis module.

## Importing time-series

In MODA, a time-series is a series of recorded values, where the sampling frequency —
the frequency at which the recordings were made — is known.

MODA can analyse multiple signals at once, provided that all signals have the same
duration and sampling frequency.

To import time-series into MODA, they must be saved in a compatible format:

- The file type must be a `.mat` file or `.csv` file (or, in the current desktop app,
  you can select multiple files at once and each file contributes one signal — see
  [Using MODA → The Desktop App](../using-moda/desktop-app.md#loading-data)).
- The file must contain a **single array, whose entries are all a single real number**.
  Each row or column of the array corresponds to a different time-series.
- For windows which inspect pairs of signals (for example, Dynamical Bayesian
  Inference), the number of time-series should be even. If there are an odd number of
  time-series, the last one is dropped and pairs are formed from the remainder.
- For Wavelet Bispectrum analysis, there must be only two time-series — the bispectrum
  computation is too costly for more.

!!! note
    The sampling frequency is entered in the user interface, rather than read from the
    file.

**If the array loaded into MODA is extremely large, it may run slowly or crash.** See
[Large arrays and downsampling](#large-arrays-and-downsampling) below.

## Example: Time-Frequency Analysis

Open the **Time-Frequency Analysis** tab, then use its data-loading controls to select
a `.csv` or `.mat` file.

!!! tip
    There are example signals in the `example_sigs` folder. Try
    `example_sigs/6signals_10Hz.mat` (a row-wise signal).

You'll be asked for the sampling frequency in Hz (for example, `10`), and then whether
the data is row-wise or column-wise.

**Row-wise data** — each row contains one signal:

```
| Signal 1, Value 1 | Signal 1, Value 2 | Signal 1, Value 3 |
| Signal 2, Value 1 | Signal 2, Value 2 | Signal 2, Value 3 |
| Signal 3, Value 1 | Signal 3, Value 2 | Signal 3, Value 3 |
| Signal 4, Value 1 | Signal 4, Value 2 | Signal 4, Value 3 |
```

**Column-wise data** — each column contains one signal:

```
| Signal 1, Value 1 | Signal 2, Value 1 | Signal 3, Value 1 |
| Signal 1, Value 2 | Signal 2, Value 2 | Signal 3, Value 2 |
| Signal 1, Value 3 | Signal 2, Value 3 | Signal 3, Value 3 |
| Signal 1, Value 4 | Signal 2, Value 4 | Signal 3, Value 4 |
```

!!! warning
    If the wrong orientation is selected, MODA may freeze — it will interpret the data
    as a very large number of very short signals.

After selecting the orientation, the data loads and the first signal is plotted. The
imported signals are listed in the "Select data" section; when a signal (or signal
pair) is selected, it — and any results already calculated for it — is plotted.

## Truncating signals

You may wish to analyse only a portion of a recorded signal. Once loaded, use the
zoom (magnifying-glass) tool and click-and-drag on the signal plot to zoom to a
rectangular region, or enter values directly into the "Xlim" field and click
"Refresh".

!!! tip
    Zooming resets the vertical (Y) axis to fit the selected horizontal range.

After truncating, existing result plots do not update until the calculation is
re-run.

!!! note
    For frequency/time-frequency results, the minimum frequency that can be reliably
    resolved increases as the signal is truncated.

## Plotting and saving

To save one of the graphs currently displayed, use the module's **Export** menu; this
opens the plot in a new window as a standalone MATLAB figure, which:

- can be saved in a variety of formats,
- has all of MATLAB's built-in figure tools available,
- won't be overwritten by a later calculation.

Recommended formats:

- Single-variable plots (e.g. a time series): a vector format such as `.svg`, `.pdf`
  or `.eps`.
- Two-variable plots (e.g. a spectrogram): a scalar/raster format such as `.png`.

### Saving the current session

Use `File → Save session` to save the current session as a `.mat` file, and
`File → Load previous session` to reload it later.

### Saving data

Use the module's export controls to save the underlying numeric values as `.mat` or
`.csv`. `.mat` files can be loaded in MATLAB or Python; `.csv` files are convenient for
spreadsheet tools.

!!! warning
    When saving as `.csv`, don't change the file extension or the "Save as type"
    dropdown manually.

Some values will be saved as `NaN` ("not a number"), meaning the value could not be
computed — in time-frequency analysis this happens near the start/end of the signal
when "cut edges" is enabled.

!!! warning
    When saving Ridge Extraction & Filtering results for later use in Dynamical
    Bayesian Inference, save as `.mat`, not `.csv`.

## Large arrays and downsampling

If an extremely large array is loaded into MODA, it could run very slowly or crash.
Arrays may be overly large because:

1. the number of signals stored is large,
2. the signals were measured over a very long time, or
3. the signals were measured with a very high sampling frequency.

For (1), split the array into multiple files with fewer signals each, and analyse each
file individually.

For (2), split the time interval into smaller intervals (preferably with a small
overlap, since time-frequency analysis cannot be performed too close to the start/end
of a signal), and analyse each piece separately.

For (3), if the sampling frequency is many times higher (e.g. ~10x) than the largest
frequency of interest in the signal, **downsample** before loading.
