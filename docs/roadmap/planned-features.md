# Planned Features — Preprocessing, Ridge Hints, Binned Spectra

Three cross-cutting features implemented in **both** MODA (MATLAB App Designer)
and FastMODA (Flask/Plotly) so the desktop and web apps stay at parity.

## Status

| Feature | FastMODA (web) | MODA engine (MATLAB) | MODA GUI wiring |
|---------|:--:|:--:|:--:|
| 1 · Preprocessing tab | ✅ | ✅ `cropAndDecimate.m` | ✅ `Preprocessing.m` + MODAApp tab |
| 2 · Ridge boundary hint | ✅ | ✅ `ridgeBoundaryHint.m` | ⏳ badge in Filtering |
| 3 · Binned density overlay | ✅ | ✅ `binnedSpectrum` / `uniformEdges` / `fitBinsToPeaksEdges` | ⏳ overlay on TFA marginal |

The MATLAB **engine functions** (`allguis/codes/Universal/`) mirror the verified
Python exactly, so both apps compute identically. ⚠️ The MATLAB code has not been
run here (no MATLAB in the build env) — verify on load. The two ⏳ items thread a
badge / overlay into the existing analysis modules and are best added once the
engine + Preprocessing module are confirmed to load.

---

## 1 · Preprocessing tab (dedicated) — clip, bulk-crop, decimate

> **Status:** FastMODA ✅ · MODA `Preprocessing.m` module + `cropAndDecimate.m` engine ✅ (verify in MATLAB).

A dedicated **Preprocessing** tab/module (not a panel bolted onto existing
modules) that prepares signals before any analysis and hands the result **live**
into the analysis modules, with an **optional save** to a new file.

### Capabilities

| Control | Behaviour |
|---------|-----------|
| **Clip single signal** | Select one loaded signal; set a start/stop and crop it. |
| **Bulk crop by timestamps** | Apply `{start_s, stop_s}` to a selected set of files at once. |
| **Crop to first / final length** | Keep the first *N* s or the final *N* s of each file. |
| **Integer downsample** | Reduce the sample rate by an **integer factor only** (fs → fs/k). |
| **Preview** | Plot one file with the slice region overlaid, so the user sees exactly where the cut lands before applying. |

### Shared crop/resample model (identical in both apps)

```
spec = {
  mode:   "range" | "first" | "final",
  start_s, stop_s,              # for "range"
  length_s,                     # for "first" / "final"
  decimate_factor: k (int ≥ 1)  # 1 = no change
}
```

- Crop → sample indices `round(t·fs)`, clamped to `[1, N]` per file.
- **Decimation is integer-only**: the UI offers target rates `fs/k` for
  `k = 1,2,3,…` (so 40 Hz → 20, 13.3, 10, …), never an arbitrary rate. Apply an
  anti-alias low-pass **before** dropping samples (`scipy.signal.decimate` /
  MATLAB `decimate`), not naïve subsampling, to avoid aliasing. The new `fs`
  (`fs/k`) travels with the cropped signal.
- Bulk = same `spec` across every selected file; per-file length/`fs` handled
  independently.

### MODA (MATLAB)

- New `Preprocessing.m` App Designer class, registered in
  `MODAApp.ensureModuleLoaded` (switch + module/tab list) like the other modules.
- UI: loaded-signal list · **preview axes** · start/stop and first/final fields ·
  a decimation dropdown listing valid `fs/k` targets · "Apply to selected" /
  "Apply to all" · "Save cropped…".
- Preview: shaded `patch` (alpha) + draggable `xline`s (or `drawrectangle` ROI)
  synced live with the numeric fields.
- Writes into the existing `sig`/`sig_cut`/`time_axis` handles (see
  `MODAread`/`setProp`) so downstream modules pick up the cropped, decimated
  signal; `decimate` updates `sampling_freq`.

### FastMODA (Python)

- New page `/preprocess` + `templates/preprocess.html`; add to the nav in
  `index_optimized.html` and the shared page headers.
- Endpoints:
  - `POST /preprocess_preview` → Plotly figure of one signal with an
    `add_vrect` slice overlay + rangeslider.
  - `POST /preprocess_apply` → crop + `scipy.signal.decimate`; returns the new
    `.npy` (download) **and** a reusable token/id the analysis pages can load.
- UI: upload file(s) · draggable Plotly range preview · start/stop, first/final,
  and integer-decimation controls · bulk apply.

---

## 2 · Ridge boundary hint (frequency edge only)

A small, non-intrusive indicator warning that a detected ridge may be pinned to
the **frequency** limits of the analysis (`fmin`/`fmax`), i.e. the true ridge
probably extends outside the analysed band. **Scope: frequency edge only** (time
/ cone-of-influence edges are out of scope for now).

### Detection (shared, cheap, post-ridge)

- Fraction of ridge samples whose frequency falls within a tolerance of `fmin`
  or `fmax` (e.g. within one voice, or 5% of the band edge).
- Combined with the ridge amplitude there (is there still strong energy at the
  edge, or is it decaying?).
- Emit a severity `none | low | high` + a short message, e.g.
  *"Ridge sits within 5% of fmax for 38% of its length — consider raising fmax."*

### Rendering

- **FastMODA:** compute in `_ridge_worker` (backs `/analyze_ridge`, built on
  `ridge_gpu.extract_ridge`); return `boundary_flag` + `boundary_msg`; the
  template shows a small amber/red badge beside the ridge plot with the message
  as a tooltip.
- **MODA:** compute in the Filtering module's ridge path (`ecurve.m` / ridge
  callback); render as a small coloured lamp/annotation next to the ridge axes.

---

## 3 · Binned frequency-density overlay (both apps)

Behind the continuous **marginal spectrum** (time-averaged amplitude/power),
draw a binned density in **log or linear** frequency bins, plus a button that
snaps the bins to spectral structure. Implemented in **both** MODA and FastMODA.

### Algorithm

- Marginal `P(f)` = time-average of the TFR. Per-bin density = `∫P(f) df` over
  the bin, drawn as translucent background bars behind the existing curve.
- **Uniform bins:** `linspace` (linear) or `logspace` (log) edges — a toggle.
- **"Fit bins to peaks/troughs" button:** lightly smooth `P(f)`, detect troughs
  (local minima) and use them as **bin edges**, so each bin is centred on one
  peak. (Edges-at-troughs is what makes each bin "central to a peak.") Runs on
  whichever axis (log/linear) is active.

### Rendering

- **MODA:** background `bar`/`patch` (alpha) under the line on `plot_pow` (the
  marginal axes in TFA); a log/linear toggle + a "Fit bins to peaks" button;
  peaks/troughs via `findpeaks` on `P` and `-P`.
- **FastMODA:** a background `go.Bar` trace behind the marginal in the CWT/WFT
  figure; same toggle + button; peaks/troughs via `scipy.signal.find_peaks`.

---

## Cross-cutting notes

- **Parity first:** every feature ships in both apps with the same model and
  defaults, so results and workflows match.
- **Preprocessing feeds analysis:** the cropped/decimated signal is the single
  source of truth downstream; decimation always carries its new `fs`.
- **Tests:** add ground-truth checks (crop indices, integer-decimation `fs`
  bookkeeping and anti-alias behaviour, trough-edge bin placement) to the parity
  suite under `tests/parity/`.
