# MODA — Original Feature Catalogue & FastMODA / App Coverage

Source of truth for what the original MATLAB MODA toolbox exposes, what the
FastMODA REST backend implements, and what the Flutter app surfaces in its UI.

Generated from:
- `allguis/guis/{tfa,filtering,coherence,bispectrum,bayesian}/` (MATLAB code)
- `FastMODA/app.py` + `FastMODA/fastmoda/*.py` (Python backend)
- `APP/lib/services/fastmoda_client.dart` + `APP/lib/screens/analysis_screen.dart`

Status legend: ✅ implemented · 🟡 partial · ❌ missing

---

## 1. Launcher / shell

| MODA feature | FastMODA endpoint | App route | Status |
|---|---|---|---|
| Launcher window with 5 module buttons | n/a (front-end concern) | `screens/home.dart` bottom-nav (Dashboard / Devices / Analysis / Settings) | ✅ |
| `MODA.m` version + MATLAB version check | `GET /health` (server reachable), `GET /api/gpu-info` (capabilities) | Dashboard server chip + GPU badge | ✅ |
| Load `.mat` / `.csv` time-series | All `/analyze*` accept `.mat`, `.npy`, `.csv` (`load_signal`) | `file_picker` + `addChannel()` in `SignalService` | ✅ |
| Row/column orientation prompt | server auto-detects (transposes if rows ≠ samples) | importer in `SignalService.importExtraSignals` | ✅ |
| Save session (`.mat`) / Load session | `AnalysisHistoryService` (sqflite) saves every result | History tab of Analysis screen | ✅ |
| Save figure as png/svg/pdf | Plotly JSON returned — client renders + exports | `utils/export.dart` (`exportResultJson`, `exportSignalCsv`) | ✅ |
| Truncate signal (`Xlim` / zoom-rect) | client-side: send only sliced bytes | not yet wired into UI form | 🟡 |
| Downsample large arrays | client-side: resample before upload | not yet wired into UI form | 🟡 |

---

## 2. Time-Frequency Analysis  (`allguis/guis/tfa/TimeFrequencyAnalysis.m`)

MATLAB primitives: `wt.m`, `wft.m`, `wtwrapper.m`, `testwt.m`.

| MODA feature | FastMODA endpoint | App | Status |
|---|---|---|---|
| Wavelet Transform (Lognorm / Morlet / Bump) | `POST /analyze_cwt` (`freq_min`, `freq_max`, `n_freqs`, wavelet via `_cwt_worker`) | "CWT" card | ✅ |
| Windowed Fourier Transform | `POST /analyze_wft` (Gaussian-windowed STFT) | "WFT" card | ✅ |
| Short-Time Fourier Transform (extra) | `POST /analyze_stft` | "STFT" card | ✅ |
| Plot type: Power vs Amplitude | result returns both spectrogram + amplitude surface | rendered via Plotly | ✅ |
| Preprocess (off/on) | `pp_arg` accepted by `_cwt_worker`, default off | request param (default off; exposed via dialog) | ✅ |
| Cut edges (off/on) | `cut_edges` param on `/analyze_ridge`, edges trimmed in CWT worker | ✅ | ✅ |
| Frequency interval markers ("Intervals" field) | overlays added when `intervals` param supplied | param exposed via spectral form | 🟡 |
| Statistics panel (group1 / group2 / t-test) | `POST /analyze_group` (Wilcoxon rank-sum) | "Group Comparison" card | ✅ |
| Save WT coefficients (.mat) | `feature_extraction.py` returns coefficients; result JSON has full payload | "Result JSON" export button | ✅ |
| Test wavelet smoke-check | covered by `/analyze_features` config validation | not exposed as a separate button | 🟡 |

---

## 3. Filtering & Ridge Extraction  (`allguis/guis/filtering/Filtering.m`)

MATLAB primitives: `bandpass_butter.m`, `loop_butter.m`, `ridge_extraction.m`,
`MODAridge_filter.m`, `ecurve.m`, `rectfr.m`, `Fourier.m`.

| MODA feature | FastMODA endpoint | App | Status |
|---|---|---|---|
| Bandpass Butterworth + polynomial detrend | `POST /filter_butter` (`f_low`, `f_high`, `order`, `detrend_degree`) | "Butterworth Filter" card + dialog | ✅ |
| Ridge extraction (instantaneous f, A, φ) | `POST /analyze_ridge` (`smooth_len`, `n_cycles`, `wavelet`, `cut_edges`) | "Ridge Extraction" card | ✅ |
| Signal reconstruction from ridge | included in `_ridge_worker` result (`reconstruction_plot`) | rendered in ridge result | ✅ |
| Fourier display (linear / log) | `/analyze_stft` + `/analyze_wft` both expose log axis option | dialog scale option | ✅ |
| Band-power timeline (etype=2) | `/analyze` returns `band_powers` plot + `band_freqs` | spectral result panel | ✅ |
| Phase extraction (Hilbert) | `POST /analyze_hilbert` | "Hilbert Phase" card | ✅ |
| Save filtered signal / phase / ridge plots | client-side via export utility | export buttons | ✅ |

---

## 4. Wavelet Phase Coherence  (`allguis/guis/coherence/CoherenceMulti.m`)

MATLAB primitives: `MODAwpc.m`, `wphcoh.m`, `tlphcoh.m`, `surrcalc.m`.

| MODA feature | FastMODA endpoint | App | Status |
|---|---|---|---|
| Wavelet phase coherence (WPC) | `POST /analyze_coherence` (`win`, `overlap`, `numcycles`) | "Phase Coherence" card | ✅ |
| Time-localised phase coherence (TPC) | included in coherence result (`pair_plots` heatmap = TPC) | rendered in result | ✅ |
| 2 – 6 signals, all pairs | endpoint accepts `files[]` 2–6, computes all pairs | channel-import UI | ✅ |
| Surrogate test (FT / AAFT / IAAFT1 / IAAFT2 / WIAAFT / tshift / CPP) | `POST /analyze_surrogates` (`surrogate_method`, `n_surrogates`, `test_type`) | "Surrogate Test" card + dialog | ✅ |
| Surrogate percentile threshold | `target_freq` + percentile in result | dialog | ✅ |
| Subtract surrogate mean | flag in coherence result | rendered when present | ✅ |

---

## 5. Wavelet Bispectrum  (`allguis/guis/bispectrum/Bispectrum.m`)

MATLAB primitives: `bispecWavNew.m`, `bispecWavMod.m`, `biphaseWavNew.m`,
`biphaseWavMod.m`, `wavsurrogate.m`, `compareMatrix.m`.

| MODA feature | FastMODA endpoint | App | Status |
|---|---|---|---|
| Wavelet bispectrum (1 signal) | `POST /analyze_bispectrum` (`freq_min`, `freq_max`, `n_freqs`, `bispec_type`) | "Bispectrum" card | ✅ |
| 4 cross-bispectra (b111 / b222 / b122 / b211) | `POST /analyze_bispectrum4` (`nfft`) | "4-Way Bispectrum" card | ✅ |
| Biphase + biamplitude at freq pair | `POST /analyze_biphase` (`f1`, `f2`, `wavelet`, `n_cycles`) | "Biphase Time Series" card + dialog | ✅ |
| IAAFT2 wavelet surrogates | `surrogate_method=iaaft2` in `/analyze_surrogates`; embedded in bispectrum worker | dialog | ✅ |
| Detrend before bispectrum | `pp_arg` flag in worker; default on | exposed via spectral params | ✅ |
| Save bispectrum amplitude / phase | result includes `bispec_amp`, `bispec_phase` plots + raw arrays | "Result JSON" export | ✅ |

---

## 6. Dynamical Bayesian Inference  (`allguis/guis/bayesian/Bayesian.m`)

MATLAB primitives: `bayes_main.m`, `full_bayesian.m`, `bayesPhs.m`, `dirc.m`,
`sync_map.m`, `CFprint.m`.

| MODA feature | FastMODA endpoint | App | Status |
|---|---|---|---|
| Dynamical Bayesian inference (2 signals, 2 bands) | `POST /analyze_bayesian` (`band1_*`, `band2_*`, `window_s`, `n_surrogates`) | "Bayesian Inference" card | ✅ |
| Coupling strength + direction (`cpl1`, `cpl2`, `dirc`) | results include `coupling_strength_plot`, `coupling_direction_plot` | rendered | ✅ |
| Coupling functions (`cf1`, `cf2`, mean, video) | result includes `cf1_surface`, `cf2_surface` | rendered | ✅ |
| Synchronisation map | `POST /analyze_syncmap` (`bn`, `win_s`, band pairs) | "Synchronisation Map" card + dialog | ✅ |
| Surrogate phase shuffling | embedded in `/analyze_bayesian` (`n_surrogates`); also `/analyze_surrogates` | dialogs | ✅ |
| Coupling-function OLS estimator (independent) | `POST /analyze_coupling` | "Coupling Functions" card + dialog | ✅ |
| Load filtered signal from Filtering window | client passes ridge-filter output as 2nd signal via `addChannel` | channel-import row | ✅ |

---

## 7. Cross-cutting

| Capability | FastMODA endpoint | App | Status |
|---|---|---|---|
| MODWT decomposition (extra — not in MATLAB MODA) | `POST /analyze_modwt` | "MODWT" card | ✅ |
| Async task lifecycle (`task_id` + `/status/<id>`) | `GET /status/<task_id>` | `SignalService._awaitTask()` | ✅ |
| GPU / CPU backend reporting | `GET /api/gpu-info` | Dashboard chip | ✅ |
| Health check | `GET /health` | Dashboard chip | ✅ |
| Pre-shared API key | `X-API-Key` header (Helm secret) | `_ApiKeyInterceptor` in `FastModaClient` | ✅ |
| ML-ready feature extraction | `POST /analyze_features` | "Feature Extraction" card | ✅ |
| Group statistics (two cohorts) | `POST /analyze_group` (Wilcoxon) | "Group Comparison" card | ✅ |
| Session history | n/a (client-side) | `AnalysisHistoryService` + History tab | ✅ |
| BLE live capture (custom MODA-BLE-SP) | n/a | `BleService` + Devices screen | ✅ |

---

## Coverage summary

- **All five MATLAB modules** (TFA, Filtering/Ridge, Coherence, Bispectrum,
  Bayesian) have at least one FastMODA endpoint and at least one app surface.
- **Previously orphaned `FastModaClient.submitModwt`** is now wired through
  `SignalService` and surfaced as an app card.
- **`/analyze_group`** is wired into `FastModaClient` and `SignalService` with
  its own "Group Comparison" card.
- **No MATLAB MODA primitive is missing a server endpoint.**

Build & test:

```bash
cd APP && flutter pub get && flutter analyze && flutter test
flutter build web    # or apk / ios
```

Run the backend:

```bash
cd FastMODA && ./start_fastmoda.sh        # local
# or
docker compose up -d fastmoda-cpu          # CPU
docker compose --profile gpu up -d fastmoda-gpu
```
