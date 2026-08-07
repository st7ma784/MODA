"""Canonical inventory of MODA (MATLAB desktop) UI features and their FastMODA
(Python/Flask) equivalents.

Each row is one *user-facing capability* of the MODA GUIs (not one button) drawn
from the five App-Designer modules under ``allguis/guis`` plus the shared
``allguis/codes/Universal`` I/O layer. For every capability we record:

  * ``module``        - which MODA GUI it belongs to
  * ``feature``       - the capability, in user terms
  * ``moda_file``     - MATLAB source that implements it
  * ``fm_route``      - FastMODA HTTP route that exposes it (or None if the
                        capability is client-side / not a route)
  * ``fm_symbol``     - "module:function" in the ``fastmoda`` package that backs
                        it (or None)

The parity test (``test_ui_parity.py``) asserts that every non-None ``fm_route``
is actually registered in ``FastMODA/app.py`` and every non-None ``fm_symbol``
is importable. ``expected_gap=True`` marks capabilities that are intentionally
MODA-only (desktop-specific, e.g. native file dialogs) so they don't fail the
suite but are still reported.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Feature:
    module: str
    feature: str
    moda_file: str
    fm_route: Optional[str] = None
    fm_symbol: Optional[str] = None
    expected_gap: bool = False
    note: str = ""


INVENTORY = [
    # ── Time-Frequency Analysis ────────────────────────────────────────────
    Feature("TFA", "Continuous wavelet transform (Morlet/Lognorm/Bump)",
            "allguis/guis/tfa/Functions/wt.m",
            "/analyze_cwt", "analysis_gpu:cwt_gpu"),
    Feature("TFA", "Windowed Fourier transform (Gaussian/Hann/Blackman/Kaiser…)",
            "allguis/guis/tfa/Functions/wft.m",
            "/analyze_wft", "filtering:wft"),
    Feature("TFA", "Short-time Fourier / spectrogram",
            "allguis/guis/tfa/Functions/wft.m",
            "/analyze_stft", "analysis_gpu:stft_gpu"),
    Feature("TFA", "Sliding-window power spectrum + band powers",
            "allguis/guis/tfa/TimeFrequencyAnalysis.m",
            "/analyze", "fastmoda:sliding_fft"),
    Feature("TFA", "Ridge / curve extraction from TFR",
            "allguis/guis/filtering/Functions/ecurve.m",
            "/analyze_ridge", "ridge_gpu:extract_ridge"),
    Feature("TFA", "Instantaneous phase/amplitude (Hilbert)",
            "allguis/guis/tfa/Functions/wt.m",
            "/analyze_hilbert", "analysis_gpu:compute_instantaneous_phase_gpu"),
    Feature("TFA", "MODWT maximal-overlap DWT decomposition",
            "allguis/guis/tfa/Functions/wt.m",
            "/analyze_modwt", "modwt_gpu:modwt_gpu",
            note="FastMODA superset — MODA exposes CWT/WFT, not MODWT directly"),

    # ── Coherence ──────────────────────────────────────────────────────────
    Feature("Coherence", "Wavelet phase coherence",
            "allguis/guis/coherence/Functions/wphcoh.m",
            "/analyze_coherence", "coherence_gpu:wavelet_phase_coherence_gpu"),
    Feature("Coherence", "Time-localised phase coherence",
            "allguis/guis/coherence/Functions/tlphcoh.m",
            "/analyze_coherence", "coherence_gpu:time_localized_coherence_gpu"),
    Feature("Coherence", "Multi-signal / group coherence",
            "allguis/guis/coherence/CoherenceMulti.m",
            "/analyze_group", "coherence_gpu:compute_multi_pair_coherence_gpu"),
    Feature("Coherence", "Surrogate significance testing",
            "allguis/codes/Universal/MODAread.m",
            "/analyze_surrogates", "surrogates_gpu:batched_iaaft_surrogates_gpu"),

    # ── Bispectrum ─────────────────────────────────────────────────────────
    Feature("Bispectrum", "Wavelet bispectrum / bicoherence",
            "allguis/guis/bispectrum/Functions/bispecWavNew.m",
            "/analyze_bispectrum", "bispectrum_gpu:wavelet_bispectrum_gpu"),
    Feature("Bispectrum", "Biphase / biamplitude time series",
            "allguis/guis/bispectrum/Functions/bispecWavNew.m",
            "/analyze_biphase", "biphase_gpu:biphase_timeseries"),
    Feature("Bispectrum", "Four-component bispectrum (b111/b122/…)",
            "allguis/guis/bispectrum/Bispectrum.m",
            "/analyze_bispectrum4", "biphase_gpu:bispectrum4"),

    # ── Bayesian inference / coupling ──────────────────────────────────────
    Feature("Bayesian", "Dynamical Bayesian phase inference",
            "allguis/guis/bayesian/Functions/bayes_main.m",
            "/analyze_bayesian", "bayesian_full_gpu:bayesian_phase_inference_gpu"),
    Feature("Bayesian", "Coupling functions",
            "allguis/guis/bayesian/Functions/CFprint.m",
            "/analyze_coupling", "coupling_gpu:estimate_coupling_functions"),
    Feature("Bayesian", "Coupling direction / synchronisation map",
            "allguis/guis/bayesian/Functions/dirc.m",
            "/analyze_syncmap", "bayesian_gpu:compute_coupling_direction"),

    # ── Filtering ──────────────────────────────────────────────────────────
    Feature("Filtering", "Butterworth band-pass filtering",
            "allguis/guis/filtering/Functions/loop_butter.m",
            "/filter_butter", "filtering:butterworth_bandpass"),

    # ── Preprocessing (dedicated tab, both apps) ───────────────────────────
    Feature("Preprocessing", "Clip / bulk-crop / integer-decimate",
            "allguis/guis/preprocessing/Preprocessing.m",
            "/preprocess_apply", "preprocess:crop_and_decimate"),

    # ── Changepoint detection (dedicated tab, both apps) ───────────────────
    Feature("Changepoints", "Single-frequency changepoints",
            "allguis/guis/changepoints/Changepoints.m",
            "/analyze_changepoints", "changepoint:changepoints_at_frequency"),
    Feature("Changepoints", "Log-binned full-power changepoints",
            "allguis/guis/changepoints/Changepoints.m",
            "/analyze_changepoints", "changepoint:changepoints_logbinned_power"),

    # ── Universal I/O & data handling ──────────────────────────────────────
    Feature("I/O", "Load time series (.mat/.csv/.npy)",
            "allguis/codes/Universal/MODAread.m",
            None, "fastmoda:load_signal",
            note="load_signal is called by every /analyze_* route"),
    Feature("I/O", "Sampling-frequency / preprocessing settings",
            "allguis/codes/Universal/MODAsettings.m",
            "/analyze", None,
            note="fs / window / penalty are form fields on the analyze routes"),
    Feature("I/O", "Save results as .mat / .csv",
            "allguis/guis/tfa/TimeFrequencyAnalysis.m",
            None, None, expected_gap=True,
            note="Desktop-only native save dialog; web app downloads plots/JSON"),
    Feature("I/O", "Load / save analysis session",
            "allguis/guis/tfa/TimeFrequencyAnalysis.m",
            None, None, expected_gap=True,
            note="Desktop-only session persistence"),
    Feature("I/O", "Export report (plot + parameters)",
            "allguis/codes/Universal/exportReportPDF.m",
            None, None, expected_gap=True,
            note="Desktop-only PDF export; web app renders interactive Plotly"),

    # ── ML / classification layer (FastMODA-forward capability) ────────────
    Feature("ML", "Feature-vector extraction for ML",
            "allguis/guis/tfa/TimeFrequencyAnalysis.m",
            "/analyze_features", "pipeline:compute_feature_vector",
            note="FastMODA superset — MODA has no ML layer"),
    Feature("ML", "Condition classification against baseline",
            "-",
            "/classify", "condition_models:classify",
            note="FastMODA superset"),
]


def summary():
    mods = {}
    for f in INVENTORY:
        mods.setdefault(f.module, []).append(f)
    return mods
