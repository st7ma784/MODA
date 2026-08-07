"""Tests for the changepoint library (fastmoda.changepoint) — two modes:
single-frequency and log-binned full-signal power. Ground-truth signals with a
known change time; mirrored by MODA's changepointsAtFrequency / *LogBinnedPower.
"""

import numpy as np

from fastmoda.changepoint import (
    changepoints_at_frequency, changepoints_logbinned_power, bin_power_over_time,
)
from fastmoda import sliding_fft
from fastmoda.spectral_bins import uniform_edges

FS = 40.0
DUR = 40.0
N = int(FS * DUR)
T = np.arange(N) / FS
T_CHANGE = 20.0                       # known change time (s)


def _amplitude_step_5hz():
    """5 Hz tone whose amplitude triples halfway through."""
    amp = np.where(T < T_CHANGE, 1.0, 3.0)
    return amp * np.sin(2 * np.pi * 5.0 * T)


def _freq_switch_3_to_8():
    """3 Hz for the first half, 8 Hz for the second half."""
    x = np.empty(N)
    half = T < T_CHANGE
    x[half] = np.sin(2 * np.pi * 3.0 * T[half])
    x[~half] = np.sin(2 * np.pi * 8.0 * T[~half])
    return x


def _near(cp_times, target, tol=2.0):
    return any(abs(c - target) <= tol for c in cp_times)


# ── mode 1: single frequency ─────────────────────────────────────────────────

def test_single_frequency_detects_amplitude_step():
    x = _amplitude_step_5hz()
    r = changepoints_at_frequency(x, target_freq=5.0, fs=FS, win_s=1.0)
    assert abs(r["actual_freq"] - 5.0) <= 1.0
    assert len(r["changepoint_times"]) >= 1
    assert _near(r["changepoint_times"], T_CHANGE), r["changepoint_times"]


def test_single_frequency_snaps_to_nearest_bin():
    x = _amplitude_step_5hz()
    r = changepoints_at_frequency(x, target_freq=5.3, fs=FS, win_s=1.0)
    assert abs(r["actual_freq"] - 5.0) <= 1.0     # snapped to a real bin


def test_single_frequency_accepts_precomputed_spectrogram():
    freqs, times, Sxx = sliding_fft(_amplitude_step_5hz(), fs=FS, win_s=1.0)
    r = changepoints_at_frequency(freqs, target_freq=5.0, times=times, Sxx=Sxx, fs=FS)
    assert _near(r["changepoint_times"], T_CHANGE)


def test_single_frequency_steady_tone_has_no_changepoint():
    x = np.sin(2 * np.pi * 5.0 * T)               # constant amplitude
    r = changepoints_at_frequency(x, target_freq=5.0, fs=FS, win_s=1.0)
    assert len(r["changepoint_times"]) == 0


# ── mode 2: log-binned full power ────────────────────────────────────────────

def test_logbinned_detects_frequency_switch():
    x = _freq_switch_3_to_8()
    r = changepoints_logbinned_power(x, fs=FS, win_s=1.0, n_bins=12)
    assert r["scale"] == "log" and r["n_bins"] == 12
    assert len(r["changepoint_times"]) >= 1
    assert _near(r["changepoint_times"], T_CHANGE), r["changepoint_times"]


def test_logbinned_band_matrix_shape():
    freqs, times, Sxx = sliding_fft(_freq_switch_3_to_8(), fs=FS, win_s=1.0)
    edges = uniform_edges(0.5, FS / 2, 10, "log")
    band = bin_power_over_time(freqs, Sxx, edges)
    assert band.shape == (len(times), 10)
    # the 3 Hz bin should carry more energy in the first half, the 8 Hz bin more
    # in the second half
    half = len(times) // 2
    c = np.sqrt(edges[:-1] * edges[1:])
    b3 = int(np.argmin(np.abs(c - 3.0)))
    b8 = int(np.argmin(np.abs(c - 8.0)))
    assert band[:half, b3].mean() > band[half:, b3].mean()
    assert band[half:, b8].mean() > band[:half, b8].mean()


def test_logbinned_steady_signal_stable():
    x = np.sin(2 * np.pi * 4.0 * T)
    r = changepoints_logbinned_power(x, fs=FS, win_s=1.0, n_bins=12)
    assert len(r["changepoint_times"]) == 0


def test_linear_scale_option():
    x = _freq_switch_3_to_8()
    r = changepoints_logbinned_power(x, fs=FS, win_s=1.0, n_bins=8, scale="linear")
    assert r["scale"] == "linear"
    assert _near(r["changepoint_times"], T_CHANGE)


def test_handles_nan_in_spectrogram():
    # A cone-of-influence-masked wavelet spectrogram has NaN at the time edges;
    # the PELT backend rejects non-finite input, so both modes must sanitise it
    # (regression: MODA wt.m defaults to CutEdges='on').
    freqs, times, Sxx = sliding_fft(_amplitude_step_5hz(), fs=FS, win_s=1.0)
    Sxx = Sxx.copy()
    Sxx[:, :3] = np.nan            # mask the leading edge, as the COI would
    Sxx[:, -3:] = np.nan
    r1 = changepoints_at_frequency(freqs, 5.0, times=times, Sxx=Sxx, fs=FS)
    r2 = changepoints_logbinned_power(freqs, times=times, Sxx=Sxx, fs=FS, n_bins=10)
    # no crash, and the amplitude step is still found
    assert _near(r1["changepoint_times"], T_CHANGE)
    assert isinstance(r2["changepoint_times"], list)
