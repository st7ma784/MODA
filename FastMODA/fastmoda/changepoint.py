"""Changepoint detection library — two focused modes.

1. ``changepoints_at_frequency`` — track the power (or amplitude) at one chosen
   frequency over time and find changepoints in that single 1-D series.

2. ``changepoints_logbinned_power`` — split the full signal power into
   logarithmic frequency bins, forming a (time × n_bins) feature matrix, and find
   changepoints jointly across all bins (a change in *any* band shows up).

Both wrap ruptures' PELT (already a FastMODA dependency) via the shared
``detect_changepoints`` helper, standardise their features, and return
changepoint **time indices** (columns of the spectrogram) plus the changepoint
times in seconds — so callers can overlay them directly.

The log-bin edges come from ``spectral_bins.uniform_edges`` so this library and
the binned-density overlay stay at parity, and the MATLAB mirror
(``changepointsAtFrequency.m`` / ``changepointsLogBinnedPower.m``) matches too.
"""

from __future__ import annotations

import numpy as np

from fastmoda import detect_changepoints, sliding_fft
from fastmoda.spectral_bins import uniform_edges


# ── shared helpers ────────────────────────────────────────────────────────────

def _as_spectrogram(x_or_freqs, times=None, Sxx=None, fs=1.0, win_s=1.0):
    """Accept either a raw 1-D signal (compute the spectrogram) or a
    precomputed (freqs, times, Sxx). Returns (freqs, times, Sxx)."""
    if times is None and Sxx is None:
        x = np.asarray(x_or_freqs, dtype=float).ravel()
        return sliding_fft(x, fs=fs, win_s=win_s)
    return np.asarray(x_or_freqs, float), np.asarray(times, float), np.asarray(Sxx, float)


def _standardize(a):
    a = np.asarray(a, float)
    z = (a - np.nanmean(a, axis=0)) / (np.nanstd(a, axis=0) + 1e-12)
    # NaN/Inf (e.g. cone-of-influence-masked WT edges) would break the PELT
    # backend; after standardisation 0 is the neutral (mean) fill. Matches the
    # MATLAB engine's local_sanitize so both apps behave identically.
    z[~np.isfinite(z)] = 0.0
    return z


def bin_power_over_time(freqs, Sxx, edges, use_power=True):
    """Integrate the spectrogram over each frequency bin, per time column.

    Returns a (T, n_bins) matrix: band energy in each log/linear bin at each
    time. ``use_power`` squares the magnitude spectrogram first (energy) vs using
    amplitude directly.
    """
    freqs = np.asarray(freqs, float)
    S = np.asarray(Sxx, float)
    S = S ** 2 if use_power else S
    edges = np.asarray(edges, float)
    T = S.shape[1]
    nb = len(edges) - 1
    out = np.zeros((T, nb))
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        last = i == nb - 1
        m = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        if m.any():
            # integrate over frequency within the bin (trapz over ≥2 bins)
            if m.sum() >= 2:
                out[:, i] = np.trapezoid(S[m, :], freqs[m], axis=0) \
                    if hasattr(np, "trapezoid") else np.trapz(S[m, :], freqs[m], axis=0)
            else:
                out[:, i] = S[m, :].sum(axis=0)
    return out


# ── mode 1: single-frequency changepoints ────────────────────────────────────

def changepoints_at_frequency(x_or_freqs, target_freq, times=None, Sxx=None,
                              fs=1.0, win_s=1.0, pen="auto", use_power=True,
                              model="l2"):
    """Changepoints in the power/amplitude at one frequency over time.

    Args mirror ``_as_spectrogram``: pass a raw signal (+fs, win_s) OR a
    precomputed (freqs=x_or_freqs, times, Sxx). ``target_freq`` is snapped to the
    nearest spectrogram bin.

    Returns dict: {changepoints (time-idx), changepoint_times (s), times,
    series, target_freq (snapped), actual_freq, pen}.
    """
    freqs, times, Sxx = _as_spectrogram(x_or_freqs, times, Sxx, fs, win_s)
    fi = int(np.argmin(np.abs(freqs - float(target_freq))))
    actual = float(freqs[fi])
    series = Sxx[fi, :].astype(float)
    series = series ** 2 if use_power else series

    feat = _standardize(series).reshape(-1, 1)
    pen_val = _auto_pen(feat) if pen == "auto" else float(pen)
    cps = detect_changepoints(feat, model=model, pen=pen_val)
    cps = cps[(cps > 0) & (cps < len(times))]

    return {
        "changepoints": cps,
        "changepoint_times": [float(times[c]) for c in cps],
        "times": times,
        "series": series,
        "target_freq": actual,
        "actual_freq": actual,
        "pen": pen_val,
        "kind": "power" if use_power else "amplitude",
    }


# ── mode 2: log-binned full-power changepoints ────────────────────────────────

def changepoints_logbinned_power(x_or_freqs, times=None, Sxx=None, fs=1.0,
                                 win_s=1.0, n_bins=12, scale="log", pen="auto",
                                 use_power=True, model="l2", fmin=None, fmax=None):
    """Changepoints on the full signal power split into log (or linear) bins.

    Bins the spectrogram into ``n_bins`` bands (default logarithmic), standardises
    each band's series, and runs multivariate PELT — a change in any band is
    detected. Returns dict with the changepoints, the (T×n_bins) band matrix, and
    the bin edges (so callers can plot the band-power heatmap with the cps).
    """
    freqs, times, Sxx = _as_spectrogram(x_or_freqs, times, Sxx, fs, win_s)
    lo = float(fmin) if fmin else float(max(freqs[freqs > 0].min(), 1e-9))
    hi = float(fmax) if fmax else float(freqs.max())
    edges = uniform_edges(lo, hi, n_bins, scale)

    band = bin_power_over_time(freqs, Sxx, edges, use_power=use_power)  # (T, nb)
    feat = _standardize(band)
    pen_val = _auto_pen(feat) if pen == "auto" else float(pen)
    cps = detect_changepoints(feat, model=model, pen=pen_val)
    cps = cps[(cps > 0) & (cps < len(times))]

    centers = np.sqrt(np.maximum(edges[:-1], 1e-9) * edges[1:])
    return {
        "changepoints": cps,
        "changepoint_times": [float(times[c]) for c in cps],
        "times": times,
        "band_power": band,
        "bin_edges": edges,
        "bin_centers": centers,
        "scale": scale,
        "n_bins": int(len(centers)),
        "pen": pen_val,
        "kind": "power" if use_power else "amplitude",
    }


def _auto_pen(feat):
    """Adaptive PELT penalty: scale a BIC-like base by feature dimensionality and
    variability, so more/noisier channels don't over-segment."""
    feat = np.asarray(feat, float)
    T, d = feat.shape
    base = np.log(max(T, 2)) * d          # BIC-ish floor
    variability = float(np.mean(np.nanstd(feat, axis=0)))
    return base * (1.0 + variability)
