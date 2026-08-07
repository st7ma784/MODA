"""Binned frequency-density of a marginal (time-averaged) spectrum.

Splits the frequency axis into **linear** or **logarithmic** bins and reports the
density (integrated marginal power/amplitude) per bin, for drawing as a
background behind the continuous spectrum. Also fits bins to spectral structure:
edges are placed at the **troughs** of the marginal so each bin is centred on a
peak. Shared by the TFA overlay and mirrored by MODA's binned view.
"""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and removed as an attribute).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def uniform_edges(fmin, fmax, n_bins=20, scale="log"):
    """Bin edges over [fmin, fmax] — ``scale`` = 'log' or 'linear'."""
    n_bins = max(1, int(n_bins))
    fmin = max(float(fmin), 1e-12)
    if scale == "log":
        return np.logspace(np.log10(fmin), np.log10(float(fmax)), n_bins + 1)
    return np.linspace(float(fmin), float(fmax), n_bins + 1)


def _smooth(P, w=5):
    w = max(1, int(w) | 1)                 # force odd
    if w <= 1 or len(P) < w:
        return np.asarray(P, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(P, float), k, mode="same")


def fit_bins_to_peaks_edges(freqs, P, smooth=5):
    """Edges placed at the troughs (local minima) of the smoothed marginal, so
    each resulting bin is centred on one peak. Falls back to a single bin."""
    from scipy.signal import find_peaks
    freqs = np.asarray(freqs, float)
    Ps = _smooth(P, smooth)
    troughs, _ = find_peaks(-Ps)
    idx = np.unique(np.concatenate(([0], troughs, [len(freqs) - 1]))).astype(int)
    return freqs[idx]


def bin_spectrum(freqs, P, edges):
    """Integrate the marginal P over each [edge_i, edge_{i+1}] bin.

    Returns a list of dicts: f_lo, f_hi, f_center (geometric), density (integral),
    density_norm (density / max density, for background-bar heights).
    """
    freqs = np.asarray(freqs, float)
    P = np.asarray(P, float)
    edges = np.asarray(edges, float)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        m = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        if m.sum() >= 2:
            dens = float(_trapz(P[m], freqs[m]))
        elif m.sum() == 1:
            dens = float(P[m][0])
        else:
            dens = 0.0
        out.append({
            "f_lo": float(lo), "f_hi": float(hi),
            "f_center": float(np.sqrt(max(lo, 1e-9) * hi)),
            "density": dens,
        })
    mx = max((b["density"] for b in out), default=0.0) or 1.0
    for b in out:
        b["density_norm"] = b["density"] / mx
    return out


def binned_spectrum_all(freqs, P, n_bins=20):
    """Marginal + all three binnings (linear / log / peak-fitted) for the UI."""
    freqs = np.asarray(freqs, float)
    P = np.asarray(P, float)
    P = np.where(np.isfinite(P), P, 0.0)
    fmin, fmax = float(freqs.min()), float(freqs.max())
    return {
        "freqs": [float(f) for f in freqs],
        "marginal": [float(v) for v in P],
        "linear": bin_spectrum(freqs, P, uniform_edges(fmin, fmax, n_bins, "linear")),
        "log": bin_spectrum(freqs, P, uniform_edges(fmin, fmax, n_bins, "log")),
        "peaks": bin_spectrum(freqs, P, fit_bins_to_peaks_edges(freqs, P)),
    }
