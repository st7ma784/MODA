"""Tests for the binned frequency-density model (fastmoda.spectral_bins),
shared by the TFA overlay and mirrored by MODA's binned view.
"""

import numpy as np

from fastmoda.spectral_bins import (
    uniform_edges, bin_spectrum, fit_bins_to_peaks_edges, binned_spectrum_all,
)

F = np.linspace(1.0, 10.0, 200)
# marginal with two clear peaks (3 Hz and 8 Hz) separated by a trough near 5.5
P = (np.exp(-0.5 * ((F - 3.0) / 0.4) ** 2)
     + 0.8 * np.exp(-0.5 * ((F - 8.0) / 0.4) ** 2))


def test_uniform_edges_counts_and_monotonic():
    e_lin = uniform_edges(1, 10, 20, "linear")
    e_log = uniform_edges(1, 10, 20, "log")
    assert len(e_lin) == 21 and len(e_log) == 21
    assert np.all(np.diff(e_lin) > 0) and np.all(np.diff(e_log) > 0)
    assert abs(e_lin[0] - 1) < 1e-9 and abs(e_lin[-1] - 10) < 1e-9


def test_bin_spectrum_normalises_and_conserves_order():
    bins = bin_spectrum(F, P, uniform_edges(1, 10, 15, "linear"))
    assert len(bins) == 15
    assert max(b["density_norm"] for b in bins) == 1.0
    assert all(b["f_lo"] < b["f_hi"] for b in bins)
    # the two densest bins should bracket the 3 Hz and 8 Hz peaks
    dens_sorted = sorted(bins, key=lambda b: b["density"], reverse=True)
    centers = sorted(b["f_center"] for b in dens_sorted[:2])
    assert abs(centers[0] - 3) < 1.0 and abs(centers[1] - 8) < 1.0


def test_fit_to_peaks_places_edge_at_trough():
    edges = fit_bins_to_peaks_edges(F, P)
    # two peaks separated by one trough → interior edge near 5.5, i.e. 3 edges
    assert len(edges) == 3
    interior = edges[1]
    assert 4.5 < interior < 6.5
    # → exactly two bins, one centred on each peak
    bins = bin_spectrum(F, P, edges)
    assert len(bins) == 2


def test_binned_spectrum_all_shape():
    d = binned_spectrum_all(F, P, n_bins=12)
    assert len(d["freqs"]) == len(F) and len(d["marginal"]) == len(F)
    assert len(d["linear"]) == 12 and len(d["log"]) == 12
    assert len(d["peaks"]) >= 1
    # NaNs in the marginal are zeroed, not propagated
    Pn = P.copy(); Pn[:5] = np.nan
    d2 = binned_spectrum_all(F, Pn)
    assert all(np.isfinite(v) for v in d2["marginal"])
