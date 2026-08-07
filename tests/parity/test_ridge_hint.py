"""Tests for the ridge boundary hint (fastmoda.ridge_gpu.ridge_boundary_hint).

Frequency-edge only: flags when a detected ridge hugs fmin/fmax, so the user
knows the true ridge may extend outside the analysed band.
"""

import numpy as np

from fastmoda.ridge_gpu import ridge_boundary_hint

FMIN, FMAX = 0.5, 20.0
N = 500


def test_ridge_in_middle_is_clean():
    ifreq = np.full(N, 3.0)                      # comfortably inside the band
    h = ridge_boundary_hint(ifreq, FMIN, FMAX)
    assert h["level"] == "none" and h["edge"] is None


def test_ridge_pinned_at_upper_edge():
    ifreq = np.full(N, FMAX)                      # sits on fmax
    h = ridge_boundary_hint(ifreq, FMIN, FMAX)
    assert h["level"] == "high" and h["edge"] == "upper"
    assert "fmax" in h["message"] and h["frac"] == 1.0


def test_ridge_pinned_at_lower_edge():
    ifreq = np.full(N, FMIN)
    h = ridge_boundary_hint(ifreq, FMIN, FMAX)
    assert h["level"] == "high" and h["edge"] == "lower"
    assert "fmin" in h["message"]


def test_partial_edge_contact_is_low():
    # ~12% of the ridge near fmax → low (between 8% and 25%)
    ifreq = np.full(N, 3.0)
    ifreq[: int(0.12 * N)] = FMAX
    h = ridge_boundary_hint(ifreq, FMIN, FMAX)
    assert h["level"] == "low" and h["edge"] == "upper"


def test_nan_and_degenerate_inputs():
    assert ridge_boundary_hint(np.full(N, np.nan), FMIN, FMAX)["level"] == "none"
    assert ridge_boundary_hint(np.full(N, 3.0), 10.0, 10.0)["level"] == "none"


def test_nans_are_ignored_not_counted():
    # COI-masked (NaN) samples must not count as "near edge"
    ifreq = np.full(N, 3.0)
    ifreq[: N // 2] = np.nan
    h = ridge_boundary_hint(ifreq, FMIN, FMAX)
    assert h["level"] == "none"
