"""Tests for the preprocessing crop + integer-decimation model
(fastmoda.preprocess), shared by the /preprocess endpoints and mirrored by
MODA's Preprocessing module.

Run inside the FastMODA image:
    pytest -q /repo/tests/parity/test_preprocess.py
"""

import numpy as np
import pytest

from fastmoda.preprocess import (
    crop_and_decimate, crop_indices, decimate_integer, integer_rate_options,
)

FS, N = 40.0, 4000            # 100 s at 40 Hz
T = np.arange(N) / FS
X = np.sin(2 * np.pi * 2.0 * T)


def test_crop_range_indices():
    i0, i1 = crop_indices(N, FS, "range", start_s=10, stop_s=30)
    assert i0 == 400 and i1 == 1200


def test_crop_first_and_final():
    i0, i1 = crop_indices(N, FS, "first", length_s=25)
    assert (i0, i1) == (0, 1000)
    i0, i1 = crop_indices(N, FS, "final", length_s=25)
    assert (i0, i1) == (3000, 4000)


def test_crop_clamps_to_bounds():
    # stop beyond the signal is clamped to n
    i0, i1 = crop_indices(N, FS, "range", start_s=90, stop_s=999)
    assert i1 == N and i0 == 3600


def test_empty_crop_raises():
    with pytest.raises(ValueError):
        crop_indices(N, FS, "range", start_s=30, stop_s=10)


def test_crop_and_decimate_range():
    y, fs_new, info = crop_and_decimate(X, FS, mode="range",
                                        start_s=10, stop_s=30, decimate_factor=1)
    assert fs_new == FS
    assert len(y) == 800
    assert info["t_start"] == 10.0 and info["t_stop"] == 30.0


def test_integer_decimation_halves_rate_and_length():
    y, fs_new, info = crop_and_decimate(X, FS, mode="none", decimate_factor=2)
    assert fs_new == 20.0
    assert abs(len(y) - N / 2) <= 1
    assert info["decimate_factor"] == 2


def test_decimation_preserves_low_frequency_tone():
    # a 2 Hz tone survives decimation to 20 Hz (well below Nyquist=10)
    y = decimate_integer(X, 2)
    fs2 = FS / 2
    freqs = np.fft.rfftfreq(len(y), 1 / fs2)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(peak - 2.0) < 0.2


def test_large_factor_is_staged():
    # factor 24 = 2*2*2*3 → staged, no crash, correct rate
    y, fs_new, _ = crop_and_decimate(X, FS, mode="none", decimate_factor=24)
    assert fs_new == FS / 24
    assert len(y) > 0


def test_rate_options_are_integer_divisions():
    opts = integer_rate_options(40.0, min_rate=1.0)
    factors = [k for k, _ in opts]
    assert factors[0] == 1
    for k, r in opts:
        assert abs(r - 40.0 / k) < 1e-9
        assert r >= 1.0 or k == 1
