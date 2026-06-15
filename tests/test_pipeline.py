"""
fastmoda.pipeline unit tests.

Run from repo root:
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "FastMODA"))

from fastmoda.pipeline import (  # noqa: E402
    DEFAULT_ANALYSES,
    compute_all_analyses,
    compute_feature_vector,
)
from fastmoda.feature_extraction import extract_all_features  # noqa: E402

FS = 250.0
N = int(FS * 5)  # 5 seconds


def _signal(n=N, fs=FS):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.cos(2 * np.pi * 1.0 * t)


def test_compute_all_analyses_default_keys():
    results = compute_all_analyses(_signal(), FS)
    assert set(results.keys()) == set(DEFAULT_ANALYSES)


def test_compute_all_analyses_subset():
    results = compute_all_analyses(_signal(), FS, analyses=['spectral', 'phase'])
    assert set(results.keys()) == {'spectral', 'phase'}


def test_spectral_result_shape():
    spectral = compute_all_analyses(_signal(), FS, analyses=['spectral'])['spectral']
    for key in ('freqs', 'spec_data', 'times', 'changepoints', 'bands'):
        assert key in spectral


def test_phase_result_shape():
    x = _signal()
    phase = compute_all_analyses(x, FS, analyses=['phase'])['phase']
    for key in ('phase', 'amplitude', 'inst_freq', 'fs'):
        assert key in phase
    assert len(phase['phase']) == len(x)


def test_compute_feature_vector_matches_extract_all_features():
    x = _signal()
    vector, names = compute_feature_vector(x, FS)
    expected_vector, expected_names = extract_all_features(compute_all_analyses(x, FS))

    assert names == expected_names
    assert len(vector) == len(names)
    np.testing.assert_allclose(vector, expected_vector)


def test_compute_feature_vector_names_sorted_and_namespaced():
    _, names = compute_feature_vector(_signal(), FS)

    assert names == sorted(names)
    prefixes = {name.split('_', 1)[0] for name in names}
    assert prefixes <= {'spectral', 'phase', 'stft', 'wavelet', 'bispectrum'}


def test_compute_feature_vector_is_finite():
    vector, _ = compute_feature_vector(_signal(), FS)
    assert np.all(np.isfinite(vector))
