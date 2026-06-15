"""
fastmoda.condition_models unit tests.

Builds a tiny set of joblib model artifacts (matching the layout written by
scripts/train_condition_classifiers.py) under a tmp_path "model_dir" and
exercises classify() against them directly, so no real trained models or
datasets are required.

Run from repo root:
    pytest tests/test_condition_models.py -v
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "FastMODA"))

from fastmoda import condition_models  # noqa: E402

FEATURE_NAMES = ['spectral_band_power_alpha', 'phase_mean_amplitude', 'wavelet_energy']
CONDITIONS = ['normal', 'afib']


def _train_tiny_model(rng):
    # Two well-separated clusters so predict_proba isn't degenerate.
    x0 = rng.normal(loc=-1.0, scale=0.5, size=(20, len(FEATURE_NAMES)))
    x1 = rng.normal(loc=1.0, scale=0.5, size=(20, len(FEATURE_NAMES)))
    x = np.vstack([x0, x1])
    y = np.array([0] * 20 + [1] * 20)
    return LogisticRegression().fit(x, y)


@pytest.fixture
def model_dir(tmp_path):
    rng = np.random.default_rng(42)
    for condition in CONDITIONS:
        clf = _train_tiny_model(rng)
        joblib.dump({'model': clf, 'feature_names': FEATURE_NAMES},
                     str(tmp_path / f'{condition}.joblib'))

    meta = {'feature_names': FEATURE_NAMES, 'conditions': CONDITIONS}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))

    global_stats = {
        'feature_names': FEATURE_NAMES,
        'mean': [0.0] * len(FEATURE_NAMES),
        'std': [1.0] * len(FEATURE_NAMES),
    }
    (tmp_path / 'global_feature_stats.json').write_text(json.dumps(global_stats))

    return str(tmp_path)


def test_load_models_and_global_stats(model_dir):
    models = condition_models.load_models(model_dir)
    assert set(models.keys()) == set(CONDITIONS)
    for artifact in models.values():
        assert artifact['feature_names'] == FEATURE_NAMES

    stats = condition_models.load_global_stats(model_dir)
    assert set(stats['features'].keys()) == set(FEATURE_NAMES)


def test_load_models_missing_meta_returns_empty(tmp_path):
    assert condition_models.load_models(str(tmp_path)) == {}
    assert condition_models.load_global_stats(str(tmp_path)) is None


def test_classify_returns_probabilities_and_top_features(model_dir):
    vector = [2.5, -0.3, 1.1]
    results = condition_models.classify(vector, FEATURE_NAMES, model_dir=model_dir)

    assert set(results.keys()) == set(CONDITIONS)
    for result in results.values():
        assert 0.0 <= result['probability'] <= 1.0
        assert len(result['top_features']) > 0
        for feature in result['top_features']:
            assert set(feature.keys()) == {'name', 'value', 'deviation', 'contribution'}
            assert feature['name'] in FEATURE_NAMES


def test_classify_with_no_models_returns_empty(tmp_path):
    assert condition_models.classify([1.0], ['x'], model_dir=str(tmp_path)) == {}


def test_classify_aligns_and_zero_fills_missing_features(model_dir):
    # Caller's vector only covers 2 of the 3 model features, in a different
    # order; the missing feature should be zero-filled rather than erroring.
    vector = [1.1, 2.5]
    names = ['wavelet_energy', 'spectral_band_power_alpha']

    results = condition_models.classify(vector, names, model_dir=model_dir)
    assert set(results.keys()) == set(CONDITIONS)


def test_classify_uses_per_device_baseline_when_available(model_dir):
    vector = [2.5, -0.3, 1.1]
    baseline_stats = {
        'features': {name: {'mean': 1.0, 'std': 2.0} for name in FEATURE_NAMES}
    }

    with_baseline = condition_models.classify(
        vector, FEATURE_NAMES, baseline_stats=baseline_stats, model_dir=model_dir)
    without_baseline = condition_models.classify(
        vector, FEATURE_NAMES, model_dir=model_dir)

    # Different baseline stats -> different deviations -> different scores.
    for condition in CONDITIONS:
        assert with_baseline[condition]['probability'] != without_baseline[condition]['probability']


def test_classify_falls_back_to_global_stats_for_empty_baseline(model_dir):
    vector = [2.5, -0.3, 1.1]

    empty_baseline = condition_models.classify(
        vector, FEATURE_NAMES, baseline_stats={'features': {}}, model_dir=model_dir)
    no_baseline = condition_models.classify(
        vector, FEATURE_NAMES, baseline_stats=None, model_dir=model_dir)

    for condition in CONDITIONS:
        assert empty_baseline[condition]['probability'] == pytest.approx(
            no_baseline[condition]['probability'])
