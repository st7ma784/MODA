"""Per-condition linear classifier heads, with feature-contribution explanations.

Models are trained offline by scripts/train_condition_classifiers.py and
persisted under MODA_DATA_DIR/models/ as joblib artifacts (one-vs-rest
sklearn LogisticRegression per condition, trained on z-scored feature
deviations) plus global_feature_stats.json and meta.json.

classify() turns a raw feature vector into a per-condition probability and a
top-K list of {name, value, deviation, contribution} explaining the score via
contribution = coefficient * deviation.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np

from fastmoda import storage
from fastmoda.baseline import compute_deviation

TOP_K = 5


def load_global_stats(model_dir: Optional[str] = None) -> Optional[Dict]:
    """Return {'features': {name: {mean, std}}} from global_feature_stats.json, or None."""
    model_dir = model_dir or storage.MODELS_DIR
    path = os.path.join(model_dir, 'global_feature_stats.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    features = {
        name: {'mean': mean, 'std': std}
        for name, mean, std in zip(data['feature_names'], data['mean'], data['std'])
    }
    return {'features': features}


def load_models(model_dir: Optional[str] = None) -> Dict[str, Dict]:
    """Return {condition: {'model': clf, 'feature_names': [...]}} for available models."""
    model_dir = model_dir or storage.MODELS_DIR
    meta_path = os.path.join(model_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)

    models = {}
    for condition in meta.get('conditions', []):
        model_path = os.path.join(model_dir, f'{condition}.joblib')
        if os.path.exists(model_path):
            models[condition] = joblib.load(model_path)
    return models


def _align(feature_vector: Sequence[float], feature_names: Sequence[str],
           target_names: Sequence[str]) -> np.ndarray:
    """Reindex feature_vector (named by feature_names) to target_names, zero-filling gaps."""
    lookup = dict(zip(feature_names, feature_vector))
    return np.array([float(lookup.get(name, 0.0)) for name in target_names], dtype=np.float64)


def classify(feature_vector: Sequence[float], feature_names: Sequence[str],
             baseline_stats: Optional[Dict] = None,
             model_dir: Optional[str] = None) -> Dict[str, Dict]:
    """Score a feature vector against each available condition model.

    Args:
        feature_vector, feature_names: output of
            fastmoda.pipeline.compute_feature_vector.
        baseline_stats: per-device baseline from storage.get_baseline(), or
            None/empty to fall back to global_feature_stats.json.
        model_dir: override MODA_DATA_DIR/models (mainly for tests).

    Returns:
        {condition: {'probability': float, 'top_features': [
            {'name', 'value', 'deviation', 'contribution'}, ...]}}
        Empty dict if no trained models are available.
    """
    model_dir = model_dir or storage.MODELS_DIR
    models = load_models(model_dir)
    if not models:
        return {}

    global_stats = load_global_stats(model_dir)
    stats = baseline_stats if (baseline_stats and baseline_stats.get('features')) else global_stats

    results = {}
    for condition, artifact in models.items():
        clf = artifact['model']
        target_names = artifact['feature_names']

        aligned_vector = _align(feature_vector, feature_names, target_names)
        deviation = compute_deviation(aligned_vector, target_names, stats)

        probability = float(clf.predict_proba(deviation.reshape(1, -1))[0, 1])
        contributions = clf.coef_[0] * deviation

        top_indices = np.argsort(-np.abs(contributions))[:TOP_K]
        top_features = [
            {
                'name': target_names[i],
                'value': float(aligned_vector[i]),
                'deviation': float(deviation[i]),
                'contribution': float(contributions[i]),
            }
            for i in top_indices
        ]

        results[condition] = {'probability': probability, 'top_features': top_features}

    return results
