"""Per-patient baseline deviation computation.

Welford running mean/variance updates live in fastmoda.storage; this module
turns a feature vector + baseline stats into per-feature z-score deviations,
which are both the classifier input and the human-readable explanation
("phase coherence is 2.3 std below your normal").
"""

import numpy as np

EPS = 1e-8


def compute_deviation(feature_vector, feature_names, baseline_stats):
    """Return a z-score deviation vector for feature_vector against baseline_stats.

    Args:
        feature_vector: 1D array of raw feature values
        feature_names: list of feature names, same order as feature_vector
        baseline_stats: dict shaped like {'features': {name: {'mean', 'std'}}},
            as returned by fastmoda.storage.get_baseline() or the global
            feature stats produced during condition-model training. Features
            missing from baseline_stats deviate as 0 (no information yet).

    Returns:
        1D numpy array of per-feature z-score deviations, same length/order
        as feature_vector.
    """
    features = (baseline_stats or {}).get('features', {})
    deviation = np.zeros(len(feature_names), dtype=np.float64)
    for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
        stats = features.get(name)
        if not stats:
            continue
        std = max(float(stats.get('std', 0.0)), EPS)
        deviation[i] = (float(value) - float(stats.get('mean', 0.0))) / std
    return deviation
