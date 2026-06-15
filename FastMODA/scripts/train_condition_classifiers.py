#!/usr/bin/env python
"""Train per-condition logistic-regression classifier heads.

Reads the synthetic dataset produced by generate_cardiac_dataset.py, computes
the shared feature vector (fastmoda.pipeline.compute_feature_vector) for every
sample, builds one binary one-vs-rest label per condition (normal,
bradycardia, afib, pvc), z-scores features against the dataset's global
mean/std, and trains an L1-regularized LogisticRegression per condition.

Outputs (under --output-dir, default ${MODA_DATA_DIR}/models):
    <condition>.joblib         - {'model': clf, 'feature_names': [...]}
    global_feature_stats.json  - {'feature_names', 'mean', 'std'} fallback
                                  baseline used when a device has no
                                  calibrated baseline yet
    meta.json                  - {'feature_names', 'conditions'}

Usage:
    python scripts/train_condition_classifiers.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmoda.pipeline import compute_feature_vector  # noqa: E402
from fastmoda import storage  # noqa: E402

# Maps the fine-grained dataset condition (sub-folder name) to the coarser
# label a classifier head is trained against.
CONDITION_LABELS = {
    'normal': 'normal',
    'bradycardia': 'bradycardia',
    'mild_afib': 'afib',
    'moderate_afib': 'afib',
    'severe_afib': 'afib',
    'pvc_bigeminy': 'pvc',
    'pvc_trigeminy': 'pvc',
}
CONDITIONS = sorted(set(CONDITION_LABELS.values()))


def load_dataset(dataset_dir):
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    with open(metadata_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'No rows in {metadata_path} - run generate_cardiac_dataset.py first')

    feature_names = None
    X = []
    y_by_condition = {c: [] for c in CONDITIONS}
    for row in rows:
        filepath = os.path.join(dataset_dir, row['filepath'])
        x = np.load(filepath)
        fs = float(row['fs'])
        vector, names = compute_feature_vector(x, fs)
        if feature_names is None:
            feature_names = names
        X.append(vector)

        label = CONDITION_LABELS[row['condition']]
        for condition in CONDITIONS:
            y_by_condition[condition].append(1 if condition == label else 0)

    X = np.asarray(X, dtype=np.float64)
    y_by_condition = {c: np.asarray(v, dtype=np.int64) for c, v in y_by_condition.items()}
    return X, y_by_condition, feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cardiac_dataset')
    parser.add_argument('--dataset-dir', default=default_dataset,
                         help='Directory containing metadata.csv (default: %(default)s)')
    parser.add_argument('--output-dir', default=None,
                         help='Where to write model artifacts (default: MODA_DATA_DIR/models)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else storage.MODELS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading dataset from {dataset_dir} ...')
    X, y_by_condition, feature_names = load_dataset(dataset_dir)
    print(f'Loaded {X.shape[0]} samples x {X.shape[1]} features')

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    X_z = (X - mean) / std_safe

    for condition in CONDITIONS:
        y = y_by_condition[condition]
        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_z, y, test_size=args.test_size, random_state=args.seed, stratify=stratify)

        clf = LogisticRegression(penalty='l1', solver='liblinear',
                                  class_weight='balanced', max_iter=2000)
        clf.fit(X_train, y_train)

        if len(np.unique(y_test)) > 1:
            report = classification_report(y_test, clf.predict(X_test), zero_division=0)
            print(f'\n=== {condition} ===\n{report}')
        else:
            print(f'\n=== {condition} === (test split has a single class, skipping report)')

        joblib.dump({'model': clf, 'feature_names': feature_names},
                     os.path.join(output_dir, f'{condition}.joblib'))

    global_stats = {'feature_names': feature_names, 'mean': mean.tolist(), 'std': std.tolist()}
    with open(os.path.join(output_dir, 'global_feature_stats.json'), 'w') as f:
        json.dump(global_stats, f)

    meta = {'feature_names': feature_names, 'conditions': CONDITIONS}
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    print(f'\nSaved {len(CONDITIONS)} condition models + stats to {output_dir}')


if __name__ == '__main__':
    main()
