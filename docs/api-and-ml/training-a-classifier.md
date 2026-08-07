# Training a Classifier

FastMODA ships an example end-to-end pipeline for training a simple linear classifier
on top of its extracted signal features — using synthetic cardiac signals as the
worked example. This page is a stub outlining that pipeline; a full narrated
walkthrough is planned.

## Pipeline overview

```
scripts/generate_cardiac_dataset.py   →  synthetic signals + metadata.csv
              ↓
fastmoda.pipeline.compute_feature_vector(x, fs)   →  shared feature vector per signal
              ↓
scripts/train_condition_classifiers.py   →  one-vs-rest LogisticRegression per condition
              ↓
fastmoda/condition_models.py   →  classify() at inference time, with explanations
```

### 1. Generate (or bring your own) labelled data

`scripts/generate_cardiac_dataset.py` synthesizes labelled signals across several
conditions (`normal`, `bradycardia`, `mild_afib`, `moderate_afib`, `severe_afib`,
`pvc_bigeminy`, `pvc_trigeminy`) and writes a `metadata.csv` describing each sample's
file path, condition, and sampling rate.

### 2. Extract a shared feature vector per signal

`fastmoda.pipeline.compute_feature_vector(signal, fs)` returns `(vector, names)` — the
same feature representation used across the codebase, so a model trained this way is
compatible with the live API's outputs. Conceptually this rolls up the same kind of
information described in [REST API Reference](rest-api-reference.md#post-analyze):
dominant frequencies, band powers, changepoint statistics, and so on, into one fixed-length
numeric vector.

### 3. Train per-condition logistic regression heads

`scripts/train_condition_classifiers.py`:

- loads `metadata.csv` and computes the feature vector for every sample,
- maps each fine-grained condition to a coarser label (e.g. `mild_afib` /
  `moderate_afib` / `severe_afib` → `afib`),
- z-scores features against the dataset's global mean/std,
- trains an **L1-regularized `sklearn.linear_model.LogisticRegression`** per condition
  (one-vs-rest), and
- writes `<condition>.joblib`, `global_feature_stats.json`, and `meta.json` under
  `${MODA_DATA_DIR}/models/`.

Run it with:

```bash
python scripts/train_condition_classifiers.py
```

### 4. Classify new signals

`fastmoda/condition_models.py` loads the persisted models and, given a raw feature
vector, returns a per-condition probability plus a top-K list of
`{name, value, deviation, contribution}` — where `contribution = coefficient *
deviation` — explaining *why* the model scored a signal the way it did, not just the
score itself. This is what lets a classification result be shown to a user with a
plain-language "this looked unusual because X, Y, Z" explanation rather than a bare
label.

## Bringing your own labels

Any signal set with known ground-truth labels can be substituted for
`generate_cardiac_dataset.py`'s synthetic output — the only requirement is a
`metadata.csv` with a `filepath`, `condition`, and `fs` column per sample, matching
what `train_condition_classifiers.py` expects. See
[Programmatic Usage](programmatic-usage.md) for extracting features from your own
signals via the REST API instead of calling `compute_feature_vector` directly in
Python.
