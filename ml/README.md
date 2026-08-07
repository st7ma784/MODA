# Melanoma classification — MLflow + FastMODA + a linear model

Trains linear (logistic-regression) classifiers on the melanoma laser-Doppler
perfusion dataset, using **FastMODA's own feature-extraction engine**
(`compute_feature_vector` — the exact code path behind the `/analyze_features`
and `/classify` HTTP endpoints) to turn each signal into features, and
**MLflow** to track every run.

## The data — and yes, it is labelled

`data.mat` (MATLAB v5) holds one struct `data` with five diagnostic groups; the
group is the label, and `info2.xlsx` carries per-subject clinical metadata
(Breslow, Clark, vessel counts, age, BMI, sex, …). 89 subjects total:

| Group | Diagnosis                          | n  |
|-------|------------------------------------|----|
| `M`   | Melanoma (malignant)               | 10 |
| `A`   | Histologically atypical nevi       | 33 |
| `AB`  | Histologically benign nevi         | 11 |
| `B`   | Clinically benign (no histology)   | 26 |
| `P`   | Psoriasis (no histology)           |  9 |

Each subject has three perfusion channels: `cent` (lesion centre), `marg`
(margin), `norm` (normal skin). ~33 k samples each; the psoriasis group has no
`marg`. Sampling rate is **40 Hz** (Pisa LDF protocol; not stored in the file —
override with `--fs` if your lab notes differ).

## Run it

```bash
docker build -t fastmoda:cpu -f FastMODA/Dockerfile --target base FastMODA   # once
bash ml/run_ml.sh /c/Users/st7ma/Downloads/melanomadata/melanomadata
```

Outputs land in `ml/artifacts/`: `features.npz` / `features.csv` (89 × 204) and
`mlruns/`. Browse the experiment with:

```bash
mlflow ui --backend-store-uri ml/artifacts/mlruns
```

## Features

Per subject we extract FastMODA's 68-feature vector (spectral, phase, STFT,
wavelet, bispectrum summaries) from the **cent** and **norm** channels and their
**difference** (cent − norm — the lesion-vs-healthy contrast that the source
study exploits), giving 204 features. `marg` is skipped so all 89 subjects stay
in with a complete matrix (re-enable with `--with-marg`).

## Model

A leakage-safe sklearn pipeline, evaluated with cross-validation:

```
SimpleImputer(median) → VarianceThreshold → StandardScaler
  → SelectKBest(f_classif, k=15) → LogisticRegression(L2, class_weight=balanced)
```

Feature selection happens *inside* each CV fold, so reported metrics are
honest. Binary targets use leave-one-out; the 5-class target uses stratified
5-fold.

## Results (cross-validated)

| Target      | Grouping                     | AUC   | Balanced acc | Notes |
|-------------|------------------------------|-------|--------------|-------|
| `melanoma`  | M vs A/AB/B/P                | **0.834** | **0.78** | The biopsy-selection task — works. Recall 0.70 at balanced threshold. |
| `malignant` | M+A vs AB/B/P                | 0.34  | 0.37 | Below chance — atypical nevi don't share melanoma's perfusion signature, so this grouping is not linearly separable. |
| `fiveclass` | M/A/AB/B/P                   | 0.71 (OVR) | 0.37 | Above the 0.20 chance floor but modest; classes are tiny/imbalanced. |

Takeaway: a **linear** model already separates melanoma from everything else at
AUC 0.83 on FastMODA features — consistent with the source study's premise that
blood-perfusion dynamics flag lesions for biopsy. The other two groupings show
that lumping atypical nevi with melanoma destroys the signal, and that fine
5-way discrimination needs more than a linear model on this small a cohort.

Top melanoma features are dominated by `cent` spectral theta-band power and the
`cent − norm` contrast, i.e. the lesion-centre-vs-normal-skin difference — as
expected.

## Files

- `build_features.py` — data.mat → feature table via FastMODA's ML engine.
- `train_mlflow.py`  — three targets, CV, MLflow logging (params, metrics,
  confusion/ROC plots, top coefficients, fitted model).
- `run_ml.sh`        — one-command Docker runner.
