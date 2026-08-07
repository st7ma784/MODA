# Case Study: Melanoma Classification with MLflow

An end-to-end worked example that uses **FastMODA's feature-extraction engine**
as the front end to a classical machine-learning pipeline, tracked with
**MLflow**. It doubles as a validation that the `/analyze_features` /`/classify`
code path produces genuinely discriminative features on real clinical data.

Everything is reproducible via Docker:

```bash
docker build -t fastmoda:cpu -f FastMODA/Dockerfile --target base FastMODA
bash ml/run_ml.sh /c/Users/st7ma/Downloads/melanomadata/melanomadata
mlflow ui --backend-store-uri ml/artifacts/mlruns      # browse the runs
```

The scripts are `ml/build_features.py` and `ml/train_mlflow.py`.

---

## The dataset

Laser-Doppler flowmetry (LDF) blood-perfusion recordings of skin lesions
(Rossi et al., *Sci. Rep.* 2015). `data.mat` is a MATLAB v5 struct whose five
fields **are the class labels**; `info2.xlsx` carries per-subject clinical
metadata (Breslow thickness, Clark level, vessel counts, age, BMI, sex). 89
subjects:

| Group | Diagnosis | n |
|-------|-----------|---|
| `M` | Melanoma (malignant) | 10 |
| `A` | Histologically atypical nevi | 33 |
| `AB` | Histologically benign nevi | 11 |
| `B` | Clinically benign (no histology) | 26 |
| `P` | Psoriasis (no histology) | 9 |

Each subject has three perfusion channels — `cent` (lesion centre), `marg`
(margin), `norm` (normal skin), ~33 000 samples each. The `P` group has no
`marg`. Sampling rate is **40 Hz** (the Pisa LDF protocol; not stored in the
file, overridable via `--fs`).

---

## Feature extraction — via FastMODA

For each subject we call `fastmoda.pipeline.compute_feature_vector(x, fs)` — the
*exact* function behind the `/analyze_features` and `/classify` HTTP endpoints —
which returns a 68-dimensional summary (spectral, phase, STFT, wavelet, and
bispectrum statistics) per channel. We extract it from the **`cent`** and
**`norm`** channels and their **difference** (`cent − norm`, the
lesion-versus-healthy-skin contrast that the source study exploits for
biopsy selection), giving **204 features** per subject.

!!! note "Why `marg` is dropped by default"
    The psoriasis group has no margin channel, so including `marg` would either
    lose 9 subjects or require imputing a whole channel. Excluding it keeps all
    89 subjects with a complete matrix. Re-enable with `--with-marg`.

---

## Model

A leakage-safe scikit-learn pipeline, deliberately **linear**:

```
SimpleImputer(median) → VarianceThreshold → StandardScaler
  → SelectKBest(f_classif, k=15) → LogisticRegression(L2, class_weight=balanced)
```

Feature selection happens *inside* each cross-validation fold, so no information
leaks from test to train. Binary targets are scored by leave-one-out CV; the
five-class target by stratified 5-fold.

Three targets were evaluated (per request — "try all three and see what works"):

| Target | Grouping |
|--------|----------|
| `melanoma` | M vs A/AB/B/P (the biopsy-selection task) |
| `malignant` | M + A vs AB/B/P |
| `fiveclass` | M / A / AB / B / P |

---

## Results (cross-validated)

| Target | ROC-AUC | Balanced acc. | Verdict |
|--------|---------|---------------|---------|
| **`melanoma`** | **0.83** | **0.78** | A linear model already separates melanoma from everything else. Recall 0.70 at the balanced threshold. |
| `malignant` | 0.34 | 0.37 | **Below chance** — atypical nevi (`A`) do not share melanoma's perfusion signature, so lumping them with `M` destroys the boundary. |
| `fiveclass` | 0.71 (OVR macro) | 0.37 | Above the 0.20 chance floor but modest; the classes are small and imbalanced. |

**Takeaway.** The melanoma-vs-rest result (AUC 0.83) is consistent with the
source study's premise that blood-perfusion dynamics can flag lesions for biopsy
— and it is reached with nothing more than a *linear* model on FastMODA
features. The `malignant` failure is scientifically informative: it shows the
melanoma signal is specific to `M`, not shared with atypical nevi. Fine 5-way
discrimination would need a richer model and more subjects.

The most heavily weighted melanoma features are `cent` spectral theta-band power
and the `cent − norm` contrast — i.e. the lesion-centre-versus-normal-skin
difference, exactly as the biophysics predicts.

---

## What MLflow logs

Each target is one MLflow run under experiment `melanoma-fastmoda`, recording:
parameters (target, model, `C`, `k`, feature counts, CV scheme), metrics
(AUC, balanced accuracy, F1, precision/recall), and artifacts — confusion-matrix
and ROC plots, the top-20 model coefficients, a classification report, and the
fitted scikit-learn model (serialized with cloudpickle).

!!! warning "Environment notes"
    The FastMODA CPU image has no PyTorch, so the `*_gpu` feature functions run
    their scipy/numpy fallbacks — fully supported. MLflow is `pip install`-ed at
    runtime; its file store is written to the container filesystem and copied to
    the mounted `ml/artifacts/` afterwards (the file store misbehaves on Windows
    bind mounts). See `ml/README.md` for the full run recipe.
