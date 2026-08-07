"""MLflow experiment: linear (logistic-regression) classification of melanoma
laser-Doppler recordings, on features produced by FastMODA's ML engine.

Runs three targets (per user request — "try all 3 and see what works"):
  1. melanoma   : Melanoma (M)      vs rest            — biopsy-selection task
  2. malignant  : M + atypical (A)  vs benign+psoriasis (AB/B/P)
  3. fiveclass  : M / A / AB / B / P                    — full multiclass

Model is a leakage-safe sklearn Pipeline evaluated with cross-validation:
    SimpleImputer(median) → VarianceThreshold → StandardScaler
    → SelectKBest(f_classif, k) → LogisticRegression(L2, class_weight=balanced)

Each target becomes one MLflow run under experiment "melanoma-fastmoda":
params, metrics, confusion-matrix / ROC plots, top model coefficients, and the
fitted sklearn model are all logged.

    mlflow ui --backend-store-uri <artifacts>/mlruns     # to browse afterwards
"""

import argparse
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report,
    roc_curve,
)

import mlflow
import mlflow.sklearn

GROUP_ORDER = ["M", "A", "AB", "B", "P"]

TARGETS = {
    "melanoma":  lambda g: (g == "M").astype(int),
    "malignant": lambda g: np.isin(g, ["M", "A"]).astype(int),
    "fiveclass": lambda g: g,   # raw labels
}


def make_pipeline(k, C, multiclass):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        # lbfgs uses multinomial loss automatically for multiclass targets
        ("clf", LogisticRegression(
            penalty="l2", C=C, class_weight="balanced",
            max_iter=5000, solver="lbfgs")),
    ])


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def plot_roc(y, score, auc, path):
    fpr, tpr, _ = roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (leave-one-out)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_target(name, X, y_raw, feat_names, k, C, tmp):
    multiclass = (name == "fiveclass")
    y = TARGETS[name](y_raw)
    k_eff = int(min(k, X.shape[1]))
    pipe = make_pipeline(k_eff, C, multiclass)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "target": name, "model": "LogisticRegression(L2)",
            "C": C, "k_features": k_eff, "n_raw_features": X.shape[1],
            "n_subjects": X.shape[0], "class_weight": "balanced",
            "cv": "LOO" if not multiclass else "StratifiedKFold(5)",
        })
        metrics = {}

        if multiclass:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
            labels = GROUP_ORDER
            # predict_proba columns follow the classifier's sorted class order,
            # so the AUC call must use that same ordering (not GROUP_ORDER).
            proba_labels = sorted(np.unique(y).tolist())
            try:
                metrics["roc_auc_ovr_macro"] = roc_auc_score(
                    y, proba, multi_class="ovr", average="macro",
                    labels=proba_labels)
            except Exception as exc:
                print("  (multiclass AUC skipped:", exc, ")")
            metrics["accuracy"] = accuracy_score(y, y_pred)
            metrics["balanced_accuracy"] = balanced_accuracy_score(y, y_pred)
            metrics["f1_macro"] = f1_score(y, y_pred, average="macro",
                                           labels=labels)
            cm = confusion_matrix(y, y_pred, labels=labels)
            report = classification_report(y, y_pred, labels=labels, digits=3)
        else:
            loo = LeaveOneOut()
            y_pred = cross_val_predict(pipe, X, y, cv=loo)
            score = cross_val_predict(pipe, X, y, cv=loo,
                                      method="decision_function")
            metrics["roc_auc"] = roc_auc_score(y, score)
            metrics["accuracy"] = accuracy_score(y, y_pred)
            metrics["balanced_accuracy"] = balanced_accuracy_score(y, y_pred)
            metrics["f1_pos"] = f1_score(y, y_pred, zero_division=0)
            metrics["precision_pos"] = precision_score(y, y_pred, zero_division=0)
            metrics["recall_pos"] = recall_score(y, y_pred, zero_division=0)
            metrics["n_positive"] = int(y.sum())
            labels = ["neg", "pos"]
            cm = confusion_matrix(y, y_pred)
            report = classification_report(y, y_pred, digits=3, zero_division=0)
            roc_png = os.path.join(tmp, f"{name}_roc.png")
            plot_roc(y, score, metrics["roc_auc"], roc_png)
            mlflow.log_artifact(roc_png)

        mlflow.log_metrics({k_: float(v) for k_, v in metrics.items()})

        cm_png = os.path.join(tmp, f"{name}_cm.png")
        plot_confusion(cm, labels, f"{name} confusion", cm_png)
        mlflow.log_artifact(cm_png)

        rep_txt = os.path.join(tmp, f"{name}_report.txt")
        with open(rep_txt, "w") as fh:
            fh.write(report)
        mlflow.log_artifact(rep_txt)

        # Fit on all data → log model + inspect which features the linear model
        # leaned on (selected + standardised coefficients).
        pipe.fit(X, y)
        support = pipe.named_steps["select"].get_support()
        var_support = pipe.named_steps["var"].get_support()
        kept = np.array(feat_names)[var_support][support]
        coef = pipe.named_steps["clf"].coef_
        coef_abs = np.abs(coef).mean(axis=0) if coef.ndim == 2 and coef.shape[0] > 1 \
            else np.abs(coef).ravel()
        order = np.argsort(coef_abs)[::-1]
        coef_csv = os.path.join(tmp, f"{name}_top_coefficients.csv")
        with open(coef_csv, "w") as fh:
            fh.write("feature,abs_coef\n")
            for idx in order[:20]:
                fh.write(f"{kept[idx]},{coef_abs[idx]:.5f}\n")
        mlflow.log_artifact(coef_csv)
        # cloudpickle avoids the skops "untrusted types" audit that rejects
        # SelectKBest(f_classif) under MLflow's default sklearn serializer.
        mlflow.sklearn.log_model(pipe, name="model",
                                 serialization_format="cloudpickle")

        print(f"\n=== {name} ===")
        for k_, v in metrics.items():
            print(f"  {k_:22s} {v:.3f}")
        top = [kept[i] for i in order[:6]]
        print("  top features:", ", ".join(top))
        return name, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/artifacts/features.npz")
    ap.add_argument("--tracking", default="file:/artifacts/mlruns")
    ap.add_argument("--experiment", default="melanoma-fastmoda")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--C", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, feat_names, groups = d["X"], d["feature_names"], d["groups"]
    print(f"Loaded {X.shape[0]} subjects × {X.shape[1]} features")
    uniq, cnt = np.unique(groups, return_counts=True)
    print("Class balance:", dict(zip(uniq.tolist(), cnt.tolist())))

    mlflow.set_tracking_uri(args.tracking)
    mlflow.set_experiment(args.experiment)

    tmp = tempfile.mkdtemp()
    results = []
    for name in TARGETS:
        results.append(run_target(name, X, groups, feat_names,
                                   args.k, args.C, tmp))

    print("\n" + "=" * 60)
    print("SUMMARY (cross-validated)")
    print("=" * 60)
    for name, m in results:
        head = m.get("roc_auc", m.get("roc_auc_ovr_macro", float("nan")))
        print(f"  {name:10s}  AUC={head:.3f}  "
              f"bal_acc={m['balanced_accuracy']:.3f}  "
              f"acc={m['accuracy']:.3f}")
    print(f"\nBrowse: mlflow ui --backend-store-uri {args.tracking}")


if __name__ == "__main__":
    main()
