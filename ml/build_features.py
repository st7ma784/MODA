"""Turn the melanoma laser-Doppler recordings into an ML feature table using
FastMODA's own feature-extraction engine (``compute_feature_vector`` — the exact
code path behind the ``/analyze_features`` and ``/classify`` HTTP endpoints).

Input : data.mat  (struct ``data`` with groups M/A/AB/B/P; each element has
                    channels cent/marg/norm — laser-Doppler perfusion signals)
        info2.xlsx (per-subject clinical metadata; optional, for reference)
Output: features.npz  (X, feature_names, group letters, subject ids)
        features.csv   (same, human-readable)

Design notes
------------
* fs = 40 Hz (Pisa LDF protocol; overridable with --fs).
* We use the **cent** (lesion centre) and **norm** (adjacent normal skin)
  channels — both present for all 89 subjects — and also their difference
  (cent − norm), which encodes the lesion-vs-healthy contrast that the source
  study (Rossi et al. 2015) exploits for biopsy selection.
* **marg** is deliberately skipped: the psoriasis (P) group has no margin
  channel, so including it would either drop 9 subjects or need imputing a whole
  channel. It can be re-enabled with --with-marg.
* Signals are optionally capped to --max-samples to bound runtime; the default
  keeps the full recording.
"""

import argparse
import os
import time

import numpy as np
import scipy.io as sio

from fastmoda.pipeline import compute_feature_vector

GROUPS = ["M", "A", "AB", "B", "P"]
GROUP_DIAGNOSIS = {
    "M": "Melanoma",
    "A": "Histologically atypical nevi",
    "AB": "Histologically benign nevi",
    "B": "Clinically benign (no histology)",
    "P": "Psoriasis (no histology)",
}


def _channel(rec, name):
    if name not in getattr(rec, "_fieldnames", []):
        return None
    a = np.asarray(getattr(rec, name)).squeeze().astype(np.float64)
    return a if a.ndim == 1 and a.size > 16 else None


def build(data_path, fs, max_samples, with_marg):
    m = sio.loadmat(data_path, struct_as_record=False, squeeze_me=False)
    root = m["data"].flatten()[0]

    channels = ["cent", "norm"] + (["marg"] if with_marg else [])
    rows_X, groups, subjects = [], [], []
    feat_names = None
    t0 = time.time()

    for g in GROUPS:
        recs = np.asarray(getattr(root, g)).flatten()
        for i, rec in enumerate(recs, 1):
            subj = f"{g}{i}"
            per_channel = {}
            ok = True
            for ch in channels:
                x = _channel(rec, ch)
                if x is None:
                    ok = False
                    break
                if max_samples:
                    x = x[:max_samples]
                vec, names = compute_feature_vector(x, fs)
                per_channel[ch] = np.asarray(vec, dtype=np.float64)
                if feat_names is None and ch == "cent":
                    base_names = list(names)
            if not ok:
                print(f"  skip {subj}: missing channel")
                continue

            # assemble: cent, norm, (cent-norm) [+ marg if requested]
            parts, names_out = [], []
            for ch in channels:
                parts.append(per_channel[ch])
                names_out += [f"{ch}_{n}" for n in base_names]
            diff = per_channel["cent"] - per_channel["norm"]
            parts.append(diff)
            names_out += [f"centMinusNorm_{n}" for n in base_names]

            rows_X.append(np.concatenate(parts))
            groups.append(g)
            subjects.append(subj)
            feat_names = names_out
            print(f"  {subj:5s} ({GROUP_DIAGNOSIS[g]:34s}) "
                  f"features={rows_X[-1].shape[0]}  [{time.time()-t0:5.1f}s]")

    X = np.vstack(rows_X)
    # NaN/Inf → will be imputed downstream; record where they were
    X = np.where(np.isfinite(X), X, np.nan)
    return X, np.array(feat_names), np.array(groups), np.array(subjects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/data.mat")
    ap.add_argument("--out", default="/artifacts/features.npz")
    ap.add_argument("--fs", type=float, default=40.0)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = use full recording")
    ap.add_argument("--with-marg", action="store_true")
    args = ap.parse_args()

    X, names, groups, subjects = build(
        args.data, args.fs, args.max_samples or None, args.with_marg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, X=X, feature_names=names, groups=groups,
             subjects=subjects, fs=args.fs)

    # human-readable CSV
    csv = os.path.splitext(args.out)[0] + ".csv"
    with open(csv, "w") as fh:
        fh.write("subject,group," + ",".join(names) + "\n")
        for s, g, row in zip(subjects, groups, X):
            fh.write(f"{s},{g}," + ",".join(f"{v:.6g}" for v in row) + "\n")

    print(f"\nWrote {X.shape[0]} subjects × {X.shape[1]} features → {args.out}")
    uniq, cnt = np.unique(groups, return_counts=True)
    print("Class balance:", dict(zip(uniq.tolist(), cnt.tolist())))
    print(f"NaN cells: {int(np.isnan(X).sum())}/{X.size}")


if __name__ == "__main__":
    main()
