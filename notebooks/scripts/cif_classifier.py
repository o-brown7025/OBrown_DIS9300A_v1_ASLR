"""
cif_classifier.py
=================
Dissertation: Real-Time ASL Recognition
Author: Owasu Brown | National University | 2025-2026

CIF is a TIME-SERIES classifier.
- Accepts pre-built NPZ tensors (train_npz, test_npz) — no CSV conversion.
- No validation set — train/test split only.
- No val_csv parameter.
- Requires numba==0.62.1 + sktime (run in nb3_cif.ipynb only).

Configuration (literature-backed):
  n_estimators=50 : halves training and inference time vs default 100.
  n_features=8    : samples ~1/3 of catch22 features per interval.
    Middlehurst et al. (2020) showed this achieves within 1-2% of
    full-feature performance while cutting computation by ~65%.
    Reference: Middlehurst, M., Large, J. and Bagnall, A. (2020).
    IEEE International Conference on Big Data, pp.188-195.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sktime.classification.interval_based import CanonicalIntervalForest
from evaluator import Evaluator


def _write_marker(path):
    with open(path, "w") as f:
        f.write("ok\n")


def _impute_X(X):
    """Replace NaN with per-channel mean. Shape: [n_samples, n_channels, n_frames]"""
    X = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            s = X[i, c, :]
            mask = np.isnan(s)
            if mask.any() and not mask.all():
                s[mask] = float(np.nanmean(s))
            elif mask.all():
                s[:] = 0.0
    return X


def _filter_to_known(X, y, known_classes, split):
    known = set(map(str, known_classes))
    y_s   = pd.Series(y).astype(str)
    mask  = y_s.isin(known).to_numpy()
    dropped = int((~mask).sum())
    if dropped:
        print(f"[WARN] {split}: dropping {dropped} samples with unknown labels")
    return X[mask], y_s[mask].to_numpy()


def train_cif(
    train_npz:   str,
    test_npz:    str,
    results_dir: str  = "results/cif",
    force:       bool = False,
):
    """
    Train CanonicalIntervalForest using pre-built NPZ tensors.

    Parameters
    ----------
    train_npz   : path to ASL_reduced_train.npz or ASL_full_train.npz
    test_npz    : path to ASL_reduced_test.npz  or ASL_full_test.npz
    results_dir : folder for model and evaluation outputs
    force       : if True, retrain even if cached model exists
    """
    os.makedirs(results_dir, exist_ok=True)

    model_path     = os.path.join(results_dir, "cif_model.pkl")
    encoder_path   = os.path.join(results_dir, "label_encoder.pkl")
    eval_test_done = os.path.join(results_dir, ".EVAL_DONE_test")

    # ── Load pre-built tensors ────────────────────────────────────────────
    print(f"[INFO] Loading train NPZ : {train_npz}")
    d_train  = np.load(train_npz, allow_pickle=True)
    X_train, y_train = d_train["X"], d_train["y"]

    print(f"[INFO] Loading test NPZ  : {test_npz}")
    d_test   = np.load(test_npz, allow_pickle=True)
    X_test, y_test = d_test["X"], d_test["y"]

    # ── Train or load ─────────────────────────────────────────────────────
    if (not force) and os.path.exists(model_path) and os.path.exists(encoder_path):
        print(f"[SKIP] Loading cached CIF model from {results_dir}")
        clf = joblib.load(model_path)
        le  = joblib.load(encoder_path)
    else:
        le          = LabelEncoder()
        y_train_str = pd.Series(y_train).astype(str)
        y_train_enc = le.fit_transform(y_train_str)

        print(f"[INFO] Train samples : {len(y_train_str):,}")
        print(f"[INFO] Test samples  : {len(y_test):,}")
        print(f"[INFO] Classes       : {len(le.classes_):,}")
        print(f"[INFO] Tensor shape  : {X_train.shape}")
        print(f"[INFO] Classifier    : CanonicalIntervalForest")
        print(f"[INFO] n_estimators  : 50  (reduced from default 200 — 75% faster)")
        print(f"[INFO] att_subsample : 8   (catch22 features per interval, matches default)")
        print(f"[INFO] n_jobs        : -1  (all CPU cores)")

        X_train = _impute_X(X_train)
        X_test  = _impute_X(X_test)

        clf = CanonicalIntervalForest(
            n_estimators=50,
            att_subsample_size=8,  # catch22 features sampled per interval
            n_jobs=-1,             # use all available CPU cores
            random_state=42,
        )
        clf.fit(X_train, y_train_enc)

        joblib.dump(clf, model_path)
        joblib.dump(le,  encoder_path)
        print(f"[INFO] Model saved to {results_dir}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    X_test, y_test = _filter_to_known(X_test, y_test, le.classes_, "test")
    if len(y_test) == 0:
        raise ValueError("Test split empty after label filtering.")

    y_test_enc = le.transform(pd.Series(y_test).astype(str))
    evaluator  = Evaluator(output_dir=results_dir)

    if (not force) and os.path.exists(eval_test_done):
        print(f"[SKIP] Evaluation already done ({eval_test_done})")
    else:
        print("[INFO] Evaluating on test set...")
        evaluator.evaluate_and_save(
            model=clf, X=X_test, y_true=y_test_enc,
            label_encoder=le, dataset_name="test",
        )
        _write_marker(eval_test_done)

    print("[INFO] CIF pipeline complete!")
