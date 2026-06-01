# random_forest_classifier.py
#
# AGGREGATED FEATURES MODEL — Classical ML pathway
# Uses aggregate_time_series_long() from data_loaders.py.
# One row per video with mean/std/min/max/slope per landmark channel.
# Does NOT use raw time-series tensors.
#
# Configuration for 1,867-class ASL problem:
#   n_estimators=300      — stable ensemble estimate
#   max_features=0.3      — sample 30% of features per split; better than
#                           sqrt for high-dimensional aggregated landmarks
#   min_samples_leaf=1    — kept at 1; with only 2-3 videos per class,
#                           single samples must be allowed as leaf nodes
#   class_weight=None     — balanced_subsample was tested and hurt accuracy;
#                           with 2-3 videos per class it amplifies noise
#   n_jobs=-1             — use all CPU cores

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data_loaders import aggregate_time_series_long, aggregate_with_pca
from evaluator import Evaluator


# ── Helpers ────────────────────────────────────────────────────────────────

def _filter_to_known_labels(X, y, known_classes, split_name: str):
    known_set = set(map(str, known_classes))
    y = y.astype(str)
    mask = y.isin(known_set)
    dropped = int((~mask).sum())
    if dropped > 0:
        unseen  = sorted(y.loc[~mask].unique().tolist())
        preview = ", ".join(unseen[:10])
        if len(unseen) > 10:
            preview += ", ..."
        print(
            f"[WARN] {split_name}: dropping {dropped} row(s) with unseen "
            f"label(s) not present in training: {preview}"
        )
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _align_feature_columns(X, reference_columns, split_name: str):
    X = X.copy()
    missing = [c for c in reference_columns if c not in X.columns]
    extra   = [c for c in X.columns if c not in reference_columns]
    if missing:
        print(f"[WARN] {split_name}: adding {len(missing)} missing feature column(s) with zeros.")
        for col in missing:
            X[col] = 0.0
    if extra:
        print(f"[WARN] {split_name}: dropping {len(extra)} unexpected feature column(s).")
    return X.reindex(columns=reference_columns, fill_value=0.0)


def _write_marker(path: str):
    with open(path, "w") as f:
        f.write("ok\n")


def _load_or_aggregate(
    csv_path: str,
    cache_path: str,
    force: bool = False,
    use_masked: bool = False,
) -> pd.DataFrame:
    if (not force) and os.path.exists(cache_path):
        print(f"[SKIP] Using cached aggregated features: {cache_path}")
        return pd.read_csv(cache_path)
    print(f"[INFO] Aggregating features from: {csv_path} | use_masked={use_masked}")
    df = aggregate_time_series_long(csv_path, use_masked=use_masked)
    df.to_csv(cache_path, index=False)
    print(f"[INFO] Saved aggregated features: {cache_path}")
    return df


# ── Public training function ───────────────────────────────────────────────

def train_random_forest(
    train_csv,
    test_csv,
    results_dir: str  = "results/random_forest",
    force: bool       = False,
    use_masked: bool  = False,
    use_pca: bool     = False,   # True = apply PCA after aggregation
    pca_variance: float = 0.95,  # fraction of variance to retain
):
    os.makedirs(results_dir, exist_ok=True)

    agg_train_path = os.path.join(results_dir, "agg_train.csv")
    agg_test_path  = os.path.join(results_dir, "agg_test.csv")

    model_path   = os.path.join(results_dir, "random_forest_model.pkl")
    encoder_path = os.path.join(results_dir, "label_encoder.pkl")
    eval_test_done = os.path.join(results_dir, ".EVAL_DONE_test")

    pca_bundle_path = os.path.join(results_dir, "pca_transformer.pkl")

    if use_pca:
        # PCA pathway: aggregate then reduce dimensions
        print(f"[INFO] Aggregation + PCA pathway (pca_variance={pca_variance})")
        if (not force) and os.path.exists(agg_train_path) and os.path.exists(pca_bundle_path):
            print(f"[SKIP] Loading cached PCA-reduced features")
            df_train = pd.read_csv(agg_train_path)
        else:
            df_train, _ = aggregate_with_pca(
                train_csv, pca_bundle_path=pca_bundle_path,
                use_masked=use_masked, pca_variance=pca_variance,
            )
            df_train.to_csv(agg_train_path, index=False)

        if (not force) and os.path.exists(agg_test_path):
            df_test = pd.read_csv(agg_test_path)
        else:
            df_test, _ = aggregate_with_pca(
                test_csv, pca_bundle_path=pca_bundle_path,
                use_masked=use_masked,
            )
            df_test.to_csv(agg_test_path, index=False)
    else:
        # Standard pathway: aggregate only, no PCA
        df_train = _load_or_aggregate(train_csv, agg_train_path, force=force, use_masked=use_masked)
        df_test  = _load_or_aggregate(test_csv,  agg_test_path,  force=force, use_masked=use_masked)

    X_train = df_train.drop(columns=["video_name", "label"])
    y_train = df_train["label"].astype(str)
    X_test  = df_test.drop(columns=["video_name", "label"])
    y_test  = df_test["label"].astype(str)

    if not use_pca:
        feature_columns = X_train.columns.tolist()
        X_test = _align_feature_columns(X_test, feature_columns, "test")

    if (not force) and os.path.exists(model_path) and os.path.exists(encoder_path):
        print(f"[SKIP] Model and encoder already exist. Loading from {results_dir}")
        clf = joblib.load(model_path)
        le  = joblib.load(encoder_path)
    else:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        print(f"[INFO] Training samples : {len(y_train)} | Test: {len(y_test)}")
        print(f"[INFO] Classes          : {len(le.classes_)}")
        print(f"[INFO] use_masked       : {use_masked}")
        print(f"[INFO] n_estimators     : 300")
        print(f"[INFO] max_features     : 0.3")
        print(f"[INFO] class_weight     : None")

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            max_features=0.3,
            min_samples_leaf=1,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train_enc)

        joblib.dump(clf, model_path)
        joblib.dump(le,  encoder_path)
        print(f"[INFO] Model + encoder saved to {results_dir}")

    X_test, y_test = _filter_to_known_labels(X_test, y_test, le.classes_, "test")

    if len(y_test) == 0:
        raise ValueError("Test split empty after label filtering.")

    y_test_enc = le.transform(y_test)
    evaluator  = Evaluator(output_dir=results_dir)

    if (not force) and os.path.exists(eval_test_done):
        print(f"[SKIP] Test evaluation already completed ({eval_test_done})")
    else:
        print("[INFO] Evaluating on test set...")
        evaluator.evaluate_and_save(
            model=clf, X=X_test, y_true=y_test_enc,
            label_encoder=le, dataset_name="test",
        )
        _write_marker(eval_test_done)
        print(f"[DONE] Wrote marker: {eval_test_done}")

    print("[INFO] Random Forest pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest on ASL dataset"
    )
    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--test_csv",    type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/random_forest")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--use_masked",  action="store_true")
    args = parser.parse_args()

    train_random_forest(
        args.train_csv,
        args.test_csv,
        results_dir=args.results_dir,
        force=args.force,
        use_masked=args.use_masked,
    )
