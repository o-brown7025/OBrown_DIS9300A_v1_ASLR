# logistic_regression_classifier.py
#
# AGGREGATED FEATURES MODEL — Classical ML pathway
# Uses aggregate_time_series_long() from data_loaders.py.
# One row per video with mean/std/min/max/slope per landmark channel.
# Does NOT use raw time-series tensors.
#
# Configuration for 1,867-class ASL problem:
#   solver='saga'           — scales to large multi-class; lbfgs stalls
#   max_iter=3000           — saga needs more iterations to converge
#   class_weight='balanced' — rare signs get proportionally higher loss weight
#   C=0.1                   — stronger regularisation for many low-support classes
#   tol=1e-3                — relaxed tolerance; prevents endless iteration

import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loaders import aggregate_time_series_long, aggregate_with_pca
from evaluator import Evaluator


# ── Named wrapper so Evaluator saves files as "LogisticRegressionClassifier"
# Pipeline.__name__ returns "Pipeline" which causes confusing output filenames.
# This wrapper reports the correct class name while keeping the same interface.

class LogisticRegressionClassifier:
    """
    Thin wrapper around sklearn Pipeline (imputer → scaler → LogisticRegression).
    Defined at module top level so joblib and the Evaluator use the correct
    class name instead of the generic "Pipeline" label.
    """

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline

    def fit(self, X, y):
        self._pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self._pipeline.predict(X)

    def predict_proba(self, X):
        return self._pipeline.predict_proba(X)

    @property
    def classes_(self):
        return self._pipeline.classes_


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

def train_logistic_regression(
    train_csv,
    test_csv,
    results_dir: str  = "results/logistic_regression",
    force: bool       = False,
    use_masked: bool  = False,
    use_pca: bool     = False,   # True = apply PCA after aggregation
    pca_variance: float = 0.95,
):
    os.makedirs(results_dir, exist_ok=True)

    agg_train_path = os.path.join(results_dir, "agg_train.csv")
    agg_test_path  = os.path.join(results_dir, "agg_test.csv")

    model_path   = os.path.join(results_dir, "logistic_regression_model.pkl")
    encoder_path = os.path.join(results_dir, "label_encoder.pkl")

    eval_test_done = os.path.join(results_dir, ".EVAL_DONE_test")

    pca_bundle_path = os.path.join(results_dir, "pca_transformer.pkl")

    if use_pca:
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
        le.fit(y_train)
        y_train_enc = le.transform(y_train)

        print(f"[INFO] Training samples : {len(y_train)} | Test: {len(y_test)}")
        print(f"[INFO] Classes          : {len(le.classes_)}")
        print(f"[INFO] use_masked       : {use_masked}")
        print(f"[INFO] Solver           : saga")
        print(f"[INFO] class_weight     : balanced")
        print(f"[INFO] C                : 0.1")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler",  StandardScaler()),
            ("logreg",  LogisticRegression(
                solver="saga",
                multi_class="multinomial",
                max_iter=3000,
                tol=1e-3,
                C=0.1,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ])

        clf = LogisticRegressionClassifier(pipeline)
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

    print("[INFO] Logistic Regression pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate Logistic Regression on ASL dataset"
    )
    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--test_csv",    type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/logistic_regression")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--use_masked",  action="store_true")
    args = parser.parse_args()

    train_logistic_regression(
        args.train_csv,
        args.test_csv,
        results_dir=args.results_dir,
        force=args.force,
        use_masked=args.use_masked,
    )
