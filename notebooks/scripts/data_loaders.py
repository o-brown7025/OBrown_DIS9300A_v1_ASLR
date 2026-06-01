"""
data_loaders.py
===============
Loads ASL landmark CSVs and prepares data for two distinct model pathways.

─────────────────────────────────────────────────────────────────────────────
PATHWAY 1 — Classical ML (Random Forest, Logistic Regression)
─────────────────────────────────────────────────────────────────────────────
Function : aggregate_time_series_long()
Input    : Long-format CSV  (one row per landmark per frame)
Output   : Wide DataFrame   (one row per video, aggregated features)
Method   : Collapses the time dimension using summary statistics
           (mean, std, min, max, slope) per landmark channel per video.
           These models CANNOT model temporal order directly — they require
           a fixed-width feature vector. The aggregation discards frame
           ordering but preserves the statistical shape of each sign's motion.

─────────────────────────────────────────────────────────────────────────────
PATHWAY 2 — Time-Series Models (CIF, InceptionTime)
─────────────────────────────────────────────────────────────────────────────
Function : load_long_csv_as_timeseries()
Input    : Long-format CSV  (one row per landmark per frame)
Output   : 3-D NumPy tensor (n_samples, n_channels, n_frames)
Method   : Preserves the full temporal sequence for each video.
           CIF extracts interval-based features directly from the ordered
           sequence. InceptionTime learns temporal patterns via 1-D
           convolutions across the frame axis.
           Frame ORDER IS PRESERVED — no aggregation is performed.

─────────────────────────────────────────────────────────────────────────────
Expected CSV schema
─────────────────────────────────────────────────────────────────────────────
filename  gloss  frame  hand_index  landmark  x  y  z
x_missing  y_missing  z_missing

hand_index values:
  0 = left hand  (21 landmarks, IDs 0-20)
  1 = right hand (21 landmarks, IDs 0-20)
  2 = pose / face position (11 landmarks, IDs 0-10)
  3 = spurious third hand detection (retained, not dropped)
"""

import numpy as np
import pandas as pd

# ── Column groups ──────────────────────────────────────────────────────────
_COORD_COLS = ["x", "y", "z"]

# Only the three coordinate mask columns — frame/hand_index/landmark masks
# are not present in the current dataset and are not needed.
_MASK_COLS  = ["x_missing", "y_missing", "z_missing"]

_META_COLS  = ["filename", "gloss", "frame", "hand_index", "landmark"]


# ── Schema helpers ─────────────────────────────────────────────────────────

def standardize_long_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw CSV columns to pipeline-internal names.
        filename → video_name
        gloss    → label
    """
    df = df.copy()
    if "filename" in df.columns and "video_name" not in df.columns:
        df = df.rename(columns={"filename": "video_name"})
    if "gloss" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"gloss": "label"})
    return df


def apply_missing_mask(df: pd.DataFrame, use_masked: bool = True) -> pd.DataFrame:
    """
    Handle missing-indicator columns.

    use_masked=True  → keep x_missing / y_missing / z_missing as features.
    use_masked=False → drop them; only raw coordinates are used.

    In both cases NaN coordinate values are zero-filled.
    """
    df = df.copy()
    present_mask_cols = [c for c in _MASK_COLS if c in df.columns]

    if not use_masked and present_mask_cols:
        df = df.drop(columns=present_mask_cols)

    # Zero-fill remaining NaNs in coordinate columns
    for col in _COORD_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


# ── Wide-frame conversion (shared by both pathways) ────────────────────────

def long_to_wide_frames(
    df_long: pd.DataFrame,
    video_col: str = "video_name",
    label_col: str = "label",
    time_col:  str = "frame",
    hand_col:  str = "hand_index",
    lm_col:    str = "landmark",
) -> pd.DataFrame:
    """
    Convert long-format landmarks into wide per-frame format.

    Input  : one row per (video, frame, hand, landmark, coord)
    Output : one row per (video, frame) with columns h{hand}_l{lm}_{coord}

    x and y are required. z is optional — it is absent in the reduced
    dataset where Z was intentionally dropped before training.
    Any *_missing columns present are also pivoted as channels.
    All channel values are cast to float32 to prevent TypeError in aggregation.
    """
    df = df_long.copy()

    # z is intentionally optional — dropped in the 243-class reduced dataset
    required = {video_col, label_col, time_col, hand_col, lm_col, "x", "y"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"long_to_wide_frames: missing required columns {missing_req}")

    # Pivot x and y always; z only if present; plus any *_missing columns
    coord_like = ["x", "y"] + (["z"] if "z" in df.columns else []) + [
        c for c in df.columns if c.endswith("_missing")
    ]

    stacked_parts = []
    for coord in coord_like:
        if coord not in df.columns:
            continue

        prefix = (
            "h" + df[hand_col].astype(str)
            + "_l" + df[lm_col].astype(str)
            + f"_{coord}"
        )

        part = df[[video_col, label_col, time_col]].copy()
        part["channel"] = prefix
        # Cast to float32 immediately — prevents string columns in the pivot
        part["value"]   = pd.to_numeric(df[coord].values, errors="coerce").astype("float32")
        stacked_parts.append(part)

    stacked = pd.concat(stacked_parts, axis=0, ignore_index=True)

    wide = stacked.pivot_table(
        index=[video_col, label_col, time_col],
        columns="channel",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Ensure every non-meta column is numeric float32
    meta = {video_col, label_col, time_col}
    for col in wide.columns:
        if col not in meta:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")

    return wide


# ── PATHWAY 1: Classical ML (RF / LR) — aggregated features ───────────────

def aggregate_time_series_long(
    csv_path:  str,
    video_col: str  = "video_name",
    label_col: str  = "label",
    time_col:  str  = "frame",
    use_masked: bool = True,
) -> pd.DataFrame:
    """
    Load long-format CSV and return one aggregated feature row per video.

    Used by: Random Forest, Logistic Regression.
    NOT used by: CIF, InceptionTime (they use load_long_csv_as_timeseries).

    Aggregation statistics computed per landmark channel per video:
        mean  — average position (where the landmark usually is)
        std   — variability / extent of movement
        min   — lower bound of motion range
        max   — upper bound of motion range
        slope — average per-frame rate of change (motion direction)

    Returns
    -------
    pd.DataFrame — one row per video, columns:
        video_name, label, h{hand}_l{lm}_{coord}_mean, ..._std, ..._min,
        ..._max, ..._slope
    """
    df_long = pd.read_csv(csv_path)
    df_long = standardize_long_schema(df_long)
    df_long = apply_missing_mask(df_long, use_masked=use_masked)

    df_wide = long_to_wide_frames(
        df_long,
        video_col=video_col,
        label_col=label_col,
        time_col=time_col,
    )

    df_wide = df_wide.sort_values([video_col, label_col, time_col])

    excluded     = {video_col, label_col, time_col}
    feature_cols = [c for c in df_wide.columns if c not in excluded]

    # Enforce numeric dtype on every feature column before aggregating.
    # This prevents the TypeError that occurs when string values survive
    # the pivot (e.g. from unexpected non-numeric channel values).
    for col in feature_cols:
        df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce").fillna(0.0)

    grouped = df_wide.groupby([video_col, label_col], sort=False)

    agg_df = grouped[feature_cols].agg(["mean", "std", "min", "max"])
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]

    def _slope(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float((series.iloc[-1] - series.iloc[0]) / (len(series) - 1))

    slope_df = grouped[feature_cols].agg(_slope)
    slope_df.columns = [f"{col}_slope" for col in slope_df.columns]

    final = pd.concat([agg_df, slope_df], axis=1).reset_index()
    final = final.fillna(0)
    return final


# ── PATHWAY 2: Time-series models (CIF / InceptionTime) — 3-D tensor ──────

def load_long_csv_as_timeseries(
    csv_path:  str,
    video_col: str  = "video_name",
    label_col: str  = "label",
    time_col:  str  = "frame",
    use_masked: bool = True,
):
    """
    Load long-format CSV and return a 3-D NumPy tensor preserving temporal order.

    Used by: CIF (CanonicalIntervalForest), InceptionTime.
    NOT used by: Random Forest, Logistic Regression (they use aggregate_time_series_long).

    Frame order is fully preserved. No statistical summarisation is applied.
    CIF operates on the ordered sequence using interval-based feature extraction.
    InceptionTime applies 1-D convolutions across the frame (time) axis.

    Returns
    -------
    X           : np.ndarray, shape (n_samples, n_channels, max_series_len)
                  Zero-padded to the length of the longest video in the batch.
    y           : np.ndarray of str labels, shape (n_samples,)
    video_names : list of str
    """
    df_long = pd.read_csv(csv_path)
    df_long = standardize_long_schema(df_long)
    df_long = apply_missing_mask(df_long, use_masked=use_masked)

    df_wide = long_to_wide_frames(
        df_long,
        video_col=video_col,
        label_col=label_col,
        time_col=time_col,
    )

    # Sort so frames within each video are in ascending order
    df_wide = df_wide.sort_values([video_col, label_col, time_col])

    excluded     = {video_col, label_col, time_col}
    channel_cols = sorted([c for c in df_wide.columns if c not in excluded])

    X_list, y_list, names = [], [], []

    for (vid, lab), sub in df_wide.groupby([video_col, label_col], sort=False):
        # sub rows are ordered by frame (sort above ensures this)
        mat = sub[channel_cols].to_numpy(dtype=np.float32)  # [n_frames, n_channels]
        X_list.append(mat.T)                                 # → [n_channels, n_frames]
        y_list.append(lab)
        names.append(vid)

    if not X_list:
        raise ValueError(
            f"No samples found in {csv_path}. Check file path and CSV schema."
        )

    max_len    = max(arr.shape[1] for arr in X_list)
    n_channels = X_list[0].shape[0]

    # Zero-pad shorter videos so all tensors share the same time dimension
    X = np.zeros((len(X_list), n_channels, max_len), dtype=np.float32)
    for i, arr in enumerate(X_list):
        X[i, :, : arr.shape[1]] = arr

    y = np.array(y_list)
    return X, y, names


# ── PATHWAY 1 extension: PCA-reduced aggregated features ──────────────────

def aggregate_with_pca(
    csv_path:       str,
    pca_bundle_path: str  = None,
    use_masked:     bool  = True,
    pca_variance:   float = 0.95,
    random_state:   int   = 42,
):
    """
    Aggregate features then apply PCA dimensionality reduction.

    Used by: Random Forest, Logistic Regression (full 1,867-class pipeline).
    NOT used by: CIF, InceptionTime.

    If pca_bundle_path is provided and exists, loads the pre-fitted PCA
    (transform only — no refitting). This ensures test data is projected
    using the same PCA fitted on training data, preventing leakage.

    If pca_bundle_path is None or does not exist, fits PCA on this data
    (use for training set only).

    Returns
    -------
    df_pca     : pd.DataFrame — video_name, label, PC1, PC2, ...
    pca_bundle : dict with keys: imputer, scaler, pca, feature_cols, n_components
    """
    import os
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    df_agg = aggregate_time_series_long(csv_path, use_masked=use_masked)
    meta_cols    = ["video_name", "label"]
    feature_cols = [c for c in df_agg.columns if c not in meta_cols]
    X            = df_agg[feature_cols].values

    if pca_bundle_path and os.path.exists(pca_bundle_path):
        # Transform only — load pre-fitted components
        bundle  = joblib.load(pca_bundle_path)
        imputer = bundle["imputer"]
        scaler  = bundle["scaler"]
        pca     = bundle["pca"]
        X_imp   = imputer.transform(X)
        X_scl   = scaler.transform(X_imp)
        X_pca   = pca.transform(X_scl)
    else:
        # Fit on this data (training set)
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        scaler  = StandardScaler()
        pca     = PCA(n_components=pca_variance, random_state=random_state)
        X_imp   = imputer.fit_transform(X)
        X_scl   = scaler.fit_transform(X_imp)
        X_pca   = pca.fit_transform(X_scl)

        if pca_bundle_path:
            bundle = {
                "imputer"     : imputer,
                "scaler"      : scaler,
                "pca"         : pca,
                "feature_cols": feature_cols,
                "n_components": pca.n_components_,
            }
            joblib.dump(bundle, pca_bundle_path)
            print(f"[INFO] Saved PCA bundle: {pca_bundle_path}")

    n_components = X_pca.shape[1]
    pc_cols      = [f"PC{i+1}" for i in range(n_components)]
    df_pca       = pd.DataFrame(X_pca, columns=pc_cols)
    df_pca.insert(0, "label",      df_agg["label"].values)
    df_pca.insert(0, "video_name", df_agg["video_name"].values)

    print(f"[INFO] Features before PCA : {len(feature_cols):,}")
    print(f"[INFO] Components after PCA: {n_components}")

    bundle = {
        "imputer": imputer, "scaler": scaler, "pca": pca,
        "feature_cols": feature_cols, "n_components": n_components,
    }
    return df_pca, bundle
