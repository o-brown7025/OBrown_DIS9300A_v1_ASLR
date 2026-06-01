# comparison_evaluator.py
# ============================================================
# Dissertation: Real-Time ASL Recognition – Model Comparison
# Author: Owasu Brown | National University | 2025-2026
#
# CHANGES FROM PREVIOUS VERSION:
#   - Viridis-family colour palette (print-safe, accessible)
#   - All chart backgrounds white, no grid lines
#   - Radar chart keeps angular grid only (required for readability)
#   - Inference speed chart is the primary focus — log scale,
#     annotated clearly, error bars visible
#   - No overlapping titles, axis labels, or footnotes
#   - APA 7 spine style (top/right removed) on all Cartesian axes
#   - WEASEL+MUSE removed — four models only
#   - Viridis discrete: IT=#440154, RF=#31688e, LR=#35b779, CIF=#fde725
# ============================================================

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import joblib

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    sns = None
    _HAS_SNS = False

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    confusion_matrix, top_k_accuracy_score,
    roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ═══════════════════════════════════════════════════════════════
# CONSTANTS  —  four models only, Viridis discrete palette
# ═══════════════════════════════════════════════════════════════

MODEL_LABELS = {
    "inceptiontime":       "InceptionTime",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "cif":                 "CIF",
}

# Viridis discrete: sampled at 0, 0.33, 0.66, 1.0
MODEL_COLORS = {
    "inceptiontime":       "#440154",   # viridis dark purple
    "random_forest":       "#31688e",   # viridis blue
    "logistic_regression": "#35b779",   # viridis green
    "cif":                 "#fde725",   # viridis yellow
}

MODEL_MARKERS = {
    "inceptiontime":       "o",
    "random_forest":       "s",
    "logistic_regression": "^",
    "cif":                 "D",
}

MODEL_ORDER = [
    "inceptiontime", "random_forest", "logistic_regression", "cif"
]

TOP_N_MAX      = 10
N_SPEED_TRIALS = 50


# ═══════════════════════════════════════════════════════════════
# STYLE HELPERS
# ═══════════════════════════════════════════════════════════════

def _apply_apa_style(ax):
    """Remove top/right spines and all grid lines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")


def _savefig(fig, path: str, dpi: int = 150):
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    [SAVED] {os.path.basename(path)}")


def _model_pkl_size_mb(model_path: str) -> float:
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 ** 2)
    return float("nan")


# ═══════════════════════════════════════════════════════════════
# ARTIFACT LOADERS
# ═══════════════════════════════════════════════════════════════

def _load_predictions(split_dir, model_class_name, dataset_name):
    # Search all sensible path patterns — handles both old and new evaluator layouts
    candidates = [
        # New layout: results/<folder>/test/<ModelClass>_test_predictions.csv
        os.path.join(split_dir, dataset_name,
                     f"{model_class_name}_{dataset_name}_predictions.csv"),
        # Old layout: results/<folder>/test/<ModelClass>/<ModelClass>_predictions.csv
        os.path.join(split_dir, dataset_name, model_class_name,
                     f"{model_class_name}_predictions.csv"),
        # Fallback: walk the test directory for any predictions CSV
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            return df["y_true"].values, df["y_pred"].values
    # Walk fallback
    test_dir = os.path.join(split_dir, dataset_name)
    if os.path.exists(test_dir):
        for root, _, files in os.walk(test_dir):
            for fname in files:
                if fname.endswith("_predictions.csv"):
                    df = pd.read_csv(os.path.join(root, fname))
                    if "y_true" in df.columns and "y_pred" in df.columns:
                        return df["y_true"].values, df["y_pred"].values
    return None, None


def _load_proba(split_dir, model_class_name, dataset_name):
    candidates = [
        os.path.join(split_dir, dataset_name,
                     f"{model_class_name}_{dataset_name}_proba.npy"),
        os.path.join(split_dir, dataset_name, model_class_name,
                     f"{model_class_name}_{dataset_name}_proba.npy"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return np.load(candidate, allow_pickle=True)
    # Walk fallback
    test_dir = os.path.join(split_dir, dataset_name)
    if os.path.exists(test_dir):
        for root, _, files in os.walk(test_dir):
            for fname in files:
                if fname.endswith("_proba.npy"):
                    return np.load(os.path.join(root, fname),
                                   allow_pickle=True)
    return None


# ═══════════════════════════════════════════════════════════════
# INFERENCE SPEED BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark_inference_speed(model, X_sample, n_trials=N_SPEED_TRIALS,
                               model_key="model"):
    """Single-sample CPU inference latency over n_trials repeats."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    latencies_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    lat = np.array(latencies_ms)
    result = {
        "model":     model_key,
        "mean_ms":   float(np.mean(lat)),
        "std_ms":    float(np.std(lat)),
        "median_ms": float(np.median(lat)),
        "min_ms":    float(np.min(lat)),
        "max_ms":    float(np.max(lat)),
        "n_trials":  n_trials,
    }
    print(f"    [{model_key}] mean={result['mean_ms']:.2f} ms  "
          f"median={result['median_ms']:.2f} ms  "
          f"std={result['std_ms']:.2f} ms")
    return result


# ═══════════════════════════════════════════════════════════════
# INFERENCE SPEED CHART  (primary deliverable — log scale)
# ═══════════════════════════════════════════════════════════════

def plot_inference_speed(speed_df: pd.DataFrame, out_dir: str):
    """
    Horizontal bar chart of mean CPU inference latency.
    Log x-axis to show the large range between models clearly.
    Error bars show ±1 SD. Values annotated inside bars.
    Footnote explains log scale and trial count.
    """
    df = speed_df.copy()
    df["label"] = df["model"].map(MODEL_LABELS)
    df["color"] = df["model"].map(MODEL_COLORS)
    # Sort slowest→fastest so fastest is at top
    df = df.sort_values("mean_ms", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.18, top=0.88)

    y_pos = np.arange(len(df))
    bars  = ax.barh(
        y_pos,
        df["mean_ms"],
        xerr=df["std_ms"],
        color=df["color"].tolist(),
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
        alpha=0.92,
    )

    # Annotate each bar with mean ± std
    for i, row in df.iterrows():
        ax.text(
            row["mean_ms"] * 1.05,
            i,
            f"{row['mean_ms']:.1f} ms  (±{row['std_ms']:.1f})",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].tolist(), fontsize=10)
    ax.set_xlabel("Mean Inference Latency (ms, log scale)", fontsize=10,
                  labelpad=8)
    ax.set_title(
        "Single-Sample CPU Inference Speed — Four Classifiers",
        fontsize=12, pad=12, loc="left"
    )

    # Footnote below x-axis
    n = int(df["n_trials"].iloc[0]) if len(df) else N_SPEED_TRIALS
    fig.text(
        0.22, 0.03,
        f"Note. CPU-only benchmark (CUDA disabled). "
        f"n = {n} trials per model. "
        f"Error bars = ±1 SD. "
        f"Log scale used to accommodate large latency range across classifiers.",
        fontsize=7.5, ha="left", va="bottom", color="#444444"
    )

    _apply_apa_style(ax)
    # Keep left spine for reference on log scale
    ax.spines["left"].set_visible(True)

    _savefig(fig, os.path.join(out_dir, "inference_speed_comparison.png"))


# ═══════════════════════════════════════════════════════════════
# TOP-N ACCURACY CURVE
# ═══════════════════════════════════════════════════════════════

def plot_top_n_accuracy(model_probas, label_names, out_dir, n_max=TOP_N_MAX):
    n_classes = len(label_names)
    k_values  = list(range(1, n_max + 1))
    rows      = []

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(left=0.1, right=0.72, bottom=0.14, top=0.88)

    for model_key in MODEL_ORDER:
        if model_key not in model_probas:
            continue
        y_true, y_proba = model_probas[model_key]
        if y_proba is None:
            continue
        label  = MODEL_LABELS.get(model_key, model_key)
        color  = MODEL_COLORS[model_key]
        marker = MODEL_MARKERS[model_key]
        accs   = []
        for k in k_values:
            try:
                accs.append(top_k_accuracy_score(
                    y_true, y_proba, k=min(k, n_classes),
                    labels=range(n_classes)
                ))
            except Exception:
                accs.append(float("nan"))
            rows.append({"model": model_key, "k": k,
                          "top_k_accuracy": accs[-1]})
        ax.plot(k_values, accs, marker=marker, label=label,
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("k  (Top-k)", fontsize=10, labelpad=6)
    ax.set_ylabel("Accuracy", fontsize=10, labelpad=6)
    ax.set_title("Top-k Accuracy — All Classifiers",
                 fontsize=12, pad=10, loc="left")
    ax.set_xticks(k_values)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=False, fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    _apply_apa_style(ax)
    _savefig(fig, os.path.join(out_dir, "top_n_accuracy.png"))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "top_n_accuracy.csv"), index=False)


# ═══════════════════════════════════════════════════════════════
# ROC-AUC BAR CHART
# ═══════════════════════════════════════════════════════════════

def plot_roc_auc_comparison(model_probas, label_names, out_dir,
                             max_classes_for_roc=200):
    n_classes = len(label_names)
    rows      = []
    labels_plot, values, colors = [], [], []

    for model_key in MODEL_ORDER:
        if model_key not in model_probas:
            continue
        y_true, y_proba = model_probas[model_key]
        if y_proba is None:
            continue

        unique, counts = np.unique(y_true, return_counts=True)
        valid = unique[counts >= 2]
        if len(valid) > max_classes_for_roc:
            valid = np.sort(
                np.random.default_rng(42).choice(
                    valid, max_classes_for_roc, replace=False
                )
            )
        y_bin = label_binarize(y_true, classes=range(n_classes))
        try:
            auc = roc_auc_score(
                y_bin[:, valid], y_proba[:, valid],
                average="macro", multi_class="ovr"
            )
        except Exception:
            auc = float("nan")

        rows.append({"model": model_key, "roc_auc_macro": auc})
        labels_plot.append(MODEL_LABELS[model_key])
        values.append(auc)
        colors.append(MODEL_COLORS[model_key])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.2, top=0.88)
    x = np.arange(len(labels_plot))
    bars = ax.bar(x, values, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.92, width=0.55)

    for xi, v in zip(x, values):
        if not np.isnan(v):
            ax.text(xi, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro ROC-AUC", fontsize=10, labelpad=6)
    ax.set_title("Macro ROC-AUC Comparison",
                 fontsize=12, pad=10, loc="left")
    fig.text(0.12, 0.03,
             f"Note. One-vs-rest OvR ROC-AUC. "
             f"Sampled up to {max_classes_for_roc} classes with ≥2 test videos.",
             fontsize=7.5, ha="left", va="bottom", color="#444444")
    _apply_apa_style(ax)
    _savefig(fig, os.path.join(out_dir, "roc_auc_comparison.png"))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "roc_auc_summary.csv"), index=False)


# ═══════════════════════════════════════════════════════════════
# PRECISION-RECALL AP
# ═══════════════════════════════════════════════════════════════

def plot_pr_curve_comparison(model_probas, label_names, out_dir,
                              max_classes=200):
    n_classes = len(label_names)
    rows      = []
    labels_plot, values, colors = [], [], []

    for model_key in MODEL_ORDER:
        if model_key not in model_probas:
            continue
        y_true, y_proba = model_probas[model_key]
        if y_proba is None:
            continue

        unique, counts = np.unique(y_true, return_counts=True)
        valid = unique[counts >= 2]
        if len(valid) > max_classes:
            valid = np.sort(
                np.random.default_rng(42).choice(
                    valid, max_classes, replace=False
                )
            )
        y_bin = label_binarize(y_true, classes=range(n_classes))
        try:
            ap = average_precision_score(
                y_bin[:, valid], y_proba[:, valid], average="macro"
            )
        except Exception:
            ap = float("nan")

        rows.append({"model": model_key, "macro_avg_precision": ap})
        labels_plot.append(MODEL_LABELS[model_key])
        values.append(ap)
        colors.append(MODEL_COLORS[model_key])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.2, top=0.88)
    x = np.arange(len(labels_plot))
    ax.bar(x, values, color=colors, edgecolor="white",
           linewidth=0.5, alpha=0.92, width=0.55)
    for xi, v in zip(x, values):
        if not np.isnan(v):
            ax.text(xi, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro Average Precision (AP)", fontsize=10, labelpad=6)
    ax.set_title("Macro Precision-Recall AP Comparison",
                 fontsize=12, pad=10, loc="left")
    fig.text(0.12, 0.03,
             f"Note. Sampled up to {max_classes} classes with ≥2 test videos.",
             fontsize=7.5, ha="left", va="bottom", color="#444444")
    _apply_apa_style(ax)
    _savefig(fig, os.path.join(out_dir, "pr_ap_comparison.png"))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "pr_ap_summary.csv"), index=False)


# ═══════════════════════════════════════════════════════════════
# F1 DISTRIBUTION  (histogram, not boxplot — readable at scale)
# ═══════════════════════════════════════════════════════════════

def plot_f1_distribution_comparison(model_preds, label_names, out_dir):
    """
    Per-class F1 histogram for each model on the same axes.
    Histogram is more readable than boxplot at 1,683 classes because
    boxplots collapse to a flat line at zero with a few outliers.
    """
    n_classes = len(label_names)
    fig, axes = plt.subplots(
        1, len(MODEL_ORDER), figsize=(13, 4.5), sharey=True
    )
    fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97,
                        bottom=0.18, top=0.88)
    bins = np.linspace(0, 1, 21)

    for ax, model_key in zip(axes, MODEL_ORDER):
        if model_key not in model_preds:
            ax.set_visible(False)
            continue
        y_true, y_pred = model_preds[model_key]
        if y_true is None:
            ax.set_visible(False)
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(n_classes), zero_division=0
        )
        label = MODEL_LABELS[model_key]
        color = MODEL_COLORS[model_key]
        ax.hist(f1, bins=bins, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.90)
        ax.axvline(np.mean(f1), color="#333333", linestyle="--",
                   linewidth=1.2, label=f"Mean={np.mean(f1):.3f}")
        ax.set_title(label, fontsize=10, pad=6)
        ax.set_xlabel("F1 Score", fontsize=9, labelpad=5)
        ax.legend(frameon=False, fontsize=8)
        _apply_apa_style(ax)

    axes[0].set_ylabel("Number of Classes", fontsize=10, labelpad=6)
    fig.suptitle("Per-Class F1 Score Distribution — All Classifiers",
                 fontsize=12, y=0.97, x=0.04, ha="left")
    fig.text(0.07, 0.03,
             "Note. Each panel shows the distribution of F1 scores "
             "across all classes for one classifier. "
             "Dashed line = mean F1. Classes at F1 = 0 dominate "
             "because most signs appear only once in the test set.",
             fontsize=7.5, ha="left", va="bottom", color="#444444")
    _savefig(fig, os.path.join(out_dir, "f1_distribution_comparison.png"))


# ═══════════════════════════════════════════════════════════════
# CONFUSION MATRICES  (top-N by support, Viridis cmap)
# ═══════════════════════════════════════════════════════════════

def plot_normalised_confusion_matrices(model_preds, label_names, out_dir,
                                        top_n_classes=25):
    n_classes = len(label_names)
    for model_key in MODEL_ORDER:
        if model_key not in model_preds:
            continue
        y_true, y_pred = model_preds[model_key]
        if y_true is None:
            continue
        label = MODEL_LABELS[model_key]

        unique, counts = np.unique(y_true, return_counts=True)
        top_idx    = np.sort(unique[np.argsort(counts)[::-1][:top_n_classes]])
        top_labels = [label_names[i] for i in top_idx]

        mask = np.isin(y_true, top_idx)
        yt   = y_true[mask];  yp = y_pred[mask]
        remap = {orig: new for new, orig in enumerate(top_idx)}
        yt_r  = np.array([remap[v] for v in yt])
        yp_r  = np.array([remap.get(v, -1) for v in yp])
        valid = yp_r >= 0
        yt_r  = yt_r[valid];  yp_r = yp_r[valid]

        cm      = confusion_matrix(yt_r, yp_r, labels=range(len(top_idx)))
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm.astype(float) / row_sum

        sz  = max(10, len(top_idx) * 0.32)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
        fig.subplots_adjust(bottom=0.22, top=0.90, left=0.18, right=0.95)

        if _HAS_SNS:
            sns.heatmap(
                cm_norm, ax=ax, cmap="viridis",
                vmin=0, vmax=1,
                xticklabels=top_labels, yticklabels=top_labels,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Recall (row-normalised)",
                           "shrink": 0.8}
            )
        else:
            im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1,
                            aspect="auto")
            cb = plt.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label("Recall (row-normalised)", fontsize=9)
            ax.set_xticks(range(len(top_labels)))
            ax.set_xticklabels(top_labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(top_labels)))
            ax.set_yticklabels(top_labels, fontsize=7)

        ax.set_xlabel("Predicted Sign", fontsize=10, labelpad=8)
        ax.set_ylabel("True Sign", fontsize=10, labelpad=8)
        ax.set_title(
            f"Normalised Confusion Matrix — {label}\n"
            f"(Top {top_n_classes} most frequent classes)",
            fontsize=11, pad=10
        )
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        _savefig(fig, os.path.join(out_dir,
                 f"confusion_matrix_normalised_{model_key}.png"))

        # Save full raw matrix
        full_cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        np.save(os.path.join(out_dir,
                f"confusion_matrix_full_{model_key}.npy"), full_cm)


# ═══════════════════════════════════════════════════════════════
# WORST / BEST CLASSES
# ═══════════════════════════════════════════════════════════════

def analyse_class_performance(model_preds, label_names, out_dir, n=20):
    n_classes = len(label_names)
    f1_matrix = {}
    for model_key in MODEL_ORDER:
        if model_key not in model_preds:
            continue
        y_true, y_pred = model_preds[model_key]
        if y_true is None:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(n_classes), zero_division=0
        )
        f1_matrix[model_key] = f1

    if not f1_matrix:
        return

    df_f1 = pd.DataFrame(f1_matrix, index=label_names)
    df_f1["mean_f1"] = df_f1.mean(axis=1)
    df_f1 = df_f1.reset_index().rename(columns={"index": "gloss"})
    worst = df_f1.sort_values("mean_f1").head(n)
    best  = df_f1.sort_values("mean_f1", ascending=False).head(n)
    worst.to_csv(os.path.join(out_dir, "worst_classes.csv"), index=False)
    best.to_csv(os.path.join(out_dir, "best_classes.csv"),  index=False)

    width = 0.8 / max(len(f1_matrix), 1)

    for title_suffix, subset, fname in [
        (f"Bottom {n} Signs by Mean F1",  worst, "worst_classes_comparison.png"),
        (f"Top {n} Signs by Mean F1",     best,  "best_classes_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.subplots_adjust(bottom=0.30, top=0.88,
                            left=0.07, right=0.97)
        x = np.arange(len(subset))
        for i, model_key in enumerate(MODEL_ORDER):
            if model_key not in f1_matrix:
                continue
            ax.bar(x + i * width,
                   subset[model_key].values,
                   width,
                   label=MODEL_LABELS[model_key],
                   color=MODEL_COLORS[model_key],
                   alpha=0.90,
                   edgecolor="white",
                   linewidth=0.4)
        ax.set_xticks(x + width * (len(f1_matrix) - 1) / 2)
        ax.set_xticklabels(subset["gloss"].values,
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score", fontsize=10, labelpad=6)
        ax.set_title(title_suffix, fontsize=12, pad=10, loc="left")
        ax.legend(frameon=False, fontsize=9, ncol=2,
                  bbox_to_anchor=(1, 1), loc="upper left")
        _apply_apa_style(ax)
        _savefig(fig, os.path.join(out_dir, fname))


# ═══════════════════════════════════════════════════════════════
# RADAR / SPIDER CHART
# ═══════════════════════════════════════════════════════════════

def plot_radar_chart(summary_df: pd.DataFrame, out_dir: str):
    """
    Multi-metric radar chart. Angular grid lines are kept because
    they are necessary for reading values on a polar axis.
    Radial grid lines removed. Background white.
    Inference speed inverted so 'more' = better on all axes.
    """
    metrics       = ["accuracy", "f1_macro", "top5_accuracy",
                     "roc_auc_macro", "speed_score"]
    metric_labels = ["Accuracy", "F1 Macro", "Top-5\nAccuracy",
                     "ROC-AUC", "Speed\n(inverted ms)"]

    df = summary_df.copy()

    # Speed score: invert ms so higher = faster (better)
    if "mean_inference_ms" in df.columns:
        max_ms = df["mean_inference_ms"].replace(0, np.nan).max()
        df["speed_score"] = 1 - (df["mean_inference_ms"] /
                                 (max_ms * 1.1)).clip(0, 1)
    else:
        df["speed_score"] = 0.0

    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        print("    [SKIP] Not enough metrics for radar chart.")
        return

    N      = len(available)
    angles = [i / N * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Keep angular grid (required) — style it subtly
    ax.grid(True, color="#CCCCCC", linewidth=0.6, linestyle=":")
    ax.spines["polar"].set_color("#CCCCCC")

    # Remove radial (concentric) gridlines — keep only the angular ones
    ax.yaxis.grid(False)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       fontsize=7, color="#999999")

    for _, row in df.iterrows():
        mk     = row.get("model_key", row.get("model", ""))
        label  = MODEL_LABELS.get(mk, mk)
        color  = MODEL_COLORS.get(mk, "#888888")
        values = [float(row.get(m, 0) or 0) for m in available]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=label, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [metric_labels[metrics.index(m)] for m in available],
        fontsize=10
    )
    ax.set_title("Multi-Metric Classifier Comparison",
                 fontsize=13, pad=20, loc="center")
    ax.legend(loc="lower center",
              bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=False, fontsize=10)

    fig.text(0.5, 0.01,
             "Note. All metrics normalised to [0, 1]. "
             "Speed score = 1 − (mean_ms / max_ms), "
             "so higher = faster inference. "
             "Angular grid lines retained for readability.",
             fontsize=7.5, ha="center", va="bottom", color="#444444")

    _savefig(fig, os.path.join(out_dir, "radar_chart.png"))


# ═══════════════════════════════════════════════════════════════
# CALIBRATION PLOT
# ═══════════════════════════════════════════════════════════════

def plot_calibration(model_probas, label_names, out_dir, n_bins=10):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.18, top=0.88)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    for model_key in MODEL_ORDER:
        if model_key not in model_probas:
            continue
        y_true, y_proba = model_probas[model_key]
        if y_proba is None:
            continue
        label  = MODEL_LABELS[model_key]
        color  = MODEL_COLORS[model_key]
        marker = MODEL_MARKERS[model_key]
        y_conf    = y_proba.max(axis=1)
        y_correct = (y_proba.argmax(axis=1) == y_true).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(
                y_correct, y_conf, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, f"{marker}-",
                    label=label, color=color, linewidth=2, markersize=5)
        except Exception as e:
            print(f"    [WARN] Calibration failed for {model_key}: {e}")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=10, labelpad=6)
    ax.set_ylabel("Fraction Correct", fontsize=10, labelpad=6)
    ax.set_title("Calibration Plot (Reliability Diagram)",
                 fontsize=12, pad=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.10, 0.03,
             "Note. Max-class softmax confidence vs. fraction of correct "
             "predictions per bin. Diagonal = perfect calibration.",
             fontsize=7.5, ha="left", va="bottom", color="#444444")
    _apply_apa_style(ax)
    _savefig(fig, os.path.join(out_dir, "calibration_plot.png"))


# ═══════════════════════════════════════════════════════════════
# GLOBAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

def build_summary_table(model_preds, model_probas, label_names,
                         speed_rows, model_sizes, out_dir):
    n_classes = len(label_names)
    rows      = []

    for model_key in MODEL_ORDER:
        if model_key not in model_preds:
            continue
        y_true, y_pred = model_preds[model_key]
        if y_true is None:
            continue
        y_proba = model_probas.get(model_key, (None, None))[1]

        acc      = accuracy_score(y_true, y_pred)
        f1_mac   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_mic   = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        top_k = {}
        for k in [1, 3, 5, 10]:
            if y_proba is not None and k <= n_classes:
                try:
                    top_k[k] = top_k_accuracy_score(
                        y_true, y_proba, k=k, labels=range(n_classes)
                    )
                except Exception:
                    top_k[k] = float("nan")
            else:
                top_k[k] = float("nan")

        roc_auc = float("nan")
        if y_proba is not None:
            unique, counts = np.unique(y_true, return_counts=True)
            valid = unique[counts >= 2]
            if len(valid) > 200:
                valid = np.sort(
                    np.random.default_rng(42).choice(valid, 200, replace=False)
                )
            y_bin = label_binarize(y_true, classes=range(n_classes))
            try:
                roc_auc = roc_auc_score(
                    y_bin[:, valid], y_proba[:, valid],
                    average="macro", multi_class="ovr"
                )
            except Exception:
                pass

        speed_match = [r for r in speed_rows if r["model"] == model_key]
        mean_ms = speed_match[0]["mean_ms"] if speed_match else float("nan")

        rows.append({
            "model_key":         model_key,
            "model":             MODEL_LABELS[model_key],
            "accuracy":          round(acc,     4),
            "f1_macro":          round(f1_mac,  4),
            "f1_micro":          round(f1_mic,  4),
            "f1_weighted":       round(f1_w,    4),
            "top1_accuracy":     round(top_k[1],  4) if not np.isnan(top_k[1])  else None,
            "top3_accuracy":     round(top_k[3],  4) if not np.isnan(top_k[3])  else None,
            "top5_accuracy":     round(top_k[5],  4) if not np.isnan(top_k[5])  else None,
            "top10_accuracy":    round(top_k[10], 4) if not np.isnan(top_k[10]) else None,
            "roc_auc_macro":     round(roc_auc, 4) if not np.isnan(roc_auc) else None,
            "mean_inference_ms": round(mean_ms, 3),
            "model_size_mb":     round(model_sizes.get(model_key, float("nan")), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "global_comparison.csv"), index=False)
    print(f"    [SAVED] global_comparison.csv  ({len(df)} models)")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_full_comparison(results_base_dir, dataset_name="test",
                         n_speed_trials=N_SPEED_TRIALS,
                         top_n_confusion=25,
                         folder_map=None):
    """
    folder_map : dict mapping model_key → subfolder name under results_base_dir.
    Default maps to the v2 folder names used in this dissertation.
    Example: folder_map={"random_forest": "rf_243", "cif": "cif_243", ...}
    """
    out_dir = os.path.join(results_base_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  DISSERTATION COMPARISON EVALUATOR")
    print(f"  Output → {out_dir}")
    print(f"{'='*60}\n")

    MODEL_CLASS_NAMES = {
        "inceptiontime":       "InceptionTimeClassifier",
        "random_forest":       "RandomForestClassifier",
        "logistic_regression": "LogisticRegressionClassifier",
        "cif":                 "CanonicalIntervalForest",
    }

    # Default folder map — matches OBrown_DIS9300_v2 structure
    DEFAULT_FOLDER_MAP = {
        "inceptiontime":       "it_243",
        "random_forest":       "rf_243",
        "logistic_regression": "lr_243",
        "cif":                 "cif_243",
    }
    if folder_map is None:
        folder_map = DEFAULT_FOLDER_MAP

    # Load label encoder from the mapped folders
    label_encoder = None
    for mk in MODEL_ORDER:
        folder = folder_map.get(mk, mk)
        enc = os.path.join(results_base_dir, folder, "label_encoder.pkl")
        if os.path.exists(enc):
            label_encoder = joblib.load(enc)
            print(f"  [INFO] Label encoder from: {mk}")
            break
    if label_encoder is None:
        raise FileNotFoundError(
            f"No label_encoder.pkl found under {results_base_dir}."
        )
    label_names = list(label_encoder.classes_)
    n_classes   = len(label_names)
    print(f"  [INFO] Classes: {n_classes}")

    # Load predictions and probabilities
    model_preds  = {}
    model_probas = {}
    for model_key, class_name in MODEL_CLASS_NAMES.items():
        folder    = folder_map.get(model_key, model_key)
        split_dir = os.path.join(results_base_dir, folder)
        y_true, y_pred = _load_predictions(split_dir, class_name, dataset_name)
        if y_true is None:
            print(f"  [SKIP] {model_key} — predictions not found")
            model_preds[model_key]  = (None, None)
            model_probas[model_key] = (None, None)
            continue
        y_proba = _load_proba(split_dir, class_name, dataset_name)
        model_preds[model_key]  = (y_true, y_pred)
        model_probas[model_key] = (y_true, y_proba)
        print(f"  [OK]  {model_key:25s} samples={len(y_true):5d} "
              f"proba={'yes' if y_proba is not None else 'no'}")

    # Model sizes
    MODEL_PKL_NAMES = {
        "inceptiontime":       "inceptiontime_model.pkl",
        "random_forest":       "random_forest_model.pkl",
        "logistic_regression": "logistic_regression_model.pkl",
        "cif":                 "cif_model.pkl",
    }
    model_sizes = {}
    size_rows   = []
    for mk, pkl in MODEL_PKL_NAMES.items():
        folder = folder_map.get(mk, mk)
        path = os.path.join(results_base_dir, folder, pkl)
        mb   = _model_pkl_size_mb(path)
        model_sizes[mk] = mb
        size_rows.append({"model": mk,
                           "model_label": MODEL_LABELS.get(mk, mk),
                           "size_mb": round(mb, 2) if not np.isnan(mb) else None})
    pd.DataFrame(size_rows).to_csv(
        os.path.join(out_dir, "model_sizes.csv"), index=False)
    print("\n  [INFO] Model sizes:")
    for r in size_rows:
        print(f"    {r['model_label']:25s} → {r['size_mb']} MB")

    speed_rows = []

    # Generate all plots
    print("\n  [INFO] Generating visualisations...")
    print("  → Top-N accuracy")
    plot_top_n_accuracy(model_probas, label_names, out_dir)
    print("  → ROC-AUC comparison")
    plot_roc_auc_comparison(model_probas, label_names, out_dir)
    print("  → Precision-Recall AP")
    plot_pr_curve_comparison(model_probas, label_names, out_dir)
    print("  → F1 distribution")
    plot_f1_distribution_comparison(model_preds, label_names, out_dir)
    print("  → Confusion matrices")
    plot_normalised_confusion_matrices(model_preds, label_names, out_dir,
                                       top_n_confusion)
    print("  → Worst / best class analysis")
    analyse_class_performance(model_preds, label_names, out_dir)
    print("  → Calibration")
    plot_calibration(model_probas, label_names, out_dir)
    print("  → Summary table")
    summary_df = build_summary_table(
        model_preds, model_probas, label_names,
        speed_rows, model_sizes, out_dir
    )
    print("  → Radar chart")
    plot_radar_chart(summary_df, out_dir)
    print(f"\n  [DONE] All outputs saved to:\n  {out_dir}\n")
    return summary_df, model_preds, model_probas, label_names


def add_speed_results(speed_rows, summary_df, out_dir):
    speed_df = pd.DataFrame(speed_rows)
    speed_df.to_csv(os.path.join(out_dir, "inference_speed.csv"), index=False)
    plot_inference_speed(speed_df, out_dir)
    speed_map = {r["model"]: r["mean_ms"] for r in speed_rows}
    if "mean_inference_ms" not in summary_df.columns:
        summary_df["mean_inference_ms"] = None
    for idx, row in summary_df.iterrows():
        mk = row.get("model_key", "")
        if mk in speed_map:
            summary_df.at[idx, "mean_inference_ms"] = round(speed_map[mk], 3)
    summary_df.to_csv(os.path.join(out_dir, "global_comparison.csv"),
                      index=False)
    plot_radar_chart(summary_df, out_dir)
    print("  [DONE] Speed results merged.")
    return summary_df
