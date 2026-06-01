# evaluator.py
# ============================================================
# Dissertation: Real-Time ASL Recognition
# Author: Owasu Brown | National University | 2025-2026
#
# CHANGES FROM PREVIOUS VERSION:
#   - dataset_label added to ALL chart titles and filenames.
#   - Confusion matrix replaced with performance table image
#     (Top-10 best / Bottom-10 worst classes, global metrics banner).
#   - F1 boxplot replaced with three new plots:
#       1. F1 histogram  — class count per F1 bin
#       2. Cumulative F1 curve — % of classes above each threshold
#       3. F1 vs support scatter — does more training data help?
#   - Top-20 barplot kept and improved.
#   - regenerate_plots_from_csv() — reruns plots on completed models
#     without retraining using saved CSV and JSON files.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    sns = None
    _HAS_SNS = False

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

_BLUE   = "#2E75B6"
_TEAL   = "#2A9D8F"
_RED    = "#E63946"
_ORANGE = "#F4A261"
_DARK   = "#1F3864"


class Evaluator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    def _compute_metrics(self, model_name, dataset_label,
                         y_true, y_pred, y_proba, label_names):
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        n_classes = len(label_names)

        acc      = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        f1_wt    = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        top5 = float("nan")
        if (y_proba is not None and
                np.asarray(y_proba).ndim == 2 and
                np.asarray(y_proba).shape[1] >= 5):
            try:
                top5 = top_k_accuracy_score(
                    y_true, np.asarray(y_proba), k=5,
                    labels=list(range(n_classes))
                )
            except Exception:
                pass

        global_metrics = {
            "model"         : model_name,
            "dataset"       : dataset_label,
            "accuracy"      : float(acc),
            "f1_macro"      : float(f1_macro),
            "f1_micro"      : float(f1_micro),
            "f1_weighted"   : float(f1_wt),
            "top5_accuracy" : float(top5) if not np.isnan(top5) else None,
            "n_classes"     : n_classes,
            "n_test_samples": int(len(y_true)),
        }

        tag = f"_{dataset_label}" if dataset_label else ""
        with open(os.path.join(model_dir,
                  f"{model_name}{tag}_global_metrics.json"), "w") as fh:
            json.dump(global_metrics, fh, indent=4)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(n_classes)), zero_division=0
        )
        per_class_df = pd.DataFrame({
            "label"    : label_names,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
            "support"  : support,
        })
        per_class_df.to_csv(
            os.path.join(model_dir,
                         f"{model_name}{tag}_per_class_metrics.csv"),
            index=False
        )

        self._plot_top20_barplot(per_class_df, model_name, dataset_label,
                                 model_dir, tag)
        self._plot_top_bottom_table(per_class_df, model_name, dataset_label,
                                    global_metrics, model_dir, tag)
        self._plot_f1_histogram(per_class_df, model_name, dataset_label,
                                model_dir, tag)
        self._plot_cumulative_f1(per_class_df, model_name, dataset_label,
                                 model_dir, tag)
        self._plot_f1_vs_support(per_class_df, model_name, dataset_label,
                                 model_dir, tag)

        return global_metrics

    # ── Plot 1: Top-20 barplot ────────────────────────────────────────────
    def _plot_top20_barplot(self, df, model_name, dataset_label,
                            model_dir, tag):
        top20 = df.sort_values("f1", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top20)))[::-1]
        ax.barh(range(len(top20)), top20["f1"].values,
                color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["label"].values, fontsize=9)
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_title(
            f"Top 20 Labels by F1\n{model_name}  |  Dataset: {dataset_label}",
            fontsize=12, color=_DARK
        )
        ax.set_xlim(0, 1.05)
        mean_f1 = df["f1"].mean()
        ax.axvline(mean_f1, color=_RED, linestyle="--", linewidth=1.2,
                   label=f"Mean F1 = {mean_f1:.3f}")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir,
                    f"{model_name}{tag}_top20_barplot.png"), dpi=150)
        plt.close()

    # ── Plot 2: Top-10 / Bottom-10 table image ────────────────────────────
    def _plot_top_bottom_table(self, df, model_name, dataset_label,
                               global_metrics, model_dir, tag):
        top10    = df.sort_values("f1", ascending=False).head(10)
        bottom10 = (df[df["support"] > 0]
                    .sort_values("f1", ascending=True).head(10))

        top10.to_csv(os.path.join(model_dir,
                     f"{model_name}{tag}_top10_classes.csv"), index=False)
        bottom10.to_csv(os.path.join(model_dir,
                        f"{model_name}{tag}_bottom10_classes.csv"),
                        index=False)

        fig = plt.figure(figsize=(14, 13))
        fig.patch.set_facecolor("white")
        gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 2.5, 2.5],
                                hspace=0.55)

        # Banner
        ax0 = fig.add_subplot(gs[0])
        ax0.axis("off")
        top5_str = (f"{global_metrics['top5_accuracy']*100:.2f}%"
                    if global_metrics.get("top5_accuracy") else "N/A")
        banner = (
            f"Model: {model_name}   |   Dataset: {dataset_label}\n"
            f"Accuracy: {global_metrics['accuracy']*100:.2f}%     "
            f"F1 Macro: {global_metrics['f1_macro']*100:.2f}%     "
            f"F1 Weighted: {global_metrics.get('f1_weighted',0)*100:.2f}%     "
            f"Top-5 Accuracy: {top5_str}"
        )
        ax0.text(0.5, 0.5, banner, ha="center", va="center",
                 fontsize=11, color=_DARK, fontweight="bold",
                 transform=ax0.transAxes)
        ax0.set_title(
            f"Performance Summary — {model_name} ({dataset_label})",
            fontsize=13, color=_DARK, pad=8
        )

        col_labels = ["Sign", "Precision", "Recall", "F1", "Test videos"]

        # Top-10
        ax1 = fig.add_subplot(gs[1])
        ax1.axis("off")
        ax1.set_title("Top 10 Classes by F1 Score",
                      fontsize=11, color=_TEAL, pad=4)
        rows_top = [[r["label"], f"{r['precision']:.3f}",
                     f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                     str(int(r["support"]))]
                    for _, r in top10.iterrows()]
        tbl1 = ax1.table(cellText=rows_top, colLabels=col_labels,
                         cellLoc="center", loc="center",
                         bbox=[0, 0, 1, 1])
        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(9)
        for (row, col), cell in tbl1.get_celld().items():
            if row == 0:
                cell.set_facecolor(_TEAL)
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#E8F5F3")
            cell.set_edgecolor("#CCCCCC")

        # Bottom-10
        ax2 = fig.add_subplot(gs[2])
        ax2.axis("off")
        ax2.set_title(
            "Bottom 10 Classes by F1 Score (classes with test videos only)",
            fontsize=11, color=_RED, pad=4
        )
        rows_bot = [[r["label"], f"{r['precision']:.3f}",
                     f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                     str(int(r["support"]))]
                    for _, r in bottom10.iterrows()]
        tbl2 = ax2.table(cellText=rows_bot, colLabels=col_labels,
                         cellLoc="center", loc="center",
                         bbox=[0, 0, 1, 1])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        for (row, col), cell in tbl2.get_celld().items():
            if row == 0:
                cell.set_facecolor(_RED)
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#FFF0F0")
            cell.set_edgecolor("#CCCCCC")

        plt.savefig(os.path.join(model_dir,
                    f"{model_name}{tag}_performance_table.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 3: F1 histogram ──────────────────────────────────────────────
    def _plot_f1_histogram(self, df, model_name, dataset_label,
                           model_dir, tag):
        f1vals = df["f1"].values
        bins   = np.arange(0, 1.05, 0.05)
        counts, edges = np.histogram(f1vals, bins=bins)
        bar_colors = [_BLUE if e < 0.5 else _TEAL for e in edges[:-1]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(edges[:-1], counts, width=0.045, color=bar_colors,
               edgecolor="white", linewidth=0.5, align="edge")
        ax.axvline(f1vals.mean(), color=_RED, linestyle="--",
                   linewidth=1.5, label=f"Mean F1 = {f1vals.mean():.3f}")
        ax.axvline(np.median(f1vals), color=_ORANGE, linestyle=":",
                   linewidth=1.5,
                   label=f"Median F1 = {np.median(f1vals):.3f}")
        n_zero    = int((f1vals == 0).sum())
        n_nonzero = int((f1vals > 0).sum())
        ax.set_xlabel("F1 Score", fontsize=11)
        ax.set_ylabel("Number of Classes", fontsize=11)
        ax.set_title(
            f"F1 Score Distribution Across All Classes\n"
            f"{model_name}  |  {dataset_label}  |  "
            f"{n_zero} classes at F1=0  |  {n_nonzero} classes F1>0",
            fontsize=11, color=_DARK
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir,
                    f"{model_name}{tag}_f1_histogram.png"), dpi=150)
        plt.close()

    # ── Plot 4: Cumulative F1 curve ───────────────────────────────────────
    def _plot_cumulative_f1(self, df, model_name, dataset_label,
                            model_dir, tag):
        f1vals     = np.sort(df["f1"].values)[::-1]
        thresholds = np.linspace(0, 1, 200)
        pct_above  = [100*(f1vals >= t).sum()/len(f1vals)
                      for t in thresholds]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, pct_above, color=_BLUE, linewidth=2.5)
        ax.fill_between(thresholds, pct_above, alpha=0.12, color=_BLUE)

        for t, lbl in [(0.1, "F1≥0.1"), (0.3, "F1≥0.3"),
                       (0.5, "F1≥0.5")]:
            pct = 100*(f1vals >= t).sum()/len(f1vals)
            ax.axvline(t, color="grey", linestyle=":", alpha=0.5)
            ax.annotate(
                f"{pct:.1f}%\n{lbl}",
                xy=(t, pct), xytext=(t+0.03, min(pct+5, 98)),
                fontsize=8, color=_DARK,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8)
            )

        ax.set_xlabel("F1 Threshold", fontsize=11)
        ax.set_ylabel("% of Classes at or above threshold", fontsize=11)
        ax.set_title(
            f"Cumulative F1 Performance Curve\n"
            f"{model_name}  |  {dataset_label}",
            fontsize=11, color=_DARK
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir,
                    f"{model_name}{tag}_cumulative_f1.png"), dpi=150)
        plt.close()

    # ── Plot 5: F1 vs support scatter ─────────────────────────────────────
    def _plot_f1_vs_support(self, df, model_name, dataset_label,
                            model_dir, tag):
        df_plot = df[df["support"] > 0].copy()
        if len(df_plot) < 3:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(df_plot["support"], df_plot["f1"],
                   alpha=0.45, s=20, color=_BLUE, edgecolors="none")
        try:
            z = np.polyfit(df_plot["support"], df_plot["f1"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df_plot["support"].min(),
                                 df_plot["support"].max(), 100)
            ax.plot(x_line, p(x_line), color=_RED,
                    linewidth=1.8, linestyle="--",
                    label=f"Trend (slope={z[0]:.3f})")
            ax.legend(fontsize=9)
        except Exception:
            pass

        ax.set_xlabel("Number of test videos per class", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(
            f"F1 Score vs Test Support per Class\n"
            f"{model_name}  |  {dataset_label}",
            fontsize=11, color=_DARK
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir,
                    f"{model_name}{tag}_f1_vs_support.png"), dpi=150)
        plt.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate_model(self, model_name, y_true, y_pred,
                       y_proba, label_names, dataset_label=""):
        return self._compute_metrics(
            model_name, dataset_label,
            y_true, y_pred, y_proba, label_names
        )

    def evaluate_and_save(self, model, X, y_true, label_encoder,
                          dataset_name="test", model_name=None,
                          dataset_label=None):
        """
        Run inference then evaluate and save all outputs.

        dataset_label : human-readable label shown in chart titles,
                        e.g. '243-class' or '1683-class'.
                        If None, falls back to dataset_name.
        """
        if model_name is None:
            model_name = type(model).__name__
        if dataset_label is None:
            dataset_label = dataset_name

        split_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(split_dir, exist_ok=True)

        y_pred  = np.array(model.predict(X)).reshape(-1)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                pass

        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            label_names = list(label_encoder.classes_)
        else:
            n_cls = int(np.max(y_true)) + 1
            label_names = [str(i) for i in range(n_cls)]

        original = self.output_dir
        try:
            self.output_dir = split_dir
            metrics = self._compute_metrics(
                model_name, dataset_label,
                y_true, y_pred, y_proba, label_names
            )
        finally:
            self.output_dir = original

        pd.DataFrame({"y_true": np.array(y_true).reshape(-1),
                      "y_pred": y_pred}).to_csv(
            os.path.join(split_dir,
                         f"{model_name}_{dataset_name}_predictions.csv"),
            index=False
        )
        if y_proba is not None:
            np.save(
                os.path.join(split_dir,
                             f"{model_name}_{dataset_name}_proba.npy"),
                y_proba
            )
        return metrics

    def compare_models(self, results_list):
        df = pd.DataFrame(results_list)
        df.to_csv(os.path.join(self.output_dir,
                               "global_comparison.csv"), index=False)
        return df

    def reload_saved_metrics(self):
        results = []
        for model_dir in os.listdir(self.output_dir):
            model_path = os.path.join(self.output_dir, model_dir)
            if os.path.isdir(model_path):
                for fname in os.listdir(model_path):
                    if fname.endswith("_global_metrics.json"):
                        with open(os.path.join(model_path, fname)) as fh:
                            results.append(json.load(fh))
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(self.output_dir,
                                   "global_comparison.csv"), index=False)
            return df
        print("No saved JSON metrics found.")
        return pd.DataFrame()

    def regenerate_plots_from_csv(self, results_dir, dataset_label):
        """
        Regenerate all plots for a completed model run without retraining.
        Uses the saved *_per_class_metrics.csv and *_global_metrics.json.

        Usage in Colab (paste as a new cell):
        ─────────────────────────────────────
        from evaluator import Evaluator
        evaluator = Evaluator(output_dir='placeholder')

        # Regenerate RF 243-class plots
        evaluator.regenerate_plots_from_csv(
            results_dir   = os.path.join(PROJECT_DIR,
                'results/rf_243/test/RandomForestClassifier'),
            dataset_label = '243-class'
        )

        # Regenerate LR 1683-class plots
        evaluator.regenerate_plots_from_csv(
            results_dir   = os.path.join(PROJECT_DIR,
                'results/lr_full/test/LogisticRegressionClassifier'),
            dataset_label = '1683-class'
        )
        ─────────────────────────────────────
        """
        csv_path   = None
        json_path  = None
        model_name = None

        for fname in os.listdir(results_dir):
            if fname.endswith("_per_class_metrics.csv"):
                csv_path   = os.path.join(results_dir, fname)
                model_name = fname.replace("_per_class_metrics.csv", "")
            if fname.endswith("_global_metrics.json"):
                json_path  = os.path.join(results_dir, fname)

        if csv_path is None or json_path is None:
            raise FileNotFoundError(
                f"Missing metrics files in {results_dir}\n"
                f"  Found CSV : {csv_path}\n"
                f"  Found JSON: {json_path}"
            )

        df = pd.read_csv(csv_path)
        with open(json_path) as fh:
            global_metrics = json.load(fh)
        global_metrics.setdefault("f1_weighted", 0.0)
        global_metrics.setdefault("dataset", dataset_label)

        tag = f"_{dataset_label}" if dataset_label else ""

        self._plot_top20_barplot(df, model_name, dataset_label,
                                 results_dir, tag)
        self._plot_top_bottom_table(df, model_name, dataset_label,
                                    global_metrics, results_dir, tag)
        self._plot_f1_histogram(df, model_name, dataset_label,
                                results_dir, tag)
        self._plot_cumulative_f1(df, model_name, dataset_label,
                                 results_dir, tag)
        self._plot_f1_vs_support(df, model_name, dataset_label,
                                 results_dir, tag)

        print(f"[DONE] Regenerated 5 plots for {model_name} ({dataset_label})")
        print(f"       Location: {results_dir}")
        return model_name
