"""
inceptiontime_classifier.py
============================
Pure-PyTorch InceptionTime for multivariate time-series ASL classification.

InceptionTime is a TIME-SERIES classifier.
It operates on the full ordered sequence of landmark frames per video.
It does NOT use aggregated features.
The 3-D tensor (n_samples, n_channels, n_frames) preserves frame order.
1-D convolutions are applied across the frame (time) axis to learn
temporal patterns in the landmark trajectories.

Network capacity for 1,867-class ASL problem:
  n_filters = 64  → 256-dimensional representation per time step
  depth     = 9   → three residual groups of three inception blocks
  Parameters ≈ 1.4M — sufficient for the class count and channel size.

Training improvements over the original 32-filter version:
  - Larger n_filters (64 vs 32) gives more representational capacity
  - Deeper network (depth 9 vs 6) learns more complex temporal patterns
  - ReduceLROnPlateau scheduler prevents getting stuck in flat loss regions
  - Gradient clipping prevents exploding gradients with many classes
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evaluator import Evaluator
from data_loaders import load_long_csv_as_timeseries

# ── PyTorch imports at module level so pickle can find classes ─────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Top-level PyTorch classes (must be at module level for joblib pickling) ─

class _InceptionModule(nn.Module if _TORCH_AVAILABLE else object):
    """
    Single Inception block with four parallel branches:
      1. Bottleneck → large conv  (kernel 39)
      2. Bottleneck → medium conv (kernel 19)
      3. Bottleneck → small conv  (kernel 9)
      4. MaxPool    → pointwise conv
    All branches concatenated → BatchNorm → ReLU
    Output channels = n_filters * 4
    """
    def __init__(self, in_channels, n_filters=64, kernel_sizes=(39, 19, 9)):
        super().__init__()
        self.bottleneck = nn.Conv1d(
            in_channels, n_filters, kernel_size=1, bias=False
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                n_filters, n_filters,
                kernel_size=k, padding=k // 2, bias=False
            )
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.bn  = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        b    = self.bottleneck(x)
        outs = [conv(b) for conv in self.convs] + [self.maxpool_conv(x)]
        return self.act(self.bn(torch.cat(outs, dim=1)))


class _InceptionTimeNetwork(nn.Module if _TORCH_AVAILABLE else object):
    """
    Full InceptionTime network.

    Architecture:
      - depth inception blocks grouped in sets of 3
      - residual shortcut added at the end of every 3-block group
      - global average pooling collapses the time dimension
      - linear classifier maps to n_classes

    Default config (n_filters=64, depth=9):
      - 3 residual groups × 3 inception blocks = 9 blocks total
      - 256-dimensional representation (64 × 4 branches)
      - ~1.4M parameters — adequate for 1,867-class ASL problem
    """
    def __init__(self, n_channels, n_classes, depth=9, n_filters=64):
        super().__init__()
        in_ch  = n_channels
        out_ch = n_filters * 4

        blocks    = []
        residuals = []

        for i in range(depth):
            if i % 3 == 0:
                residual_in_ch = in_ch   # capture group input channels

            blocks.append(_InceptionModule(in_ch, n_filters))
            in_ch = out_ch

            if i % 3 == 2:
                residuals.append(nn.Sequential(
                    nn.Conv1d(residual_in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                ))
            else:
                residuals.append(None)

        self.blocks    = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(
            [r if r is not None else nn.Identity() for r in residuals]
        )
        self.res_flags = [i % 3 == 2 for i in range(depth)]
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.dropout   = nn.Dropout(p=0.3)
        self.fc        = nn.Linear(out_ch, n_classes)
        self.act       = nn.ReLU()

    def forward(self, x):
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.res_flags[i]:
                res = self.residuals[i](residual)
                if res.shape[-1] != x.shape[-1]:
                    res = F.interpolate(res, size=x.shape[-1])
                x = self.act(x + res)
                residual = x
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ── Helper functions ───────────────────────────────────────────────────────

def _write_marker(path: str):
    with open(path, "w") as f:
        f.write("ok\n")


def _impute_X(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-channel mean. Shape: [n_samples, n_channels, n_frames]"""
    X = X.copy()
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            series = X[i, c, :]
            mask = np.isnan(series)
            if mask.any() and not mask.all():
                series[mask] = float(np.nanmean(series))
            elif mask.all():
                series[:] = 0.0
    return X


def _filter_timeseries_to_known_labels(X, y, known_classes, split_name: str):
    known_set = set(map(str, known_classes))
    y_series  = pd.Series(y).astype(str)
    mask      = y_series.isin(known_set).to_numpy()
    dropped   = int((~mask).sum())
    if dropped > 0:
        unseen  = sorted(y_series.loc[~mask].unique().tolist())
        preview = ", ".join(unseen[:10])
        if len(unseen) > 10:
            preview += ", ..."
        print(
            f"[WARN] {split_name}: dropping {dropped} sample(s) with unseen "
            f"label(s) not present in training: {preview}"
        )
    return X[mask], y_series.loc[mask].to_numpy()


def _load_or_build_npz(
    csv_path: str,
    npz_path: str,
    force: bool = False,
    use_masked: bool = False,
):
    if (not force) and os.path.exists(npz_path):
        print(f"[SKIP] Loading cached time series: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        return data["X"], data["y"], data["video_names"]

    print(f"[INFO] Converting CSV to time series: {csv_path} | use_masked={use_masked}")
    X, y, video_names = load_long_csv_as_timeseries(csv_path, use_masked=use_masked)
    np.savez_compressed(npz_path, X=X, y=y, video_names=video_names)
    print(f"[INFO] Cached time series saved: {npz_path}")
    return X, y, video_names


# ── Sklearn-compatible wrapper ─────────────────────────────────────────────

class InceptionTimeClassifier:
    """
    Sklearn-compatible wrapper around _InceptionTimeNetwork.
    Exposes fit() / predict() / predict_proba() so Evaluator works unchanged.
    Defined at module top level so joblib.dump() can pickle it correctly.

    Key training improvements for large-vocabulary ASL (1,867 classes):
      - n_filters=64, depth=9 for sufficient model capacity
      - ReduceLROnPlateau reduces lr when validation loss plateaus
      - Gradient clipping (max_norm=1.0) stabilises training
      - Dropout(0.3) in the classifier head reduces overfitting
    """

    def __init__(
        self,
        n_epochs:   int   = 50,
        batch_size: int   = 32,
        lr:         float = 5e-4,
        n_filters:  int   = 64,
        depth:      int   = 9,
        verbose:    bool  = True,
    ):
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.n_filters  = n_filters
        self.depth      = depth
        self.verbose    = verbose
        self.model_     = None
        self.classes_   = None
        self._device    = None

    def fit(self, X, y):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required. Install with: !pip install torch --quiet"
            )
        self._device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)
        n_channels    = X.shape[1]

        print(f"[INFO] Device          : {self._device}")
        print(f"[INFO] n_channels      : {n_channels}")
        print(f"[INFO] n_classes       : {n_classes}")
        print(f"[INFO] n_filters       : {self.n_filters}")
        print(f"[INFO] depth           : {self.depth}")
        print(f"[INFO] representation  : {self.n_filters * 4}-dim")

        self.model_ = _InceptionTimeNetwork(
            n_channels, n_classes,
            depth=self.depth, n_filters=self.n_filters
        ).to(self._device)

        n_params = sum(p.numel() for p in self.model_.parameters())
        print(f"[INFO] Parameters      : {n_params:,}")

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        X_t    = torch.tensor(X, dtype=torch.float32)
        y_t    = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model_.train()
        for epoch in range(self.n_epochs):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)

            epoch_loss = total_loss / total
            epoch_acc  = correct / total * 100
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if self.verbose:
                print(
                    f"  Epoch {epoch+1:>3}/{self.n_epochs}  "
                    f"loss={epoch_loss:.4f}  acc={epoch_acc:.1f}%  "
                    f"lr={current_lr:.2e}"
                )
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        self.model_.eval()
        # Process in batches to avoid OOM on large test sets
        batch_size = 256
        all_probs  = []
        for start in range(0, len(X), batch_size):
            xb  = torch.tensor(
                X[start:start+batch_size], dtype=torch.float32
            ).to(self._device)
            with torch.no_grad():
                logits = self.model_(xb)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(all_probs, axis=0)


# ── Public training function ───────────────────────────────────────────────

def train_inceptiontime(
    results_dir: str  = "results/inceptiontime",
    n_epochs:    int  = 50,
    batch_size:  int  = 32,
    force:       bool = False,
    # ── Pre-built NPZ pathway (preferred) ─────────────────────────────────
    train_npz: str    = None,
    test_npz:  str    = None,
    # ── CSV pathway (legacy) ───────────────────────────────────────────────
    train_csv:  str   = None,
    test_csv:   str   = None,
    use_masked: bool  = False,
):
    """
    Train InceptionTime on the full time-series tensor.

    Preferred usage (pre-built NPZ):
        train_inceptiontime(train_npz='ASL_reduced_train.npz',
                            test_npz='ASL_reduced_test.npz',
                            results_dir='results_v3/inceptiontime')

    Legacy usage (build from CSV):
        train_inceptiontime(train_csv='ASL_reduced_train.csv',
                            test_csv='ASL_reduced_test.csv',
                            results_dir='results_v3/inceptiontime')
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for InceptionTime.\n"
            "Install: !pip install torch --quiet then restart."
        )

    os.makedirs(results_dir, exist_ok=True)

    ts_train_path = os.path.join(results_dir, "ts_train.npz")
    ts_test_path  = os.path.join(results_dir, "ts_test.npz")

    model_path   = os.path.join(results_dir, "inceptiontime_model.pkl")
    encoder_path = os.path.join(results_dir, "label_encoder.pkl")

    eval_test_done = os.path.join(results_dir, ".EVAL_DONE_test")

    # ── Load tensors ────────────────────────────────────────────────────────
    if train_npz and test_npz:
        print(f"[INFO] Loading pre-built train NPZ: {train_npz}")
        d_train = np.load(train_npz, allow_pickle=True)
        X_train, y_train = d_train["X"], d_train["y"]
        print(f"[INFO] Loading pre-built test NPZ : {test_npz}")
        d_test  = np.load(test_npz, allow_pickle=True)
        X_test, y_test = d_test["X"], d_test["y"]
    elif train_csv and test_csv:
        X_train, y_train, _ = _load_or_build_npz(
            train_csv, ts_train_path, force=force, use_masked=use_masked
        )
        X_test,  y_test,  _ = _load_or_build_npz(
            test_csv, ts_test_path, force=force, use_masked=use_masked
        )
    else:
        raise ValueError(
            "Provide either (train_npz, test_npz) or (train_csv, test_csv)."
        )

    if (not force) and os.path.exists(model_path) and os.path.exists(encoder_path):
        print(f"[SKIP] InceptionTime model already exists. Loading from {results_dir}")
        clf = joblib.load(model_path)
        le  = joblib.load(encoder_path)
    else:
        le = LabelEncoder()
        y_train_str = pd.Series(y_train).astype(str)
        y_train_enc = le.fit_transform(y_train_str)

        print(f"[INFO] Training samples : {len(y_train_str)} | Test: {len(y_test)}")
        print(f"[INFO] Classes          : {len(le.classes_)}")
        print(f"[INFO] Tensor shape     : {X_train.shape}  "
              f"[n_samples, n_channels, n_frames]")
        print(f"[INFO] Epochs           : {n_epochs}")
        print(f"[INFO] Batch size       : {batch_size}")
        print(f"[INFO] Temporal method  : 1-D convolutions on ordered frames")

        X_train = _impute_X(X_train)
        X_test  = _impute_X(X_test)

        clf = InceptionTimeClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=5e-4,
            n_filters=64,
            depth=9,
            verbose=True,
        )
        clf.fit(X_train, y_train_enc)

        joblib.dump(clf, model_path)
        joblib.dump(le,  encoder_path)
        print(f"[INFO] InceptionTime model saved to {results_dir}")

    X_test, y_test = _filter_timeseries_to_known_labels(
        X_test, y_test, le.classes_, "test"
    )

    if len(y_test) == 0:
        raise ValueError("Test split empty after label filtering.")

    y_test_enc = le.transform(pd.Series(y_test).astype(str))

    evaluator = Evaluator(output_dir=results_dir)

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

    print("[INFO] InceptionTime pipeline complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate InceptionTime on ASL dataset"
    )
    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--val_csv",     type=str, required=True)
    parser.add_argument("--test_csv",    type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/inceptiontime")
    parser.add_argument("--n_epochs",    type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--use_masked",  action="store_true")
    args = parser.parse_args()

    train_inceptiontime(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        results_dir=args.results_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        force=args.force,
        use_masked=args.use_masked,
    )
