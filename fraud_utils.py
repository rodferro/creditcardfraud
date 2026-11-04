"""Helpers for the credit card fraud notebook.

Kept out of the notebook so the analysis reads as a story and the reusable
pieces (transformers, evaluation helpers, plot style) can be imported and
tested on their own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

SECONDS_PER_DAY = 24 * 60 * 60

# Colours: categorical slots 1-3 of a colour-blind-safe palette.
GENUINE = "#2a78d6"
FRAUD = "#eb6834"
ACCENT = "#1baf7a"
VIOLET = "#6250d6"
# Model colours avoid orange, which means "fraud" in the class plots.
LR_COLOUR = GENUINE
LGBM_COLOUR = "#008300"  # green: clearly distinct from the LR blue
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path: str | Path = "data/creditcard.csv") -> pd.DataFrame:
    """Load the Kaggle dataset, drop exact duplicate rows, sort by time."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip"
        )
    df = pd.read_csv(path)
    return df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_size: float = 0.2, target: str = "Class"):
    """Train on the earliest transactions, test on the latest ones.

    This mimics deployment: a model is trained on the past and scores the
    future. A random split leaks information from the future into training.
    """
    cutoff = df["Time"].quantile(1 - test_size)
    train, test = df[df["Time"] <= cutoff], df[df["Time"] > cutoff]
    return (
        train.drop(columns=target),
        test.drop(columns=target),
        train[target],
        test[target],
    )


# --------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------
class TimeAmountFeatures(BaseEstimator, TransformerMixin):
    """Replace raw ``Time``/``Amount`` with model-friendly versions.

    * ``Time`` is seconds since the first transaction (the data covers two
      days), so the raw value only encodes "which day". Hour of day is the
      useful signal, encoded on the unit circle so 23h and 0h are neighbours.
    * ``Amount`` is heavily right-skewed; ``log1p`` tames it.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()
        angle = 2 * np.pi * (X["Time"] % SECONDS_PER_DAY) / SECONDS_PER_DAY
        X["hour_sin"] = np.sin(angle)
        X["hour_cos"] = np.cos(angle)
        X["log_amount"] = np.log1p(X["Amount"])
        return X.drop(columns=["Time", "Amount"])

    def get_feature_names_out(self, input_features=None):
        names = [c for c in self.feature_names_in_ if c not in ("Time", "Amount")]
        return np.asarray(names + ["hour_sin", "hour_cos", "log_amount"], dtype=object)


class OutlierScore(BaseEstimator, TransformerMixin):
    """Append an anomaly score from an unsupervised detector as a feature.

    The detector is fitted once, on the training data, in ``fit``. At predict
    time it only *scores* rows, so a transaction gets the same value whether it
    is scored alone or in a batch.
    """

    def __init__(self, detector=None):
        self.detector = detector

    def fit(self, X, y=None):
        detector = self.detector if self.detector is not None else IsolationForest(random_state=42)
        self.detector_ = clone(detector).fit(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = X.copy()
        # score_samples: lower = more abnormal; flip so higher = more abnormal.
        X["outlier_score"] = -self.detector_.score_samples(X)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + ["outlier_score"], dtype=object)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def cv_summary(search, top: int | None = None) -> pd.DataFrame:
    """Mean/std of the CV score per candidate of a fitted *SearchCV."""
    res = pd.DataFrame(search.cv_results_)
    out = res[["params", "mean_test_score", "std_test_score", "mean_fit_time"]]
    out = out.sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    return out.head(top) if top else out


def forward_oof_predict(estimator, X: pd.DataFrame, y: pd.Series, cv) -> pd.Series:
    """Out-of-fold fraud probabilities from a forward-chaining splitter.

    Unlike ``cross_val_predict`` this works with ``TimeSeriesSplit``, whose
    validation blocks don't cover the first chunk of data. Each row is scored
    by a model trained only on earlier rows.
    """
    preds = pd.Series(np.nan, index=X.index)
    for train_idx, val_idx in cv.split(X):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return preds.dropna()


def bootstrap_ci(y_true, y_score, metric, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Point estimate and percentile bootstrap CI for ``metric(y_true, y_score)``."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, y_score), lo, hi


def cost_curve(y_true, y_score, amount, review_cost: float, thresholds=None) -> pd.DataFrame:
    """Expected cost of operating the model at each threshold.

    Cost model (deliberately simple, see the report for the assumptions):
      * every alert is reviewed by an analyst at ``review_cost`` each;
      * every missed fraud loses its full transaction ``amount``;
      * a caught fraud is blocked and costs nothing beyond its review.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    amount = np.asarray(amount)
    if thresholds is None:
        thresholds = np.unique(np.quantile(y_score, np.linspace(0.95, 1, 2001)))
    rows = []
    for t in thresholds:
        alert = y_score >= t
        tp = (alert & y_true).sum()
        fp = (alert & ~y_true).sum()
        fn = (~alert & y_true).sum()
        missed = amount[~alert & y_true].sum()
        rows.append(
            dict(
                threshold=t,
                alerts=int(alert.sum()),
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1),
                review_cost=alert.sum() * review_cost,
                fraud_loss=missed,
                total_cost=alert.sum() * review_cost + missed,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------
def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def savefig(fig, name: str, folder: str | Path = "figures"):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.png")
