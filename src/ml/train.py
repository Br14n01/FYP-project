"""
Training pipeline: XGBoost / Random Forest with walk-forward validation,
multi-configuration comparison, and SHAP feature importance.

Usage:
    from src.ml.train import run_experiment
    results = run_experiment("AAPL")
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

from src.ml.features import (
    build_feature_matrix,
    get_feature_columns,
    get_sentiment_only_columns,
    OHLCV_COLS,
    SENTIMENT_FEATURE_COLS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _xgb_model(n_classes: int = 3) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
    )


def _xgb_model_with_params(params: dict, n_classes: int = 3) -> XGBClassifier:
    """Build an XGBClassifier from an arbitrary parameter dict."""
    defaults = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "random_state": 42,
    }
    merged = {**defaults, **params}
    return XGBClassifier(**merged)


def _rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Purged time-series cross-validation
# ---------------------------------------------------------------------------

class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit with a purge gap between train and test to avoid
    lookahead leakage from rolling-window labels.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            test_start = train_end + self.purge_gap
            test_end = test_start + fold_size
            if test_end > n:
                test_end = n
            if test_start >= n:
                break
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, test_end))
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    n_trials: int = 50,
    n_splits: int = 5,
    purge_gap: int = 10,
    timeout: int | None = 600,
) -> dict:
    """
    Bayesian hyperparameter search for XGBoost using Optuna with
    purged time-series cross-validation and early stopping.

    Returns the best parameter dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df[feature_cols]
    y = df[label_col]
    n_classes = int(y.nunique())
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        f1_scores = []
        for train_idx, test_idx in cv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # Hold out the last 10% of training data for early stopping
            es_split = int(len(X_tr) * 0.9)
            X_tr_fit, X_tr_es = X_tr.iloc[:es_split], X_tr.iloc[es_split:]
            y_tr_fit, y_tr_es = y_tr.iloc[:es_split], y_tr.iloc[es_split:]

            model = _xgb_model_with_params(params, n_classes=n_classes)
            model.set_params(early_stopping_rounds=30)
            model.fit(
                X_tr_fit, y_tr_fit,
                eval_set=[(X_tr_es, y_tr_es)],
                verbose=False,
            )

            y_pred = model.predict(X_te)
            f1_scores.append(f1_score(y_te, y_pred, average="macro"))

        return float(np.mean(f1_scores))

    study = optuna.create_study(direction="maximize", study_name="xgb_tuning")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"\n  Optuna tuning complete: {len(study.trials)} trials")
    print(f"  Best F1-macro: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    model_type: str = "xgboost",
    n_splits: int = 5,
    purge_gap: int = 0,
) -> tuple[pd.DataFrame, object]:
    """
    Time-series walk-forward cross-validation.

    Parameters
    ----------
    purge_gap : int
        Number of rows to drop between train and test folds to avoid
        lookahead leakage from rolling-window labels.  0 = no purge.

    Returns (metrics_df, last_trained_model).
    """
    if purge_gap > 0:
        tscv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        X_train, y_train = train[feature_cols], train[label_col]
        X_test, y_test = test[feature_cols], test[label_col]

        if model_type == "xgboost":
            model = _xgb_model(n_classes=int(y_train.nunique()))
        else:
            model = _rf_model()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics.append(
            {
                "fold": fold + 1,
                "start": str(test.index[0]),
                "end": str(test.index[-1]),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
            }
        )

    return pd.DataFrame(fold_metrics), model


# ---------------------------------------------------------------------------
# Full train/test split evaluation with detailed metrics
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    model_type: str = "xgboost",
    train_ratio: float = 0.8,
) -> tuple[dict, object, np.ndarray, np.ndarray]:
    """
    Simple chronological train/test split.
    Returns (metrics_dict, model, y_test, y_pred).
    """
    split = int(len(df) * train_ratio)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[feature_cols], train[label_col]
    X_test, y_test = test[feature_cols], test[label_col]

    if model_type == "xgboost":
        model = _xgb_model(n_classes=int(y_train.nunique()))
    else:
        model = _rf_model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }
    return metrics, model, np.array(y_test), y_pred


# ---------------------------------------------------------------------------
# Multi-config comparison
# ---------------------------------------------------------------------------

CONFIG_NAMES = [
    "technical_only_xgb",
    "sentiment_only_xgb",
    "hybrid_xgb",
    "hybrid_rf",
]


def run_experiment(
    symbol: str,
    start: str = "2022-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.01,
    n_splits: int = 5,
    output_dir: str = "results",
) -> dict:
    """
    Run the full comparison experiment for a single ticker.

    Trains four configurations:
      1. technical_only_xgb  – XGBoost on technical indicators only
      2. sentiment_only_xgb  – XGBoost on sentiment features only
      3. hybrid_xgb          – XGBoost on combined features
      4. hybrid_rf           – Random Forest on combined features

    Returns dict of results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {symbol}")
    print(f"{'='*60}")

    # Build feature matrix (with sentiment)
    df = build_feature_matrix(
        symbol,
        start=start,
        end=end,
        lookahead=lookahead,
        thresh=thresh,
        include_sentiment=True,
    )

    all_features = get_feature_columns(df, include_sentiment=True)
    tech_features = get_feature_columns(df, include_sentiment=False)
    sent_features = [c for c in get_sentiment_only_columns() if c in df.columns]

    has_sentiment = len(sent_features) > 0

    configs = {
        "technical_only_xgb": {"features": tech_features, "model": "xgboost"},
        "hybrid_xgb": {"features": all_features, "model": "xgboost"},
        "hybrid_rf": {"features": all_features, "model": "random_forest"},
    }
    if has_sentiment:
        configs["sentiment_only_xgb"] = {
            "features": sent_features,
            "model": "xgboost",
        }

    results = {"symbol": symbol, "configs": {}}

    for config_name, cfg in configs.items():
        print(f"\n--- {config_name} ({len(cfg['features'])} features) ---")

        # Walk-forward
        wf_df, last_model = walk_forward_validate(
            df,
            cfg["features"],
            model_type=cfg["model"],
            n_splits=n_splits,
        )
        wf_summary = {
            "accuracy_mean": wf_df["accuracy"].mean(),
            "accuracy_std": wf_df["accuracy"].std(),
            "f1_macro_mean": wf_df["f1_macro"].mean(),
            "f1_macro_std": wf_df["f1_macro"].std(),
        }

        # Simple train/test
        metrics, model, y_test, y_pred = train_and_evaluate(
            df, cfg["features"], model_type=cfg["model"]
        )

        results["configs"][config_name] = {
            "train_test": metrics,
            "walk_forward": wf_summary,
            "walk_forward_folds": wf_df.to_dict(orient="records"),
            "n_features": len(cfg["features"]),
        }

        print(f"  Train/Test  acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}")
        print(
            f"  Walk-Fwd    acc={wf_summary['accuracy_mean']:.4f}±{wf_summary['accuracy_std']:.4f}"
            f"  F1={wf_summary['f1_macro_mean']:.4f}±{wf_summary['f1_macro_std']:.4f}"
        )

        # Save confusion matrix for hybrid_xgb
        if config_name == "hybrid_xgb":
            _save_confusion_matrix(y_test, y_pred, symbol, output_dir)
            _save_roc_curves(model, df, cfg["features"], symbol, output_dir)

    # Save SHAP for hybrid_xgb
    if "hybrid_xgb" in configs:
        _save_shap(
            df,
            configs["hybrid_xgb"]["features"],
            symbol,
            output_dir,
        )

    # Save comparison table
    _save_comparison_table(results, symbol, output_dir)

    # Persist full results as JSON
    json_path = os.path.join(output_dir, f"{symbol}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {json_path}")

    return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    symbol: str,
    output_dir: str,
):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Neutral", "Bearish", "Bullish"],
        yticklabels=["Neutral", "Bearish", "Bullish"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{symbol} – Normalised Confusion Matrix (Hybrid XGB)")
    fig.tight_layout()
    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)
    fig.savefig(os.path.join(chart_dir, f"{symbol}_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _save_roc_curves(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    symbol: str,
    output_dir: str,
):
    split = int(len(df) * 0.8)
    test = df.iloc[split:]
    X_test, y_test = test[feature_cols], test["label"]
    y_proba = model.predict_proba(X_test)

    n_classes = 3
    fpr, tpr, roc_auc_vals = {}, {}, {}
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve((y_test == c).astype(int), y_proba[:, c])
        roc_auc_vals[c] = auc(fpr[c], tpr[c])

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Neutral", "Bearish", "Bullish"]
    for c in range(n_classes):
        ax.plot(fpr[c], tpr[c], label=f"{labels[c]} (AUC={roc_auc_vals[c]:.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{symbol} – ROC Curves (Hybrid XGB)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)
    fig.savefig(os.path.join(chart_dir, f"{symbol}_roc_curves.png"), dpi=150)
    plt.close(fig)


def _save_shap(
    df: pd.DataFrame,
    feature_cols: list[str],
    symbol: str,
    output_dir: str,
):
    """Train a model on full train set and compute SHAP values."""
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping SHAP analysis.")
        return

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    X_train = train[feature_cols]
    y_train = train["label"]

    model = _xgb_model(n_classes=int(y_train.nunique()))
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train.iloc[-200:])

    # Mean absolute SHAP per feature (average across classes)
    if isinstance(shap_values, list):
        mean_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance_df.to_csv(
        os.path.join(output_dir, f"{symbol}_feature_importance.csv"), index=False
    )

    # Bar plot of top 20 features
    top = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"{symbol} – Top 20 Features by SHAP Importance")

    # Highlight sentiment features
    for i, feat in enumerate(top["feature"][::-1]):
        if feat.startswith("sent_"):
            ax.get_children()[i].set_color("orange")

    fig.tight_layout()
    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)
    fig.savefig(
        os.path.join(chart_dir, f"{symbol}_shap_importance.png"), dpi=150
    )
    plt.close(fig)
    print(f"  SHAP analysis saved -> {chart_dir}/{symbol}_shap_importance.png")


def save_shap_from_model(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    output_dir: str = "results",
    sample_size: int = 500,
):
    """
    Compute SHAP values for an already-trained model and save a CSV plus
    bar chart of mean absolute feature importance.

    Parameters
    ----------
    model : fitted tree model
        Typically an XGBoost model bundle["model"].
    X : DataFrame
        Feature matrix to explain.
    feature_cols : list[str]
        Feature column names in model order.
    label : str
        Output file prefix, e.g. "universal_test" or "AAPL_pretrained".
    sample_size : int
        Maximum number of rows to explain.  Rows are sampled randomly for
        speed and memory safety on larger datasets.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping SHAP analysis.")
        return

    if X.empty:
        raise ValueError("No rows available for SHAP analysis.")

    os.makedirs(output_dir, exist_ok=True)
    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)

    X_use = X[feature_cols]
    if len(X_use) > sample_size:
        X_use = X_use.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_use)

    # Mean absolute SHAP per feature (average across classes)
    if isinstance(shap_values, list):
        mean_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            mean_shap = np.abs(shap_arr).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(shap_arr).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )
    csv_path = os.path.join(output_dir, f"{label}_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)

    top = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"{label} - Top 20 Features by SHAP Importance")

    for bar, feat in zip(bars, top["feature"][::-1]):
        if feat.startswith("sent_"):
            bar.set_color("orange")

    fig.tight_layout()
    png_path = os.path.join(chart_dir, f"{label}_shap_importance.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"  SHAP CSV saved -> {csv_path}")
    print(f"  SHAP chart saved -> {png_path}")


def _save_comparison_table(results: dict, symbol: str, output_dir: str):
    """Save a CSV comparing all configs side-by-side."""
    rows = []
    for config_name, data in results["configs"].items():
        row = {"config": config_name, "n_features": data["n_features"]}
        for k, v in data["train_test"].items():
            row[f"tt_{k}"] = v
        for k, v in data["walk_forward"].items():
            row[f"wf_{k}"] = v
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{symbol}_comparison.csv")
    comp_df.to_csv(path, index=False)
    print(f"  Comparison table -> {path}")


# ---------------------------------------------------------------------------
# Run experiments for multiple tickers
# ---------------------------------------------------------------------------

def run_all_experiments(
    tickers: list[str],
    start: str = "2022-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.01,
    n_splits: int = 5,
    output_dir: str = "results",
) -> dict[str, dict]:
    """Run experiments for a list of tickers and return all results."""
    all_results = {}
    for ticker in tickers:
        try:
            all_results[ticker] = run_experiment(
                ticker,
                start=start,
                end=end,
                lookahead=lookahead,
                thresh=thresh,
                n_splits=n_splits,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"  ERROR with {ticker}: {e}")
            all_results[ticker] = {"error": str(e)}

    # Cross-ticker summary
    _save_cross_ticker_summary(all_results, output_dir)
    return all_results


def _save_cross_ticker_summary(all_results: dict, output_dir: str):
    rows = []
    for symbol, res in all_results.items():
        if "error" in res:
            continue
        for config_name, data in res.get("configs", {}).items():
            rows.append(
                {
                    "symbol": symbol,
                    "config": config_name,
                    "tt_accuracy": data["train_test"]["accuracy"],
                    "tt_f1_macro": data["train_test"]["f1_macro"],
                    "wf_accuracy_mean": data["walk_forward"]["accuracy_mean"],
                    "wf_f1_macro_mean": data["walk_forward"]["f1_macro_mean"],
                }
            )
    if rows:
        summary = pd.DataFrame(rows)
        path = os.path.join(output_dir, "cross_ticker_summary.csv")
        summary.to_csv(path, index=False)
        print(f"\nCross-ticker summary -> {path}")


# ---------------------------------------------------------------------------
# Model persistence (pretrain / save / load)
# ---------------------------------------------------------------------------

MODELS_DIR = "models"


def pretrain_and_save(
    symbol: str,
    start: str = "2022-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.01,
    include_sentiment: bool = True,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Train a hybrid XGBoost model on all available data for a ticker
    and persist it to disk.

    Saves to models/{symbol}_model.pkl containing:
      - model: the trained XGBClassifier
      - feature_columns: list of feature names (in order)
      - include_sentiment: whether sentiment features were used
      - metadata: symbol, date range, lookahead, threshold, train size
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PRE-TRAINING: {symbol}")
    print(f"{'='*60}")

    df = build_feature_matrix(
        symbol,
        start=start,
        end=end,
        lookahead=lookahead,
        thresh=thresh,
        include_sentiment=include_sentiment,
    )

    feature_cols = get_feature_columns(df, include_sentiment=include_sentiment)
    X = df[feature_cols]
    y = df["label"]

    n_classes = int(y.nunique())

    # Evaluate on last 20% using a model trained on first 80% only
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    eval_model = _xgb_model(n_classes=n_classes)
    print(f"  Evaluating: train on {len(X_train)} samples, test on {len(X_test)} samples ...")
    eval_model.fit(X_train, y_train)

    y_pred = eval_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  Holdout accuracy: {acc:.4f}  F1-macro: {f1:.4f}")

    # Train final production model on ALL data
    model_final = _xgb_model(n_classes=n_classes)
    print(f"  Retraining on all {len(X)} samples for production model ...")
    model_final.fit(X, y)

    bundle = {
        "model": model_final,
        "feature_columns": feature_cols,
        "include_sentiment": include_sentiment,
        "metadata": {
            "symbol": symbol,
            "start": start,
            "end": end or "latest",
            "lookahead": lookahead,
            "threshold": thresh,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "holdout_accuracy": acc,
            "holdout_f1_macro": f1,
        },
    }

    path = os.path.join(models_dir, f"{symbol}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model saved -> {path}")

    return bundle


def load_pretrained(symbol: str, models_dir: str = MODELS_DIR) -> dict:
    """
    Load a pretrained model bundle from disk.

    Returns dict with keys: model, feature_columns, include_sentiment, metadata.
    """
    path = os.path.join(models_dir, f"{symbol}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pretrained model for {symbol}. Run pretrain first."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Loaded pretrained model for {bundle['metadata']['symbol']}")
    print(f"  Trained on {bundle['metadata']['n_samples']} samples, "
          f"{bundle['metadata']['n_features']} features")
    return bundle


def pretrain_multiple(
    tickers: list[str],
    start: str = "2022-01-01",
    end: str | None = None,
    include_sentiment: bool = True,
):
    """Pretrain and save models for multiple tickers."""
    for ticker in tickers:
        try:
            pretrain_and_save(
                ticker, start=start, end=end, include_sentiment=include_sentiment
            )
        except Exception as e:
            print(f"  ERROR pre-training {ticker}: {e}")


# ---------------------------------------------------------------------------
# Universal (cross-stock) model training
# ---------------------------------------------------------------------------

def train_universal_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    train_end: str = "2023-12-31",
    val_end: str | None = "2024-06-30",
    xgb_params: dict | None = None,
    tune: bool = False,
    tune_trials: int = 50,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Train a single XGBoost model on pooled multi-stock data.

    Parameters
    ----------
    df : DataFrame
        Output of ``build_universal_dataset()`` — must contain ``ticker``,
        ``sector_id``, ``label``, and all feature columns.
    feature_cols : list[str]
        Column names to use as features (should exclude OHLCV, label, ticker).
    train_end / val_end : str
        Date boundaries for temporal train / val / test split.
    xgb_params : dict or None
        Pre-tuned XGBoost params.  If None, defaults are used (or tuned).
    tune : bool
        If True, run Optuna hyperparameter search on the training fold.
    models_dir : str
        Directory to save the model bundle.

    Returns
    -------
    dict  – bundle with model, feature_columns, metadata, and evaluation
    """
    from src.ml.universe import temporal_train_test_split

    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TRAINING UNIVERSAL MODEL")
    print(f"{'='*60}")

    if val_end:
        train_df, val_df, test_df = temporal_train_test_split(df, train_end, val_end)
    else:
        train_df, test_df = temporal_train_test_split(df, train_end)
        val_df = None

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    n_classes = int(y_train.nunique())

    print(f"  Train: {len(train_df)} rows")
    if val_df is not None:
        print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    print(f"  Features: {len(feature_cols)}")

    # Optional hyperparameter tuning on training fold
    if tune:
        print("\n  Running Optuna hyperparameter search ...")
        xgb_params = tune_hyperparameters(
            train_df, feature_cols, label_col=label_col, n_trials=tune_trials,
        )

    if xgb_params:
        model = _xgb_model_with_params(xgb_params, n_classes=n_classes)
    else:
        model = _xgb_model(n_classes=n_classes)

    # Early stopping using validation set
    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feature_cols]
        y_val = val_df[label_col]
        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  Test results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    F1-macro:  {f1:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")

    # Per-sector breakdown
    sector_metrics = _compute_sector_metrics(test_df, feature_cols, model, label_col)

    # Detect whether sentiment features are in the feature set
    include_sentiment = any(c.startswith("sent_") for c in feature_cols)

    bundle = {
        "model": model,
        "feature_columns": feature_cols,
        "model_type": "universal",
        "include_sentiment": include_sentiment,
        "xgb_params": xgb_params,
        "metadata": {
            "train_end": train_end,
            "val_end": val_end,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(feature_cols),
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "sector_metrics": sector_metrics,
        },
    }

    path = os.path.join(models_dir, "universal_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Universal model saved -> {path}")

    return bundle


def finetune_for_sector(
    base_bundle: dict,
    df: pd.DataFrame,
    sector_id: int,
    feature_cols: list[str],
    label_col: str = "label",
    n_extra_rounds: int = 50,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Fine-tune a universal base model on data from a single sector using
    XGBoost warm-start (``xgb_model`` parameter of fit).

    Parameters
    ----------
    base_bundle : dict
        Bundle returned by ``train_universal_model()``.
    sector_id : int
        Integer sector id to filter on.
    n_extra_rounds : int
        How many additional boosting rounds to run on sector data.
    """
    os.makedirs(models_dir, exist_ok=True)

    sector_df = df[df["sector_id"] == sector_id].copy()
    if sector_df.empty:
        raise ValueError(f"No data for sector_id={sector_id}")

    X_sector = sector_df[feature_cols]
    y_sector = sector_df[label_col]

    base_model = base_bundle["model"]
    n_classes = len(base_model.classes_)

    # Build a new model that warm-starts from the base model's booster
    ft_model = XGBClassifier(
        n_estimators=n_extra_rounds,
        learning_rate=0.01,
        max_depth=base_model.get_params().get("max_depth", 6),
        subsample=base_model.get_params().get("subsample", 0.9),
        colsample_bytree=base_model.get_params().get("colsample_bytree", 0.8),
        objective="multi:softprob",
        num_class=n_classes,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
    )
    ft_model.fit(X_sector, y_sector, xgb_model=base_model.get_booster())

    # Evaluate on the sector data (in-sample, but useful for sanity check)
    y_pred = ft_model.predict(X_sector)
    acc = accuracy_score(y_sector, y_pred)
    f1 = f1_score(y_sector, y_pred, average="macro")
    print(f"  Sector {sector_id} fine-tune: acc={acc:.4f}  F1={f1:.4f} ({len(sector_df)} rows)")

    bundle = {
        "model": ft_model,
        "feature_columns": feature_cols,
        "model_type": "sector_finetuned",
        "metadata": {
            "sector_id": sector_id,
            "n_samples": len(sector_df),
            "n_extra_rounds": n_extra_rounds,
            "sector_accuracy": acc,
            "sector_f1_macro": f1,
        },
    }

    path = os.path.join(models_dir, f"sector_{sector_id}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Fine-tuned model saved -> {path}")

    return bundle


def finetune_all_sectors(
    base_bundle: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    n_extra_rounds: int = 50,
    models_dir: str = MODELS_DIR,
) -> dict[int, dict]:
    """Fine-tune the base model for every sector present in the data."""
    results = {}
    sector_ids = sorted(df["sector_id"].unique())

    print(f"\n{'='*60}")
    print(f"  SECTOR FINE-TUNING ({len(sector_ids)} sectors)")
    print(f"{'='*60}")

    for sid in sector_ids:
        if sid < 0:
            continue
        try:
            results[sid] = finetune_for_sector(
                base_bundle, df, sid, feature_cols,
                label_col=label_col, n_extra_rounds=n_extra_rounds,
                models_dir=models_dir,
            )
        except Exception as e:
            print(f"  ERROR sector {sid}: {e}")
            results[sid] = {"error": str(e)}
    return results


def finetune_with_sentiment(
    base_bundle: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    sentiment_start: str,
    label_col: str = "label",
    n_extra_rounds: int = 100,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Phase 2 of two-phase training: warm-start the base universal model
    on recent data where sentiment features are available.

    The base model was trained with sentiment columns present but zeroed
    out.  This fine-tune step adds trees that can split on real sentiment
    values from the recent window.

    Parameters
    ----------
    base_bundle : dict
        Bundle from ``train_universal_model()`` (Phase 1).
    df : pd.DataFrame
        Full dataset (must include sentiment columns).
    sentiment_start : str
        Date from which sentiment data is available.  Only rows on or
        after this date are used for fine-tuning.
    n_extra_rounds : int
        Additional boosting rounds on the sentiment-rich data.
    """
    os.makedirs(models_dir, exist_ok=True)

    recent = df[df.index >= pd.Timestamp(sentiment_start)].copy()
    if recent.empty:
        raise ValueError(f"No data on or after {sentiment_start}")

    X_recent = recent[feature_cols]
    y_recent = recent[label_col]

    base_model = base_bundle["model"]
    n_classes = len(base_model.classes_)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: SENTIMENT FINE-TUNING")
    print(f"  Data from {sentiment_start}: {len(recent)} rows")
    print(f"  Extra boosting rounds: {n_extra_rounds}")
    print(f"{'='*60}")

    ft_model = XGBClassifier(
        n_estimators=n_extra_rounds,
        learning_rate=0.01,
        max_depth=base_model.get_params().get("max_depth", 6),
        subsample=base_model.get_params().get("subsample", 0.9),
        colsample_bytree=base_model.get_params().get("colsample_bytree", 0.8),
        objective="multi:softprob",
        num_class=n_classes,
        n_jobs=-1,
        eval_metric="mlogloss",
        random_state=42,
    )
    ft_model.fit(X_recent, y_recent, xgb_model=base_model.get_booster())

    y_pred = ft_model.predict(X_recent)
    acc = accuracy_score(y_recent, y_pred)
    f1 = f1_score(y_recent, y_pred, average="macro")
    print(f"\n  In-sample results (sentiment window):")
    print(f"    Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")

    bundle = {
        "model": ft_model,
        "feature_columns": feature_cols,
        "model_type": "universal",
        "include_sentiment": True,
        "xgb_params": base_bundle.get("xgb_params"),
        "metadata": {
            **base_bundle["metadata"],
            "sentiment_finetuned": True,
            "sentiment_start": sentiment_start,
            "n_sentiment_samples": len(recent),
            "n_extra_rounds": n_extra_rounds,
            "sentiment_ft_accuracy": acc,
            "sentiment_ft_f1_macro": f1,
        },
    }

    path = os.path.join(models_dir, "universal_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Sentiment-tuned universal model saved -> {path}")

    return bundle


def _compute_sector_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    label_col: str = "label",
) -> dict:
    """Compute per-sector accuracy and F1 on a test DataFrame."""
    metrics = {}
    if "sector_id" not in df.columns:
        return metrics

    for sid in sorted(df["sector_id"].unique()):
        if sid < 0:
            continue
        sub = df[df["sector_id"] == sid]
        if len(sub) < 10:
            continue
        y_true = sub[label_col]
        y_pred = model.predict(sub[feature_cols])
        metrics[int(sid)] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "n_samples": len(sub),
        }
    return metrics


def load_universal_model(models_dir: str = MODELS_DIR) -> dict:
    """Load the universal model bundle."""
    path = os.path.join(models_dir, "universal_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("No universal model found. Train one first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    meta = bundle["metadata"]
    print(f"  Loaded universal model: {meta['n_features']} features, "
          f"trained on {meta['n_train']} samples")
    return bundle


def load_sector_model(sector_id: int, models_dir: str = MODELS_DIR) -> dict:
    """Load a sector-fine-tuned model bundle."""
    path = os.path.join(models_dir, f"sector_{sector_id}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No fine-tuned model for sector {sector_id}.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle
