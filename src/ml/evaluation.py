"""
Evaluation utilities for the universal (cross-stock) model.

Provides held-out-stock evaluation, per-sector performance breakdown,
and formatted reporting.

Usage:
    from src.ml.evaluation import evaluate_universal_model
    report = evaluate_universal_model(bundle, test_df, feature_cols)
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from src.ml.universe import SECTORS, SECTOR_ID


SECTOR_NAME_BY_ID = {v: k for k, v in SECTOR_ID.items()}


def evaluate_universal_model(
    bundle: dict,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    output_dir: str = "results",
    report_label: str = "universal",
) -> dict:
    """
    Comprehensive evaluation of a universal model on a test set.

    Returns a dict with overall metrics, per-sector metrics, and
    per-ticker metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = bundle["model"]

    X_test = test_df[feature_cols]
    y_test = test_df[label_col]
    y_pred = model.predict(X_test)

    overall = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_samples": len(test_df),
    }

    sector_metrics = _per_sector_metrics(test_df, feature_cols, model, label_col)
    ticker_metrics = _per_ticker_metrics(test_df, feature_cols, model, label_col)

    report = {
        "overall": overall,
        "per_sector": sector_metrics,
        "per_ticker": ticker_metrics,
    }

    _print_report(report)
    _save_report(report, output_dir)
    _save_confusion_matrix(y_test, y_pred, report_label, output_dir)
    _plot_sector_comparison(sector_metrics, output_dir)

    return report


def evaluate_held_out_stocks(
    bundle: dict,
    df: pd.DataFrame,
    held_out_tickers: list[str],
    feature_cols: list[str],
    label_col: str = "label",
    output_dir: str = "results",
    report_label: str = "universal_held_out",
) -> dict:
    """
    Evaluate model on stocks it has never seen during training.
    """
    from src.ml.universe import held_out_stock_split

    _, held_out_df = held_out_stock_split(df, held_out_tickers)

    if held_out_df.empty:
        print("  No held-out stock data found.")
        return {}

    print(f"\n{'='*60}")
    print(f"  HELD-OUT STOCK EVALUATION")
    print(f"  Stocks: {', '.join(held_out_tickers)}")
    print(f"  Samples: {len(held_out_df)}")
    print(f"{'='*60}")

    return evaluate_universal_model(
        bundle, held_out_df, feature_cols, label_col, output_dir, report_label
    )


def _per_sector_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    label_col: str,
) -> dict:
    metrics = {}
    if "sector_id" not in df.columns:
        return metrics
    for sid in sorted(df["sector_id"].unique()):
        if sid < 0:
            continue
        sub = df[df["sector_id"] == sid]
        if len(sub) < 5:
            continue
        y_true = sub[label_col]
        y_pred = model.predict(sub[feature_cols])
        name = SECTOR_NAME_BY_ID.get(sid, f"sector_{sid}")
        metrics[name] = {
            "sector_id": int(sid),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "n_samples": len(sub),
        }
    return metrics


def _per_ticker_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    label_col: str,
) -> dict:
    metrics = {}
    if "ticker" not in df.columns:
        return metrics
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        if len(sub) < 5:
            continue
        y_true = sub[label_col]
        y_pred = model.predict(sub[feature_cols])
        metrics[ticker] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "n_samples": len(sub),
        }
    return metrics


def _print_report(report: dict):
    overall = report["overall"]
    print(f"\n{'='*70}")
    print(f"  UNIVERSAL MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"  Overall ({overall['n_samples']} samples):")
    print(f"    Accuracy:   {overall['accuracy']:.4f}")
    print(f"    F1-macro:   {overall['f1_macro']:.4f}")
    print(f"    Precision:  {overall['precision_macro']:.4f}")
    print(f"    Recall:     {overall['recall_macro']:.4f}")

    if report["per_sector"]:
        print(f"\n  {'Sector':<16} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'N':>8}")
        print(f"  {'─'*56}")
        for name, m in sorted(report["per_sector"].items(), key=lambda x: x[1]["f1_macro"], reverse=True):
            print(
                f"  {name:<16} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} "
                f"{m['precision_macro']:>8.4f} {m['recall_macro']:>8.4f} {m['n_samples']:>8}"
            )

    if report["per_ticker"]:
        print(f"\n  {'Ticker':<8} {'Acc':>8} {'F1':>8} {'N':>8}")
        print(f"  {'─'*32}")
        for ticker, m in sorted(report["per_ticker"].items(), key=lambda x: x[1]["f1_macro"], reverse=True):
            print(f"  {ticker:<8} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} {m['n_samples']:>8}")

    print(f"{'='*70}\n")


def _save_report(report: dict, output_dir: str):
    import json
    path = os.path.join(output_dir, "universal_eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if report["per_sector"]:
        rows = [{"sector": k, **v} for k, v in report["per_sector"].items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "universal_sector_metrics.csv"), index=False
        )

    if report["per_ticker"]:
        rows = [{"ticker": k, **v} for k, v in report["per_ticker"].items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "universal_ticker_metrics.csv"), index=False
        )

    print(f"  Evaluation report saved -> {output_dir}/universal_eval_report.json")


def _plot_sector_comparison(sector_metrics: dict, output_dir: str):
    if not sector_metrics:
        return

    names = list(sector_metrics.keys())
    f1s = [sector_metrics[n]["f1_macro"] for n in names]
    accs = [sector_metrics[n]["accuracy"] for n in names]

    order = np.argsort(f1s)[::-1]
    names = [names[i] for i in order]
    f1s = [f1s[i] for i in order]
    accs = [accs[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, f1s, width, label="F1-macro")
    ax.bar(x + width / 2, accs, width, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Universal Model – Per-Sector Performance")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)
    fig.savefig(os.path.join(chart_dir, "universal_sector_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Sector comparison chart saved -> {chart_dir}/universal_sector_comparison.png")


def _save_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    label: str,
    output_dir: str,
):
    """Save a normalized confusion matrix for a universal-model evaluation."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)

    class_names = ["Neutral", "Bearish", "Bullish"]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{label.replace('_', ' ').title()} Confusion Matrix")

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            val = cmn[i, j]
            color = "white" if val >= 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    chart_dir = os.path.join(output_dir, "chart")
    os.makedirs(chart_dir, exist_ok=True)
    path = os.path.join(chart_dir, f"{label}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")
