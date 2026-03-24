"""
Aggregate per-article sentiment scores into daily trading features.

Usage:
    from src.sentiment.sentiment_features import build_daily_sentiment
    daily = build_daily_sentiment("dataset/AAPL_sentiment.csv")
"""

import os
import pandas as pd
import numpy as np


def build_daily_sentiment(
    sentiment_csv: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Collapse article-level sentiment into one row per calendar date.

    Input CSV must have columns: date, sentiment_label, sentiment_score,
    sentiment_numeric (produced by score_news_csv).

    Output columns
    --------------
    sent_mean           : daily mean of sentiment_numeric
    sent_std            : daily std  of sentiment_numeric (0 if single article)
    sent_count          : number of articles that day
    sent_positive_ratio : fraction of articles labelled 'positive'
    sent_negative_ratio : fraction of articles labelled 'negative'
    sent_max            : most positive score that day
    sent_min            : most negative score that day
    """
    df = pd.read_csv(sentiment_csv)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date").agg(
        sent_mean=("sentiment_numeric", "mean"),
        sent_std=("sentiment_numeric", "std"),
        sent_count=("sentiment_numeric", "count"),
        sent_max=("sentiment_numeric", "max"),
        sent_min=("sentiment_numeric", "min"),
        sent_positive_ratio=(
            "sentiment_label",
            lambda s: (s == "positive").mean(),
        ),
        sent_negative_ratio=(
            "sentiment_label",
            lambda s: (s == "negative").mean(),
        ),
    )

    daily["sent_std"] = daily["sent_std"].fillna(0.0)

    # Rolling / momentum features
    daily.sort_index(inplace=True)
    daily["sent_momentum_3d"] = daily["sent_mean"].rolling(3, min_periods=1).mean()
    daily["sent_momentum_5d"] = daily["sent_mean"].rolling(5, min_periods=1).mean()
    daily["sent_momentum_10d"] = daily["sent_mean"].rolling(10, min_periods=1).mean()
    daily["sent_vol_5d"] = daily["sent_mean"].rolling(5, min_periods=1).std().fillna(0)

    if save:
        out_path = sentiment_csv.replace("_sentiment.csv", "_daily_sentiment.csv")
        daily.to_csv(out_path)
        print(f"Saved daily sentiment -> {out_path}")

    return daily


def load_daily_sentiment(ticker: str, dataset_dir: str = "dataset") -> pd.DataFrame:
    """Load pre-computed daily sentiment features for a ticker."""
    path = os.path.join(dataset_dir, f"{ticker}_daily_sentiment.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the sentiment pipeline first."
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df
