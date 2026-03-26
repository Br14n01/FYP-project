"""
Feature engineering: technical indicators, label generation, and
merging with sentiment features.

Usage:
    from src.ml.features import build_feature_matrix
    df = build_feature_matrix("AAPL", start="2022-01-01")
"""

import os

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from src.sentiment.sentiment_features import load_daily_sentiment


# ---------------------------------------------------------------------------
# Technical indicators (extracted from financial_data.ipynb)
# ---------------------------------------------------------------------------

def _merge(df_main: pd.DataFrame, ta_obj) -> pd.DataFrame:
    if ta_obj is None:
        return df_main
    if isinstance(ta_obj, pd.DataFrame):
        return pd.concat([df_main, ta_obj], axis=1)
    return pd.concat([df_main, ta_obj.rename(ta_obj.name)], axis=1)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Momentum
    for length in [5, 10, 15]:
        df[f"rsi_{length}"] = ta.rsi(df["Close"], length=length)
    df["roc_10"] = ta.roc(df["Close"], length=10)
    df["mom_10"] = ta.mom(df["Close"], length=10)

    # Oscillators
    df = _merge(df, ta.stochrsi(df["Close"]))
    df["cci_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    df["wr_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    df = _merge(df, ta.kst(df["Close"]))
    df["macd"] = ta.macd(df["Close"])["MACD_12_26_9"]

    # Trend
    for length in [5, 10, 20]:
        df[f"sma_{length}"] = ta.sma(df["Close"], length=length)
        df[f"ema_{length}"] = ta.ema(df["Close"], length=length)
    df["vwma_20"] = ta.vwma(df["Close"], df["Volume"], length=20)

    # Volatility
    df = _merge(df, ta.bbands(df["Close"], length=20))
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df = _merge(df, ta.kc(df["High"], df["Low"], df["Close"], length=20))

    # Volume
    _obv = ta.obv(df["Close"], df["Volume"])
    if _obv is not None:
        df["obv"] = _obv
    _ad = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"])
    if _ad is not None:
        df["ad"] = _ad
    _efi = ta.efi(df["Close"], df["Volume"])
    if _efi is not None:
        df["efi"] = _efi
    df = _merge(df, ta.nvi(df["Close"], df["Volume"]))
    df = _merge(df, ta.pvi(df["Close"], df["Volume"]))

    return df


# ---------------------------------------------------------------------------
# Label generation (from notebook)
# ---------------------------------------------------------------------------

def generate_label(
    data: pd.DataFrame,
    lookahead: int = 5,
    thresh: float = 0.01,
    col: str = "Close",
) -> pd.Series:
    """
    3-class label from forward-looking mean close:
      2 = bullish, 1 = bearish, 0 = neutral.
    """
    future_mean = (
        data[col]
        .shift(-lookahead)
        .rolling(window=lookahead, min_periods=lookahead)
        .mean()
    )
    pct_change = (future_mean - data[col]) / data[col]
    labels = np.select(
        [pct_change >= thresh, pct_change <= -thresh],
        [2, 1],
        default=0,
    )
    return pd.Series(labels, index=data.index)


# ---------------------------------------------------------------------------
# Price data download
# ---------------------------------------------------------------------------

def download_price_data(
    symbol: str,
    start: str = "2022-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Combined feature matrix
# ---------------------------------------------------------------------------

OHLCV_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
SENTIMENT_FEATURE_COLS = [
    "sent_mean",
    "sent_std",
    "sent_count",
    "sent_max",
    "sent_min",
    "sent_positive_ratio",
    "sent_negative_ratio",
    "sent_momentum_3d",
    "sent_momentum_5d",
    "sent_momentum_10d",
    "sent_vol_5d",
]


def build_feature_matrix(
    symbol: str,
    start: str = "2022-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.01,
    include_sentiment: bool = True,
    dataset_dir: str = "dataset",
) -> pd.DataFrame:
    """
    Build a complete feature matrix for one ticker:
      1. Download price data
      2. Compute technical indicators
      3. Optionally merge daily sentiment features
      4. Generate labels

    Returns a DataFrame ready for train/test splitting.
    """
    # Price + indicators
    df = download_price_data(symbol, start=start, end=end)
    df_ta = add_indicators(df)

    # Sentiment merge
    if include_sentiment:
        try:
            sent = load_daily_sentiment(symbol, dataset_dir=dataset_dir)
            sent.index = pd.to_datetime(sent.index).tz_localize(None)
            df_ta.index = pd.to_datetime(df_ta.index).tz_localize(None)
            df_ta = df_ta.join(sent, how="left")

            for col in SENTIMENT_FEATURE_COLS:
                if col in df_ta.columns:
                    df_ta[col] = df_ta[col].ffill().fillna(0.0)
        except FileNotFoundError:
            print(f"  Warning: no sentiment data for {symbol}, skipping.")
            include_sentiment = False

    # Labels
    df_ta["label"] = generate_label(df_ta, lookahead=lookahead, thresh=thresh)
    df_ta.dropna(inplace=True)

    return df_ta


def get_feature_columns(
    df: pd.DataFrame,
    include_sentiment: bool = True,
) -> list[str]:
    """Return the list of feature column names (excluding OHLCV and labels)."""
    exclude = OHLCV_COLS | {c for c in df.columns if c.startswith("label")}
    cols = [c for c in df.columns if c not in exclude]
    if not include_sentiment:
        cols = [c for c in cols if c not in SENTIMENT_FEATURE_COLS]
    return cols


def get_sentiment_only_columns() -> list[str]:
    """Return just the sentiment feature column names."""
    return list(SENTIMENT_FEATURE_COLS)
