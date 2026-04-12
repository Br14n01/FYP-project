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
# Scale-invariant / relative features for cross-stock generalization
# ---------------------------------------------------------------------------

def add_relative_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scale-invariant features suitable for a universal cross-stock
    model.  Every feature produced here is either a ratio, a percentage,
    or an oscillator — so AAPL at $170 and JPM at $220 produce comparable
    values.
    """
    df = data.copy()
    close = df["Close"]
    volume = df["Volume"]

    # --- Price-relative (MA ratios) ----------------------------------------
    for length in [5, 10, 20]:
        sma = ta.sma(close, length=length)
        ema = ta.ema(close, length=length)
        if sma is not None:
            df[f"close_to_sma{length}"] = close / sma - 1.0
        if ema is not None:
            df[f"close_to_ema{length}"] = close / ema - 1.0

    sma5 = ta.sma(close, length=5)
    sma20 = ta.sma(close, length=20)
    if sma5 is not None and sma20 is not None:
        df["sma5_to_sma20"] = sma5 / sma20 - 1.0
    sma10 = ta.sma(close, length=10)
    if sma10 is not None and sma20 is not None:
        df["sma10_to_sma20"] = sma10 / sma20 - 1.0

    vwma = ta.vwma(close, volume, length=20)
    if vwma is not None:
        df["close_to_vwma20"] = close / vwma - 1.0

    # --- Bollinger %B (position within bands) ------------------------------
    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        bbu = bbands.filter(like="BBU").iloc[:, 0]
        bbl = bbands.filter(like="BBL").iloc[:, 0]
        band_width = bbu - bbl
        df["bb_pctb"] = (close - bbl) / band_width.replace(0, np.nan)
        df["bb_width_pct"] = band_width / close

    # --- Keltner Channel relative position ---------------------------------
    kc = ta.kc(df["High"], df["Low"], close, length=20)
    if kc is not None:
        kcu = kc.filter(like="KCU").iloc[:, 0]
        kcl = kc.filter(like="KCL").iloc[:, 0]
        kc_width = kcu - kcl
        df["kc_pctb"] = (close - kcl) / kc_width.replace(0, np.nan)

    # --- ATR as percentage of price ----------------------------------------
    atr = ta.atr(df["High"], df["Low"], close, length=14)
    if atr is not None:
        df["atr_pct"] = atr / close

    # --- Log returns at multiple horizons ----------------------------------
    for horizon in [1, 5, 10, 20]:
        df[f"log_return_{horizon}d"] = np.log(close / close.shift(horizon))

    # --- Realized volatility (rolling std of 1-day log returns) ------------
    log_ret_1d = np.log(close / close.shift(1))
    for window in [5, 10, 20]:
        df[f"realized_vol_{window}d"] = log_ret_1d.rolling(window).std()

    # --- Return acceleration (change in 5d return) -------------------------
    ret_5d = close.pct_change(5)
    df["return_accel"] = ret_5d - ret_5d.shift(5)

    # --- Volume relative features ------------------------------------------
    vol_sma5 = ta.sma(volume, length=5)
    vol_sma20 = ta.sma(volume, length=20)
    if vol_sma5 is not None:
        df["volume_ratio_5"] = volume / vol_sma5.replace(0, np.nan)
    if vol_sma20 is not None:
        df["volume_ratio_20"] = volume / vol_sma20.replace(0, np.nan)

    # --- Cross-timeframe momentum ------------------------------------------
    rsi_14 = ta.rsi(close, length=14)
    if rsi_14 is not None:
        df["rsi_roc_5d"] = rsi_14 - rsi_14.shift(5)

    roc_10 = ta.roc(close, length=10)
    if roc_10 is not None:
        df["momentum_accel"] = roc_10 - roc_10.shift(5)

    # --- Calendar features (cyclical encoding) -----------------------------
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
        month = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


# ---------------------------------------------------------------------------
# Label generation (from notebook)
# ---------------------------------------------------------------------------

def generate_label(
    data: pd.DataFrame,
    lookahead: int = 5,
    thresh: float = 0.02,
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


def generate_adaptive_label(
    data: pd.DataFrame,
    lookahead: int = 5,
    percentile: float = 33.0,
    rolling_window: int = 60,
    col: str = "Close",
) -> pd.Series:
    """
    3-class label using rolling percentile thresholds instead of a fixed
    percentage.  This keeps class balance roughly stable across different
    volatility regimes.

      2 = bullish (return above upper percentile)
      1 = bearish (return below lower percentile)
      0 = neutral  (in between)
    """
    future_mean = (
        data[col]
        .shift(-lookahead)
        .rolling(window=lookahead, min_periods=lookahead)
        .mean()
    )
    pct_change = (future_mean - data[col]) / data[col]

    upper = pct_change.rolling(rolling_window, min_periods=rolling_window).quantile(
        1.0 - percentile / 100.0
    )
    lower = pct_change.rolling(rolling_window, min_periods=rolling_window).quantile(
        percentile / 100.0
    )

    labels = np.select(
        [pct_change >= upper, pct_change <= lower],
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
SCALE_INVARIANT_FEATURE_COLS = [
    "close_to_sma5",
    "close_to_ema5",
    "close_to_sma10",
    "close_to_ema10",
    "close_to_sma20",
    "close_to_ema20",
    "sma5_to_sma20",
    "sma10_to_sma20",
    "close_to_vwma20",
    "bb_pctb",
    "bb_width_pct",
    "kc_pctb",
    "atr_pct",
    "log_return_1d",
    "log_return_5d",
    "log_return_10d",
    "log_return_20d",
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_20d",
    "return_accel",
    "volume_ratio_5",
    "volume_ratio_20",
    "rsi_roc_5d",
    "momentum_accel",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def build_feature_matrix(
    symbol: str,
    start: str = "2022-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.02,
    include_sentiment: bool = True,
    use_relative_features: bool = False,
    adaptive_label: bool = False,
    dataset_dir: str = "dataset",
    sector_id: int | None = None,
) -> pd.DataFrame:
    """
    Build a complete feature matrix for one ticker:
      1. Download price data
      2. Compute technical indicators
      3. Optionally compute scale-invariant relative indicators
      4. Optionally merge daily sentiment features
      5. Generate labels (fixed-threshold or adaptive)

    Parameters
    ----------
    use_relative_features : bool
        If True, also compute scale-invariant ratio / return features
        suitable for a universal cross-stock model.
    adaptive_label : bool
        If True, use rolling-percentile adaptive thresholds instead of
        fixed ``thresh`` for label generation.
    sector_id : int or None
        If provided, add a ``sector_id`` column (for cross-stock models).

    Returns a DataFrame ready for train/test splitting.
    """
    # Price + indicators
    df = download_price_data(symbol, start=start, end=end)
    df_ta = add_indicators(df)

    if use_relative_features:
        df_ta = add_relative_indicators(df_ta)

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
    if adaptive_label:
        df_ta["label"] = generate_adaptive_label(
            df_ta, lookahead=lookahead, percentile=thresh
        )
    else:
        df_ta["label"] = generate_label(
            df_ta, lookahead=lookahead, thresh=thresh
        )

    # Sector identifier
    if sector_id is not None:
        df_ta["sector_id"] = sector_id

    df_ta.dropna(inplace=True)

    return df_ta


METADATA_COLS = {"symbol", "ticker"}


def get_feature_columns(
    df: pd.DataFrame,
    include_sentiment: bool = True,
) -> list[str]:
    """Return the list of feature column names (excluding OHLCV, labels, and metadata)."""
    exclude = OHLCV_COLS | METADATA_COLS | {c for c in df.columns if c.startswith("label")}
    cols = [c for c in df.columns if c not in exclude]
    if not include_sentiment:
        cols = [c for c in cols if c not in SENTIMENT_FEATURE_COLS]
    return cols


def get_universal_feature_columns(
    df: pd.DataFrame,
    include_sentiment: bool = True,
    scale_invariant_only: bool = False,
) -> list[str]:
    """Return universal-model features, optionally restricting to relative-only features."""
    if not scale_invariant_only:
        return get_feature_columns(df, include_sentiment=include_sentiment)

    cols = [c for c in SCALE_INVARIANT_FEATURE_COLS if c in df.columns]
    if include_sentiment:
        cols.extend(c for c in SENTIMENT_FEATURE_COLS if c in df.columns)
    return cols


def get_sentiment_only_columns() -> list[str]:
    """Return just the sentiment feature column names."""
    return list(SENTIMENT_FEATURE_COLS)
