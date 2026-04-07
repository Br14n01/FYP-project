"""
Stock universe definition and cross-stock dataset builder.

Defines sector-labelled ticker lists and a function to pool normalised
feature matrices from many tickers into a single training DataFrame for
the universal (generalised) model.

Usage:
    from src.ml.universe import build_universal_dataset, SECTOR_MAP
    df = build_universal_dataset(start="2019-01-01")
"""

import numpy as np
import pandas as pd

from src.ml.features import (
    build_feature_matrix,
    get_feature_columns,
    SENTIMENT_FEATURE_COLS,
)

# ---------------------------------------------------------------------------
# Sector definitions
# ---------------------------------------------------------------------------

SECTORS: dict[str, list[str]] = {
    "Technology":       ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "INTC", "AMD", "ORCL"],
    "Financials":       ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "USB"],
    "Energy":           ["CVX", "XOM", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "Healthcare":       ["UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "MDT", "BMY"],
    "Consumer":         ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW"],
    "Industrials":      ["CAT", "BA", "HON", "UPS", "GE", "MMM", "LMT", "RTX", "DE", "UNP"],
    "ETFs":             ["VOO", "SPY", "QQQ", "IWM", "DIA"],
}

SECTOR_ID: dict[str, int] = {name: idx for idx, name in enumerate(SECTORS)}

SECTOR_MAP: dict[str, str] = {}
for _sector, _tickers in SECTORS.items():
    for _t in _tickers:
        SECTOR_MAP[_t] = _sector


def get_sector_id(ticker: str) -> int:
    """Return the integer sector id for a ticker, or -1 if unknown."""
    sector = SECTOR_MAP.get(ticker)
    if sector is None:
        return -1
    return SECTOR_ID[sector]


def get_all_tickers() -> list[str]:
    """Return a flat list of every ticker in the universe."""
    return [t for tickers in SECTORS.values() for t in tickers]


def get_tickers_for_sector(sector: str) -> list[str]:
    return SECTORS.get(sector, [])


# ---------------------------------------------------------------------------
# Universal dataset builder
# ---------------------------------------------------------------------------

def build_universal_dataset(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
    lookahead: int = 5,
    thresh: float = 0.01,
    include_sentiment: bool = False,
    sentiment_start: str | None = None,
    adaptive_label: bool = False,
    dataset_dir: str = "dataset",
) -> pd.DataFrame:
    """
    Pool normalised feature matrices from many tickers into a single
    DataFrame suitable for training a universal model.

    Each row is tagged with ``ticker`` and ``sector_id`` columns so that
    sector-aware fine-tuning can filter later.

    Parameters
    ----------
    tickers : list[str] or None
        Tickers to include.  ``None`` uses the full universe.
    include_sentiment : bool
        Whether to merge sentiment features (requires pre-scored CSVs).
    sentiment_start : str or None
        If set together with ``include_sentiment=True``, sentiment
        columns are included in the feature set but zeroed out for
        dates before this cutoff.  This enables the two-phase training
        approach: Phase 1 trains on long-history technical features
        (sentiment = 0), Phase 2 fine-tunes on recent data where
        sentiment is available.  Format: ``"2025-04-01"``.
    adaptive_label : bool
        Use rolling-percentile adaptive labels instead of fixed threshold.
    """
    if tickers is None:
        tickers = get_all_tickers()

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        sid = get_sector_id(ticker)
        try:
            df = build_feature_matrix(
                ticker,
                start=start,
                end=end,
                lookahead=lookahead,
                thresh=thresh,
                include_sentiment=include_sentiment,
                use_relative_features=True,
                adaptive_label=adaptive_label,
                dataset_dir=dataset_dir,
                sector_id=sid if sid >= 0 else None,
            )
            df["ticker"] = ticker
            if sid < 0:
                df["sector_id"] = -1
            frames.append(df)
            print(f"  [{ticker}] {len(df)} rows")
        except Exception as e:
            print(f"  [{ticker}] FAILED: {e}")
            failed.append(ticker)

    if not frames:
        raise RuntimeError("No data collected for any ticker.")

    combined = pd.concat(frames, axis=0)
    combined.sort_index(inplace=True)

    # Ensure sentiment columns always exist when requested, even if
    # some tickers had no sentiment CSV (pd.concat fills with NaN).
    if include_sentiment:
        for col in SENTIMENT_FEATURE_COLS:
            if col not in combined.columns:
                combined[col] = 0.0
            else:
                combined[col] = combined[col].fillna(0.0)

        # Zero out sentiment for dates before the cutoff so Phase 1
        # of two-phase training sees only technical signal.
        if sentiment_start is not None:
            mask = combined.index < pd.Timestamp(sentiment_start)
            for col in SENTIMENT_FEATURE_COLS:
                combined.loc[mask, col] = 0.0
            n_zeroed = mask.sum()
            print(f"  Sentiment zeroed for {n_zeroed} rows before {sentiment_start}")

    print(f"\n  Universal dataset: {len(combined)} rows from "
          f"{len(frames)} tickers ({len(failed)} failed)")

    return combined


def temporal_train_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a pooled DataFrame by date (all stocks together).

    Returns (train, test) or (train, val, test) if ``val_end`` is given.
    """
    df = df.sort_index()
    train_mask = df.index <= pd.Timestamp(train_end)
    if val_end is not None:
        val_mask = (df.index > pd.Timestamp(train_end)) & (df.index <= pd.Timestamp(val_end))
        test_mask = df.index > pd.Timestamp(val_end)
        return df[train_mask], df[val_mask], df[test_mask]
    test_mask = ~train_mask
    return df[train_mask], df[test_mask]


def held_out_stock_split(
    df: pd.DataFrame,
    held_out_tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by ticker: some stocks are never seen during training.
    """
    mask = df["ticker"].isin(held_out_tickers)
    return df[~mask], df[mask]
