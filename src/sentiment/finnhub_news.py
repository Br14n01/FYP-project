"""
Fetch historical company news from Finnhub and store as CSV.

Usage:
    from src.sentiment.finnhub_news import fetch_historical_news
    df = fetch_historical_news("AAPL", "2023-01-01", "2024-12-31")
"""

import os
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd


def _get_client() -> finnhub.Client:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        raise ValueError(
            "FINNHUB_API_KEY not set. Add it to your .env file."
        )
    return finnhub.Client(api_key=api_key)


def fetch_historical_news(
    symbol: str,
    start_date: str,
    end_date: str,
    save: bool = True,
    rate_limit_pause: float = 1.1,
) -> pd.DataFrame:
    """
    Fetch company news from Finnhub in weekly chunks to stay within API limits.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    start_date : str
        ISO date string for the start of the range (e.g. "2023-01-01").
    end_date : str
        ISO date string for the end of the range (e.g. "2024-12-31").
    save : bool
        If True, persist the result to dataset/{symbol}_news.csv.
    rate_limit_pause : float
        Seconds to sleep between API calls (free tier: 60 req/min).

    Returns
    -------
    pd.DataFrame
        Columns: date, datetime, headline, summary, source, url, category.
    """
    client = _get_client()
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_articles: list[dict] = []
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=6), end)
        from_str = cursor.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")

        print(f"  [{symbol}] Fetching news {from_str} -> {to_str} ...", end="")
        try:
            articles = client.company_news(symbol, _from=from_str, to=to_str)
        except Exception as e:
            print(f" ERROR: {e}")
            articles = []

        for a in articles:
            ts = a.get("datetime", 0)
            dt = datetime.utcfromtimestamp(ts) if ts else None
            all_articles.append(
                {
                    "date": dt.strftime("%Y-%m-%d") if dt else "",
                    "datetime": dt.isoformat() if dt else "",
                    "headline": a.get("headline", ""),
                    "summary": a.get("summary", ""),
                    "source": a.get("source", ""),
                    "url": a.get("url", ""),
                    "category": a.get("category", ""),
                }
            )
        print(f" {len(articles)} articles")

        cursor = chunk_end + timedelta(days=1)
        time.sleep(rate_limit_pause)

    df = pd.DataFrame(all_articles)
    if df.empty:
        print(f"  [{symbol}] No articles found in the date range.")
        return df

    df.sort_values("datetime", inplace=True)
    df.drop_duplicates(subset=["headline", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if save:
        os.makedirs("dataset", exist_ok=True)
        path = f"dataset/{symbol}_news.csv"
        df.to_csv(path, index=False)
        print(f"  [{symbol}] Saved {len(df)} articles -> {path}")

    return df


def fetch_news_for_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: fetch news for multiple tickers."""
    results = {}
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Fetching news for {ticker}")
        print(f"{'='*50}")
        results[ticker] = fetch_historical_news(ticker, start_date, end_date)
    return results
