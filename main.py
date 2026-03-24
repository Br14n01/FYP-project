"""
CLI entry point for the Hybrid Sentiment + Technical Trading System.

Menu:
  1. Sentiment analysis (live news demo)
  2. Download stock data
  3. Collect historical news (Finnhub)
  4. Run full pipeline (collect -> score -> train -> evaluate -> backtest)
  5. Run experiment only (train + evaluate, assumes data already collected)
"""

from dotenv import load_dotenv

load_dotenv()

import os
import numpy as np
import pandas as pd

# Target tickers for the experiments
TARGET_TICKERS = ["AAPL", "JPM", "CVX", "UNH", "VOO"]
DEFAULT_START = "2023-01-01"


def sentiment_analysis_demo():
    """Live demo: fetch current news and score with FinBERT."""
    from src.sentiment.financial_news import fetch_news
    from src.sentiment.news_sentimental_analysis import SentimentScorer

    ticker = input("Enter ticker (default AAPL): ").strip().upper() or "AAPL"
    n = int(input("Number of articles (default 10): ").strip() or "10")

    print(f"\nFetching {n} articles for {ticker} from Google News ...")
    articles = fetch_news(query=ticker, num_articles=n)

    scorer = SentimentScorer()
    class_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for idx, article in enumerate(articles, 1):
        result = scorer.score(article["title"])
        label = result["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
        print(f"\n[{idx}] {article['title']}")
        print(f"     Published: {article['published']}")
        print(f"     Sentiment: {label} ({result['score']:.4f})")

    print(f"\n--- Summary ({len(articles)} articles) ---")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")


def download_stock_data():
    """Download OHLCV data via yfinance."""
    from src.technical.stock_data import TradingStock

    ticker = input("Enter ticker: ").upper()
    period = input("Enter period (e.g. 1y, 2y, 5y, max): ") or "2y"
    stock = TradingStock(ticker=ticker, period=period)
    stock.fetch().download_indicators()
    print("Done.")


def collect_historical_news():
    """Fetch historical news from Finnhub for all target tickers."""
    from src.sentiment.finnhub_news import fetch_news_for_tickers

    start = input(f"Start date (default {DEFAULT_START}): ").strip() or DEFAULT_START
    end = input("End date (default today): ").strip() or pd.Timestamp.now().strftime("%Y-%m-%d")

    tickers_input = input(
        f"Tickers (comma-separated, default {','.join(TARGET_TICKERS)}): "
    ).strip()
    tickers = (
        [t.strip().upper() for t in tickers_input.split(",")]
        if tickers_input
        else TARGET_TICKERS
    )

    fetch_news_for_tickers(tickers, start_date=start, end_date=end)
    print("\nNews collection complete.")


def run_full_pipeline():
    """
    End-to-end pipeline:
      1. Collect historical news from Finnhub
      2. Score headlines with FinBERT
      3. Aggregate daily sentiment features
      4. Train models and run experiments
      5. Backtest strategies
    """
    from src.sentiment.finnhub_news import fetch_historical_news
    from src.sentiment.news_sentimental_analysis import score_news_csv
    from src.sentiment.sentiment_features import build_daily_sentiment
    from src.ml.train import run_all_experiments
    from src.ml.features import build_feature_matrix, get_feature_columns
    from src.technical.backtest import run_backtest_comparison

    start = input(f"Start date (default {DEFAULT_START}): ").strip() or DEFAULT_START
    end = input("End date (default today): ").strip() or pd.Timestamp.now().strftime("%Y-%m-%d")

    tickers_input = input(
        f"Tickers (comma-separated, default {','.join(TARGET_TICKERS)}): "
    ).strip()
    tickers = (
        [t.strip().upper() for t in tickers_input.split(",")]
        if tickers_input
        else TARGET_TICKERS
    )

    # Step 1: Collect news
    print("\n" + "=" * 60)
    print("  STEP 1: Collecting historical news from Finnhub")
    print("=" * 60)
    for ticker in tickers:
        csv_path = f"dataset/{ticker}_news.csv"
        if os.path.exists(csv_path):
            print(f"  [{ticker}] News CSV already exists, skipping download.")
        else:
            fetch_historical_news(ticker, start, end)

    # Step 2: Score with FinBERT
    print("\n" + "=" * 60)
    print("  STEP 2: Scoring headlines with FinBERT")
    print("=" * 60)
    for ticker in tickers:
        news_csv = f"dataset/{ticker}_news.csv"
        sent_csv = f"dataset/{ticker}_sentiment.csv"
        if not os.path.exists(news_csv):
            print(f"  [{ticker}] No news CSV found, skipping.")
            continue
        if os.path.exists(sent_csv):
            print(f"  [{ticker}] Sentiment CSV already exists, skipping.")
            continue
        score_news_csv(news_csv)

    # Step 3: Aggregate daily features
    print("\n" + "=" * 60)
    print("  STEP 3: Aggregating daily sentiment features")
    print("=" * 60)
    for ticker in tickers:
        sent_csv = f"dataset/{ticker}_sentiment.csv"
        daily_csv = f"dataset/{ticker}_daily_sentiment.csv"
        if not os.path.exists(sent_csv):
            print(f"  [{ticker}] No sentiment CSV found, skipping.")
            continue
        if os.path.exists(daily_csv):
            print(f"  [{ticker}] Daily sentiment CSV already exists, skipping.")
            continue
        build_daily_sentiment(sent_csv)

    # Step 4: Train and evaluate
    print("\n" + "=" * 60)
    print("  STEP 4: Training models and running experiments")
    print("=" * 60)
    all_results = run_all_experiments(
        tickers, start=start, end=end, output_dir="results"
    )

    # Step 5: Backtest
    print("\n" + "=" * 60)
    print("  STEP 5: Backtesting strategies")
    print("=" * 60)
    _run_backtests(tickers, start, end, all_results)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  Results saved in results/ directory")
    print("=" * 60)


def run_experiment_only():
    """Train + evaluate assuming data is already collected and scored."""
    from src.ml.train import run_all_experiments

    start = input(f"Start date (default {DEFAULT_START}): ").strip() or DEFAULT_START
    end = input("End date (default today): ").strip() or pd.Timestamp.now().strftime("%Y-%m-%d")

    tickers_input = input(
        f"Tickers (comma-separated, default {','.join(TARGET_TICKERS)}): "
    ).strip()
    tickers = (
        [t.strip().upper() for t in tickers_input.split(",")]
        if tickers_input
        else TARGET_TICKERS
    )

    all_results = run_all_experiments(
        tickers, start=start, end=end, output_dir="results"
    )

    _run_backtests(tickers, start, end, all_results)


def _run_backtests(tickers, start, end, all_results):
    """Backtest each ticker using the hybrid XGBoost model predictions."""
    from src.ml.features import build_feature_matrix, get_feature_columns
    from src.ml.train import _xgb_model
    from src.technical.backtest import run_backtest_comparison

    for ticker in tickers:
        if ticker not in all_results or "error" in all_results.get(ticker, {}):
            continue

        print(f"\n--- Backtesting {ticker} ---")
        try:
            df = build_feature_matrix(
                ticker, start=start, end=end, include_sentiment=True
            )
            features = get_feature_columns(df, include_sentiment=True)
            split = int(len(df) * 0.8)
            train, test = df.iloc[:split], df.iloc[split:]

            model = _xgb_model(n_classes=int(train["label"].nunique()))
            model.fit(train[features], train["label"])
            preds = model.predict(test[features])

            run_backtest_comparison(
                ticker,
                df_full=test,
                predictions=preds,
                output_dir="results",
            )
        except Exception as e:
            print(f"  Backtest error for {ticker}: {e}")


def main():
    print("\n=== Hybrid Sentiment + Technical Trading System ===\n")
    print("1 = Sentiment analysis (live news demo)")
    print("2 = Download stock data")
    print("3 = Collect historical news (Finnhub)")
    print("4 = Run full pipeline (collect -> score -> train -> backtest)")
    print("5 = Run experiment only (assumes data collected)")
    print()

    action = int(input("Enter action: "))
    if action == 1:
        sentiment_analysis_demo()
    elif action == 2:
        download_stock_data()
    elif action == 3:
        collect_historical_news()
    elif action == 4:
        run_full_pipeline()
    elif action == 5:
        run_experiment_only()
    else:
        print("Invalid action.")


if __name__ == "__main__":
    main()
