"""
CLI entry point for the Hybrid Sentiment + Technical Trading System.

Menu:
  1. Sentiment analysis (live news demo)
  2. Download stock data
  3. Collect historical news (Finnhub)
  4. Run full pipeline (collect -> score -> train -> evaluate -> backtest)
  5. Run experiment only (train + evaluate, assumes data already collected)
  6. Pretrain models (train and save to disk for live demo)
  7. Live trading demo (fetch today's data, predict signal)
"""

from dotenv import load_dotenv

load_dotenv()

import os
import numpy as np
import pandas as pd

# Target tickers for the experiments
TARGET_TICKERS = ["AAPL", "JPM", "CVX", "UNH", "VOO"]
PRETRAIN_TICKERS = ["VOO", "GOOGL", "JPM"]
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


def pretrain_models():
    """
    Pretrain hybrid XGBoost models for specific tickers and save to disk.

    This collects news (if needed), scores with FinBERT, aggregates
    sentiment, then trains and saves one model per ticker.
    """
    from src.sentiment.finnhub_news import fetch_historical_news
    from src.sentiment.news_sentimental_analysis import score_news_csv
    from src.sentiment.sentiment_features import build_daily_sentiment
    from src.ml.train import pretrain_multiple

    start = input(f"Start date (default {DEFAULT_START}): ").strip() or DEFAULT_START
    end = input("End date (default today): ").strip() or pd.Timestamp.now().strftime("%Y-%m-%d")

    tickers_input = input(
        f"Tickers (comma-separated, default {','.join(PRETRAIN_TICKERS)}): "
    ).strip()
    tickers = (
        [t.strip().upper() for t in tickers_input.split(",")]
        if tickers_input
        else PRETRAIN_TICKERS
    )

    # Ensure sentiment data exists for each ticker
    for ticker in tickers:
        news_csv = f"dataset/{ticker}_news.csv"
        sent_csv = f"dataset/{ticker}_sentiment.csv"
        daily_csv = f"dataset/{ticker}_daily_sentiment.csv"

        if not os.path.exists(news_csv):
            print(f"\n[{ticker}] Collecting news from Finnhub ...")
            fetch_historical_news(ticker, start, end)

        if os.path.exists(news_csv) and not os.path.exists(sent_csv):
            print(f"[{ticker}] Scoring headlines with FinBERT ...")
            score_news_csv(news_csv)

        if os.path.exists(sent_csv) and not os.path.exists(daily_csv):
            print(f"[{ticker}] Aggregating daily sentiment ...")
            build_daily_sentiment(sent_csv)

    # Train and save models
    print("\n" + "=" * 60)
    print("  TRAINING AND SAVING MODELS")
    print("=" * 60)
    pretrain_multiple(tickers, start=start, end=end, include_sentiment=True)

    print("\n" + "=" * 60)
    print("  PRE-TRAINING COMPLETE")
    print("  Models saved in models/ directory")
    print("=" * 60)


def live_demo():
    """
    Live trading signal demo.

    Loads a pretrained model, fetches today's news + price data,
    computes features, and predicts a trading signal.
    """
    from src.ml.train import load_pretrained
    from src.ml.features import (
        download_price_data,
        add_indicators,
        SENTIMENT_FEATURE_COLS,
    )
    from src.sentiment.news_sentimental_analysis import SentimentScorer

    # List available pretrained models
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No pretrained models found. Run option 6 first.")
        return

    available = [f.replace("_model.pkl", "") for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
    if not available:
        print("No pretrained models found. Run option 6 first.")
        return

    print(f"Available pretrained models: {', '.join(available)}")
    ticker = input("Enter ticker: ").strip().upper()

    if ticker not in available:
        print(f"No pretrained model for {ticker}. Available: {', '.join(available)}")
        return

    # Load the pretrained model
    bundle = load_pretrained(ticker, models_dir=models_dir)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle["include_sentiment"]
    meta = bundle["metadata"]

    print(f"\n  Model info: trained on {meta['n_samples']} days")
    print(f"  Holdout accuracy: {meta['holdout_accuracy']:.4f}, F1: {meta['holdout_f1_macro']:.4f}")

    # Step 1: Get today's technical indicators
    print(f"\n--- Fetching recent price data for {ticker} ---")
    df = download_price_data(ticker, start="2024-01-01")
    df_ta = add_indicators(df)
    df_ta.dropna(inplace=True)

    if df_ta.empty:
        print("Not enough price data to compute indicators.")
        return

    latest_date = df_ta.index[-1]
    print(f"  Latest trading day: {latest_date.strftime('%Y-%m-%d')}")

    # Step 2: Get today's sentiment (live from Google News)
    if include_sentiment:
        print(f"\n--- Fetching live news for {ticker} ---")
        try:
            from src.sentiment.financial_news import fetch_news
            articles = fetch_news(query=ticker, num_articles=10)

            if articles:
                scorer = SentimentScorer()
                scores = []
                for article in articles:
                    result = scorer.score(article["title"])
                    label = result["label"]
                    score = result["score"]
                    numeric = score if label == "positive" else (-score if label == "negative" else 0.0)
                    scores.append({"label": label, "numeric": numeric})
                    print(f"  [{label:>8}] {article['title'][:80]}")

                numerics = [s["numeric"] for s in scores]
                labels = [s["label"] for s in scores]
                n = len(scores)

                sent_values = {
                    "sent_mean": np.mean(numerics),
                    "sent_std": np.std(numerics) if n > 1 else 0.0,
                    "sent_count": n,
                    "sent_max": max(numerics),
                    "sent_min": min(numerics),
                    "sent_positive_ratio": sum(1 for l in labels if l == "positive") / n,
                    "sent_negative_ratio": sum(1 for l in labels if l == "negative") / n,
                    "sent_momentum_3d": np.mean(numerics),
                    "sent_momentum_5d": np.mean(numerics),
                    "sent_momentum_10d": np.mean(numerics),
                    "sent_vol_5d": 0.0,
                }

                for col, val in sent_values.items():
                    if col in feature_cols:
                        df_ta[col] = val
            else:
                print("  No articles found, using neutral sentiment.")
                for col in SENTIMENT_FEATURE_COLS:
                    if col in feature_cols:
                        df_ta[col] = 0.0
        except Exception as e:
            print(f"  Could not fetch live news: {e}")
            print("  Using neutral sentiment.")
            for col in SENTIMENT_FEATURE_COLS:
                if col in feature_cols:
                    df_ta[col] = 0.0

    # Step 3: Predict
    today_row = df_ta.iloc[[-1]]
    missing = [c for c in feature_cols if c not in today_row.columns]
    for c in missing:
        today_row[c] = 0.0

    X_today = today_row[feature_cols]
    prediction = model.predict(X_today)[0]
    probabilities = model.predict_proba(X_today)[0]

    signal_map = {0: "NEUTRAL (Hold)", 1: "BEARISH (Sell)", 2: "BULLISH (Buy)"}
    signal = signal_map.get(prediction, f"Unknown ({prediction})")
    proba_map = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
    neutral_proba = proba_map.get(0, 0.0)
    bearish_proba = proba_map.get(1, 0.0)
    bullish_proba = proba_map.get(2, 0.0)

    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL for {ticker}")
    print(f"  Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Close: ${today_row['Close'].values[0]:.2f}")
    print(f"{'='*60}")
    print(f"\n  >>> Signal: {signal}")
    print(f"\n  Class probabilities:")
    print(f"    Neutral : {neutral_proba:.4f}")
    print(f"    Bearish : {bearish_proba:.4f}")
    print(f"    Bullish : {bullish_proba:.4f}")
    print(f"    Confidence: {max(probabilities):.2%}")
    print()


def main():
    print("\n=== Hybrid Sentiment + Technical Trading System ===\n")
    print("1 = Sentiment analysis (live news demo)")
    print("2 = Download stock data")
    print("3 = Collect historical news (Finnhub)")
    print("4 = Run full pipeline (collect -> score -> train -> backtest)")
    print("5 = Run experiment only (assumes data collected)")
    print("6 = Pretrain models (VOO, GOOGL, JPM)")
    print("7 = Live trading demo (predict today's signal)")
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
    elif action == 6:
        pretrain_models()
    elif action == 7:
        live_demo()
    else:
        print("Invalid action.")


if __name__ == "__main__":
    main()
