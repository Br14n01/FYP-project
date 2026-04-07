"""
CLI entry point for the Hybrid Sentiment + Technical Trading System.

Menu:
  1.  Sentiment analysis (live news demo)
  2.  Download stock data
  3.  Collect historical news (Finnhub)
  4.  Run full pipeline (collect -> score -> train -> evaluate -> backtest)
  5.  Run experiment only (train + evaluate, assumes data already collected)
  6.  Pretrain models (train and save to disk for live demo)
  7.  Live trading demo (predict today's signal)
  8.  Backtest simulation (day-by-day trading simulation with evaluation)
  9.  Train universal (cross-stock) model
  10. Fine-tune universal model per sector
  11. Evaluate universal model (held-out stocks + per-sector)
  12. Hyperparameter tuning (Optuna)
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


def backtest_simulation():
    """
    Day-by-day trading simulation using a pretrained model.

    Supports both per-ticker models (trained via option 6) and the
    universal cross-stock model (trained via option 9).
    """
    from src.ml.simulation import run_generalization_test

    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No pretrained models found. Run option 6 or 9 first.")
        return

    # Check what's available
    per_ticker = [
        f.replace("_model.pkl", "")
        for f in os.listdir(models_dir)
        if f.endswith("_model.pkl") and not f.startswith(("universal", "sector_"))
    ]
    has_universal = os.path.exists(os.path.join(models_dir, "universal_model.pkl"))

    if not per_ticker and not has_universal:
        print("No pretrained models found. Run option 6 or 9 first.")
        return

    print("Available models:")
    if has_universal:
        print("  [universal] - Cross-stock universal model")
    for t in per_ticker:
        print(f"  [{t}] - Per-ticker model")

    model_choice = input(
        "\nModel to use (ticker name or 'universal', default universal): "
    ).strip()

    if not model_choice:
        use_universal = has_universal
        model_ticker = "VOO"
    elif model_choice.lower() == "universal":
        if not has_universal:
            print("No universal model found. Run option 9 first.")
            return
        use_universal = True
        model_ticker = "universal"
    else:
        model_ticker = model_choice.upper()
        if model_ticker not in per_ticker:
            print(f"No pretrained model for {model_ticker}. Available: {', '.join(per_ticker)}")
            return
        use_universal = False

    test_input = input("Test tickers (comma-separated, default VOO,GOOG,JPM): ").strip()
    test_tickers = (
        [t.strip().upper() for t in test_input.split(",")]
        if test_input
        else ["VOO", "GOOG", "JPM"]
    )

    sim_start = input("Simulation start date (default 2026-03-01): ").strip() or "2026-03-01"
    sim_end = input("Simulation end date (default 2026-04-01): ").strip() or "2026-04-01"

    capital_str = input("Initial capital in USD (default 10000): ").strip() or "10000"
    initial_capital = float(capital_str)

    run_generalization_test(
        model_ticker=model_ticker,
        test_tickers=test_tickers,
        sim_start=sim_start,
        sim_end=sim_end,
        initial_capital=initial_capital,
        use_universal=use_universal,
    )


def train_universal():
    """
    Two-phase universal model training:

    Phase 1 – Train a base model on multi-year technical data from many
    tickers.  Sentiment columns are present but zeroed out.

    Phase 2 – Fine-tune with sentiment: warm-start the base model on
    recent data (~1 year) where Finnhub news / FinBERT sentiment is
    available.  The model learns to incorporate news signals on top of
    the technical patterns learned in Phase 1.
    """
    from src.ml.universe import build_universal_dataset, get_all_tickers
    from src.ml.features import get_feature_columns
    from src.ml.train import train_universal_model, finetune_with_sentiment

    start = input("Training data start (default 2019-01-01): ").strip() or "2019-01-01"

    tickers_input = input(
        "Tickers (comma-separated, or 'all' for full universe, default all): "
    ).strip()
    if tickers_input and tickers_input.lower() != "all":
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    else:
        tickers = get_all_tickers()

    train_end = input("Train cutoff date (default 2023-12-31): ").strip() or "2023-12-31"
    val_end = input("Validation cutoff date (default 2024-06-30): ").strip() or "2024-06-30"

    sentiment_start = input(
        "Sentiment data available from (default 2025-04-01): "
    ).strip() or "2025-04-01"

    tune_input = input("Run Optuna hyperparameter tuning? (y/N): ").strip().lower()
    do_tune = tune_input in ("y", "yes")

    adaptive_input = input("Use adaptive labels? (y/N): ").strip().lower()
    adaptive = adaptive_input in ("y", "yes")

    # Build dataset with sentiment columns included but zeroed before
    # the sentiment_start date (two-phase approach).
    print(f"\n  Building universal dataset from {len(tickers)} tickers ...")
    print(f"  Sentiment columns included (zeroed before {sentiment_start})")
    df = build_universal_dataset(
        tickers=tickers,
        start=start,
        include_sentiment=True,
        sentiment_start=sentiment_start,
        adaptive_label=adaptive,
    )

    feature_cols = get_feature_columns(df, include_sentiment=True)

    # Phase 1: train base model on full date range (sentiment = zeros
    # for the vast majority of rows, so the model learns technicals).
    print("\n" + "=" * 60)
    print("  PHASE 1: Technical base model (multi-year data)")
    print("=" * 60)
    bundle = train_universal_model(
        df,
        feature_cols,
        train_end=train_end,
        val_end=val_end,
        tune=do_tune,
    )

    # Phase 2: warm-start fine-tune on recent data where sentiment is
    # available, adding trees that can leverage news signals.
    print("\n" + "=" * 60)
    print("  PHASE 2: Sentiment fine-tuning (recent data)")
    print("=" * 60)
    bundle = finetune_with_sentiment(
        bundle, df, feature_cols,
        sentiment_start=sentiment_start,
    )

    print("\n" + "=" * 60)
    print("  TWO-PHASE UNIVERSAL MODEL TRAINING COMPLETE")
    print("  Model saved in models/ directory")
    print("=" * 60)


def finetune_sectors():
    """Fine-tune the universal model for each sector."""
    from src.ml.universe import build_universal_dataset, SECTORS, SECTOR_ID, get_all_tickers
    from src.ml.features import get_feature_columns
    from src.ml.train import load_universal_model, finetune_all_sectors

    bundle = load_universal_model()
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle.get("include_sentiment", False)

    start = input("Training data start (default 2019-01-01): ").strip() or "2019-01-01"
    n_rounds = input("Extra boosting rounds per sector (default 50): ").strip() or "50"

    tickers_input = input(
        "Tickers (comma-separated, or 'all', default all): "
    ).strip()
    if tickers_input and tickers_input.lower() != "all":
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    else:
        tickers = get_all_tickers()

    print(f"\n  Building dataset for fine-tuning ...")
    df = build_universal_dataset(
        tickers=tickers, start=start, include_sentiment=include_sentiment,
    )

    finetune_all_sectors(
        bundle, df, feature_cols,
        n_extra_rounds=int(n_rounds),
    )

    print("\n" + "=" * 60)
    print("  SECTOR FINE-TUNING COMPLETE")
    print("  Models saved in models/ directory")
    print("=" * 60)


def evaluate_universal():
    """Evaluate the universal model with held-out stocks and per-sector breakdown."""
    from src.ml.universe import build_universal_dataset, SECTORS, get_all_tickers
    from src.ml.features import get_feature_columns
    from src.ml.train import load_universal_model
    from src.ml.evaluation import evaluate_universal_model, evaluate_held_out_stocks

    bundle = load_universal_model()
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle.get("include_sentiment", False)

    start = input("Data start (default 2019-01-01): ").strip() or "2019-01-01"

    held_out_input = input(
        "Held-out tickers for unseen-stock test (comma-separated, default TSLA,NFLX,DIS): "
    ).strip()
    held_out = (
        [t.strip().upper() for t in held_out_input.split(",")]
        if held_out_input
        else ["TSLA", "NFLX", "DIS"]
    )

    all_tickers = get_all_tickers() + held_out
    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            unique_tickers.append(t)
            seen.add(t)

    print(f"\n  Building dataset for evaluation ...")
    df = build_universal_dataset(
        tickers=unique_tickers, start=start, include_sentiment=include_sentiment,
    )

    # Full test set evaluation (temporal, after training period)
    train_end = bundle["metadata"].get("train_end", "2023-12-31")
    from src.ml.universe import temporal_train_test_split
    _, test_df = temporal_train_test_split(df, train_end)

    if not test_df.empty:
        print(f"\n  Temporal test set: {len(test_df)} rows after {train_end}")
        evaluate_universal_model(bundle, test_df, feature_cols)

    # Held-out stock evaluation
    if held_out:
        evaluate_held_out_stocks(bundle, df, held_out, feature_cols)


def run_hyperparam_tuning():
    """Run Optuna hyperparameter tuning on a single ticker or pooled data."""
    from src.ml.features import build_feature_matrix, get_feature_columns
    from src.ml.train import tune_hyperparameters

    ticker = input("Ticker to tune on (default AAPL): ").strip().upper() or "AAPL"
    start = input(f"Start date (default {DEFAULT_START}): ").strip() or DEFAULT_START
    n_trials = input("Number of Optuna trials (default 50): ").strip() or "50"

    print(f"\n  Building feature matrix for {ticker} ...")
    df = build_feature_matrix(
        ticker, start=start, include_sentiment=True,
        use_relative_features=True,
    )
    feature_cols = get_feature_columns(df, include_sentiment=True)

    print(f"  {len(df)} rows, {len(feature_cols)} features")
    best_params = tune_hyperparameters(df, feature_cols, n_trials=int(n_trials))

    print(f"\n  Best parameters for {ticker}:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")


def main():
    print("\n=== Hybrid Sentiment + Technical Trading System ===\n")
    print(" 1 = Sentiment analysis (live news demo)")
    print(" 2 = Download stock data")
    print(" 3 = Collect historical news (Finnhub)")
    print(" 4 = Run full pipeline (collect -> score -> train -> backtest)")
    print(" 5 = Run experiment only (assumes data collected)")
    print(" 6 = Pretrain models (VOO, GOOGL, JPM)")
    print(" 7 = Live trading demo (predict today's signal)")
    print(" 8 = Backtest simulation (generalization test)")
    print(" 9 = Train universal (cross-stock) model")
    print("10 = Fine-tune universal model per sector")
    print("11 = Evaluate universal model (held-out + per-sector)")
    print("12 = Hyperparameter tuning (Optuna)")
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
    elif action == 8:
        backtest_simulation()
    elif action == 9:
        train_universal()
    elif action == 10:
        finetune_sectors()
    elif action == 11:
        evaluate_universal()
    elif action == 12:
        run_hyperparam_tuning()
    else:
        print("Invalid action.")


if __name__ == "__main__":
    main()
