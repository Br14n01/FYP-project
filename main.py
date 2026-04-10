"""
CLI entry point for the Hybrid Sentiment + Technical Trading System.

Menu:
  1. Sentiment demo (live news scoring)
  2. Collect & score news (Finnhub + FinBERT pipeline)
  3. Per-ticker pipeline (train / evaluate / pretrain)
  4. Train universal model (two-phase: technical + sentiment)
  5. Fine-tune & evaluate universal model
  6. Live signal (predict today's trading signal)
  7. Backtest simulation (day-by-day trading simulation)
  8. SHAP analysis (pretrained or universal model)
  0. Quit
"""

from dotenv import load_dotenv

load_dotenv()

import os
import numpy as np
import pandas as pd

TARGET_TICKERS = ["AAPL", "JPM", "CVX", "UNH", "VOO"]
PRETRAIN_TICKERS = ["VOO", "GOOGL", "JPM"]
DEFAULT_START = "2023-01-01"


# ===================================================================
# 1. Sentiment demo
# ===================================================================

def sentiment_demo():
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


# ===================================================================
# 2. Collect & score news
# ===================================================================

def collect_and_score_news():
    """
    Full sentiment data pipeline (no training):
      1. Fetch historical news from Finnhub
      2. Score headlines with FinBERT
      3. Aggregate into daily sentiment features
    """
    from src.sentiment.finnhub_news import fetch_historical_news
    from src.sentiment.news_sentimental_analysis import score_news_csv
    from src.sentiment.sentiment_features import build_daily_sentiment

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
            print(f"  [{ticker}] News CSV already exists, skipping.")
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

    print("\n" + "=" * 60)
    print("  NEWS COLLECTION & SCORING COMPLETE")
    print("  Sentiment CSVs saved in dataset/ directory")
    print("=" * 60)


# ===================================================================
# 3. Per-ticker pipeline
# ===================================================================

def per_ticker_pipeline():
    """
    Per-ticker training with sub-options:
      a) Full pipeline   – collect news -> score -> train -> backtest
      b) Train & evaluate – assumes sentiment data already exists
      c) Pretrain & save  – train models and save to disk for live demo
    """
    print("\n  Per-ticker pipeline sub-options:")
    print("    a = Full pipeline (collect news -> score -> train -> backtest)")
    print("    b = Train & evaluate only (assumes news already scored)")
    print("    c = Pretrain & save models (for live demo / simulation)")
    sub = input("  Choose (a/b/c, default a): ").strip().lower() or "a"

    if sub == "a":
        _run_full_pipeline()
    elif sub == "b":
        _run_experiment_only()
    elif sub == "c":
        _pretrain_models()
    else:
        print("  Invalid sub-option.")


def _run_full_pipeline():
    """End-to-end per-ticker pipeline."""
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

    # Step 1-3: News pipeline
    print("\n" + "=" * 60)
    print("  STEP 1-3: News collection, scoring, aggregation")
    print("=" * 60)
    for ticker in tickers:
        csv_path = f"dataset/{ticker}_news.csv"
        if os.path.exists(csv_path):
            print(f"  [{ticker}] News CSV exists, skipping download.")
        else:
            fetch_historical_news(ticker, start, end)

    from src.sentiment.news_sentimental_analysis import score_news_csv as _score
    for ticker in tickers:
        news_csv = f"dataset/{ticker}_news.csv"
        sent_csv = f"dataset/{ticker}_sentiment.csv"
        if not os.path.exists(news_csv):
            continue
        if not os.path.exists(sent_csv):
            _score(news_csv)

    from src.sentiment.sentiment_features import build_daily_sentiment as _daily
    for ticker in tickers:
        sent_csv = f"dataset/{ticker}_sentiment.csv"
        daily_csv = f"dataset/{ticker}_daily_sentiment.csv"
        if os.path.exists(sent_csv) and not os.path.exists(daily_csv):
            _daily(sent_csv)

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
    print("  PIPELINE COMPLETE – Results saved in results/")
    print("=" * 60)


def _run_experiment_only():
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


def _pretrain_models():
    """Pretrain hybrid XGBoost models for specific tickers and save to disk."""
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

    print("\n" + "=" * 60)
    print("  TRAINING AND SAVING MODELS")
    print("=" * 60)
    pretrain_multiple(tickers, start=start, end=end, include_sentiment=True)

    print("\n" + "=" * 60)
    print("  PRE-TRAINING COMPLETE – Models saved in models/")
    print("=" * 60)


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


# ===================================================================
# 4. Train universal model (two-phase)
# ===================================================================

def train_universal():
    """
    Two-phase universal model training:

    Phase 1 – Train a base model on multi-year technical data from many
    tickers.  Sentiment columns are present but zeroed out.

    Phase 2 – Fine-tune with sentiment: warm-start the base model on
    recent data (~1 year) where Finnhub news / FinBERT sentiment is
    available.
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


# ===================================================================
# 5. Fine-tune & evaluate universal model
# ===================================================================

def finetune_and_evaluate():
    """
    Fine-tune the universal model per sector, then run comprehensive
    evaluation (per-sector breakdown + held-out stock test).
    """
    from src.ml.universe import build_universal_dataset, get_all_tickers, temporal_train_test_split
    from src.ml.features import get_feature_columns
    from src.ml.train import load_universal_model, finetune_all_sectors
    from src.ml.evaluation import evaluate_universal_model, evaluate_held_out_stocks

    bundle = load_universal_model()
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle.get("include_sentiment", False)

    # --- Sector fine-tuning ---
    ft_input = input("Fine-tune per sector? (y/N): ").strip().lower()
    if ft_input in ("y", "yes"):
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
        df_ft = build_universal_dataset(
            tickers=tickers, start=start, include_sentiment=include_sentiment,
        )

        finetune_all_sectors(
            bundle, df_ft, feature_cols,
            n_extra_rounds=int(n_rounds),
        )

        print("\n" + "=" * 60)
        print("  SECTOR FINE-TUNING COMPLETE")
        print("=" * 60)

    # --- Evaluation ---
    eval_input = input("\nRun evaluation? (Y/n): ").strip().lower()
    if eval_input in ("n", "no"):
        return

    start = input("Eval data start (default 2019-01-01): ").strip() or "2019-01-01"

    held_out_input = input(
        "Held-out tickers for unseen-stock test (comma-separated, default TSLA,NFLX,DIS): "
    ).strip()
    held_out = (
        [t.strip().upper() for t in held_out_input.split(",")]
        if held_out_input
        else ["TSLA", "NFLX", "DIS"]
    )

    temporal_input = input(
        "Run full temporal test on all training tickers? (y/N): "
    ).strip().lower()
    run_temporal = temporal_input in ("y", "yes")

    if run_temporal:
        all_tickers = get_all_tickers() + held_out
        seen = set()
        unique_tickers = []
        for t in all_tickers:
            if t not in seen:
                unique_tickers.append(t)
                seen.add(t)
    else:
        unique_tickers = list(held_out)

    print(f"\n  Building dataset for evaluation ({len(unique_tickers)} tickers) ...")
    df_eval = build_universal_dataset(
        tickers=unique_tickers, start=start, include_sentiment=include_sentiment,
    )

    train_end = bundle["metadata"].get("train_end", "2023-12-31")

    if run_temporal:
        _, test_df = temporal_train_test_split(df_eval, train_end)
        if not test_df.empty:
            print(f"\n  Temporal test set: {len(test_df)} rows after {train_end}")
            evaluate_universal_model(bundle, test_df, feature_cols)

    if held_out:
        evaluate_held_out_stocks(bundle, df_eval, held_out, feature_cols)


# ===================================================================
# 6. Live signal
# ===================================================================

def live_signal():
    """
    Predict today's trading signal for a stock.

    Supports both per-ticker pretrained models and the universal
    cross-stock model.  When using the universal model you can predict
    a signal for any ticker.
    """
    from src.ml.train import load_pretrained, load_universal_model
    from src.ml.features import (
        download_price_data,
        add_indicators,
        add_relative_indicators,
        SENTIMENT_FEATURE_COLS,
    )
    from src.sentiment.news_sentimental_analysis import SentimentScorer

    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models found. Run option 3c or 4 first.")
        return

    per_ticker = [
        f.replace("_model.pkl", "")
        for f in os.listdir(models_dir)
        if f.endswith("_model.pkl") and not f.startswith(("universal", "sector_"))
    ]
    has_universal = os.path.exists(os.path.join(models_dir, "universal_model.pkl"))

    if not per_ticker and not has_universal:
        print("No models found. Run option 3c or 4 first.")
        return

    print("Available models:")
    if has_universal:
        print("  [universal] - Cross-stock universal model (works for any ticker)")
    for t in per_ticker:
        print(f"  [{t}] - Per-ticker model")

    model_choice = input(
        "\nModel to use (ticker name or 'universal', default universal): "
    ).strip()

    if not model_choice:
        use_universal = has_universal
    elif model_choice.lower() == "universal":
        if not has_universal:
            print("No universal model found. Run option 4 first.")
            return
        use_universal = True
    else:
        model_choice = model_choice.upper()
        if model_choice not in per_ticker:
            print(f"No pretrained model for {model_choice}.")
            return
        use_universal = False

    if use_universal:
        bundle = load_universal_model(models_dir=models_dir)
        ticker = input("Enter ticker to predict (e.g. AAPL): ").strip().upper()
    else:
        ticker = model_choice
        bundle = load_pretrained(ticker, models_dir=models_dir)

    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle.get("include_sentiment", False)
    meta = bundle["metadata"]

    if use_universal:
        print(f"\n  Universal model: {meta['n_features']} features, "
              f"trained on {meta['n_train']} samples")
    else:
        print(f"\n  Model info: trained on {meta['n_samples']} days")
        print(f"  Holdout accuracy: {meta['holdout_accuracy']:.4f}, "
              f"F1: {meta['holdout_f1_macro']:.4f}")

    # Step 1: Technical indicators
    print(f"\n--- Fetching recent price data for {ticker} ---")
    df = download_price_data(ticker, start="2024-01-01")
    df_ta = add_indicators(df)
    if use_universal:
        df_ta = add_relative_indicators(df_ta)
    df_ta.dropna(inplace=True)

    if df_ta.empty:
        print("Not enough price data to compute indicators.")
        return

    latest_date = df_ta.index[-1]
    print(f"  Latest trading day: {latest_date.strftime('%Y-%m-%d')}")

    # Step 2: Live sentiment
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

    model_label = "Universal" if use_universal else ticker
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL for {ticker} (model: {model_label})")
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


# ===================================================================
# 7. Backtest simulation
# ===================================================================

def backtest_simulation():
    """
    Day-by-day trading simulation.

    Supports both per-ticker models and the universal cross-stock model.
    """
    from src.ml.simulation import run_generalization_test

    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models found. Run option 3c or 4 first.")
        return

    per_ticker = [
        f.replace("_model.pkl", "")
        for f in os.listdir(models_dir)
        if f.endswith("_model.pkl") and not f.startswith(("universal", "sector_"))
    ]
    has_universal = os.path.exists(os.path.join(models_dir, "universal_model.pkl"))

    if not per_ticker and not has_universal:
        print("No models found. Run option 3c or 4 first.")
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
            print("No universal model found. Run option 4 first.")
            return
        use_universal = True
        model_ticker = "universal"
    else:
        model_ticker = model_choice.upper()
        if model_ticker not in per_ticker:
            print(f"No pretrained model for {model_ticker}.")
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

    short_input = input("Allow short selling? (y/N): ").strip().lower()
    allow_short = short_input in ("y", "yes")

    run_generalization_test(
        model_ticker=model_ticker,
        test_tickers=test_tickers,
        sim_start=sim_start,
        sim_end=sim_end,
        initial_capital=initial_capital,
        use_universal=use_universal,
        allow_short=allow_short,
    )


# ===================================================================
# 8. SHAP analysis
# ===================================================================

def shap_analysis():
    """Run SHAP feature-importance analysis for saved models."""
    from src.ml.train import (
        load_pretrained,
        load_universal_model,
        save_shap_from_model,
    )
    from src.ml.features import build_feature_matrix
    from src.ml.universe import (
        build_universal_dataset,
        get_all_tickers,
        temporal_train_test_split,
    )

    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models found. Run option 3c or 4 first.")
        return

    print("\n  SHAP analysis options:")
    print("    a = Pretrained per-ticker model")
    print("    b = Universal model")
    sub = input("  Choose (a/b, default b): ").strip().lower() or "b"

    sample_str = input("Rows to explain (default 500): ").strip() or "500"
    sample_size = int(sample_str)

    if sub == "a":
        per_ticker = [
            f.replace("_model.pkl", "")
            for f in os.listdir(models_dir)
            if f.endswith("_model.pkl") and not f.startswith(("universal", "sector_"))
        ]
        if not per_ticker:
            print("No pretrained per-ticker models found. Run option 3c first.")
            return

        print("Available pretrained models:")
        for t in per_ticker:
            print(f"  [{t}]")

        ticker = input("Ticker model to explain: ").strip().upper()
        if ticker not in per_ticker:
            print(f"No pretrained model for {ticker}.")
            return

        bundle = load_pretrained(ticker, models_dir=models_dir)
        include_sentiment = bundle.get("include_sentiment", False)

        start = input("Feature data start (default 2022-01-01): ").strip() or "2022-01-01"
        end = input("Feature data end (default latest): ").strip() or None

        print(f"\n  Building feature matrix for {ticker} ...")
        df = build_feature_matrix(
            ticker,
            start=start,
            end=end,
            include_sentiment=include_sentiment,
        )
        split = int(len(df) * 0.8)
        X_target = df.iloc[split:][bundle["feature_columns"]]

        if X_target.empty:
            print("Not enough rows in the test window for SHAP analysis.")
            return

        save_shap_from_model(
            bundle["model"],
            X_target,
            bundle["feature_columns"],
            label=f"{ticker}_pretrained",
            output_dir="results",
            sample_size=sample_size,
        )
        return

    if sub != "b":
        print("  Invalid sub-option.")
        return

    if not os.path.exists(os.path.join(models_dir, "universal_model.pkl")):
        print("No universal model found. Run option 4 first.")
        return

    bundle = load_universal_model(models_dir=models_dir)
    include_sentiment = bundle.get("include_sentiment", False)
    meta = bundle["metadata"]

    start = input("Dataset start (default 2019-01-01): ").strip() or "2019-01-01"
    tickers_input = input(
        "Tickers (comma-separated, or 'all' for full universe, default all): "
    ).strip()
    if tickers_input and tickers_input.lower() != "all":
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    else:
        tickers = get_all_tickers()

    sentiment_start = meta.get("sentiment_start")
    if include_sentiment:
        prompt = (
            f"Sentiment start (default {sentiment_start}): "
            if sentiment_start
            else "Sentiment start (optional, press Enter to skip): "
        )
        sentiment_input = input(prompt).strip()
        sentiment_start = sentiment_input or sentiment_start

    subset_input = input(
        "Subset to explain: train / val / test (default test): "
    ).strip().lower() or "test"

    print(f"\n  Building universal dataset from {len(tickers)} tickers ...")
    df = build_universal_dataset(
        tickers=tickers,
        start=start,
        include_sentiment=include_sentiment,
        sentiment_start=sentiment_start,
    )

    train_end = meta.get("train_end", "2023-12-31")
    val_end = meta.get("val_end")
    if val_end:
        train_df, val_df, test_df = temporal_train_test_split(df, train_end, val_end)
    else:
        train_df, test_df = temporal_train_test_split(df, train_end)
        val_df = pd.DataFrame(columns=df.columns)

    subset_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    target_df = subset_map.get(subset_input)
    if target_df is None:
        print("Invalid subset. Choose train, val, or test.")
        return
    if target_df.empty:
        print(f"No rows found in the '{subset_input}' subset.")
        return

    save_shap_from_model(
        bundle["model"],
        target_df[bundle["feature_columns"]],
        bundle["feature_columns"],
        label=f"universal_{subset_input}",
        output_dir="results",
        sample_size=sample_size,
    )


# ===================================================================
# Main menu
# ===================================================================

def main():
    while True:
        print("\n=== Hybrid Sentiment + Technical Trading System ===\n")
        print("  1 = Sentiment demo (live Google News scoring)")
        print("  2 = Collect & score news (Finnhub + FinBERT pipeline)")
        print("  3 = Per-ticker pipeline (train / evaluate / pretrain)")
        print("  4 = Train universal model (two-phase: technical + sentiment)")
        print("  5 = Fine-tune & evaluate universal model")
        print("  6 = Live signal (predict today's trading signal)")
        print("  7 = Backtest simulation (day-by-day trading simulation)")
        print("  8 = SHAP analysis (pretrained or universal model)")
        print("  0 = Quit")
        print()

        action = input("Enter action: ").strip().lower()
        if action in ("0", "q", "quit", "exit"):
            print("Goodbye.")
            break
        if action == "1":
            sentiment_demo()
        elif action == "2":
            collect_and_score_news()
        elif action == "3":
            per_ticker_pipeline()
        elif action == "4":
            train_universal()
        elif action == "5":
            finetune_and_evaluate()
        elif action == "6":
            live_signal()
        elif action == "7":
            backtest_simulation()
        elif action == "8":
            shap_analysis()
        else:
            print("Invalid action.")


if __name__ == "__main__":
    main()
