"""
Day-by-day trading simulation using a pretrained hybrid model.

Simulates realistic trading: each day fetches news, computes indicators,
generates a signal, and executes a trade.

Usage:
    from src.ml.simulation import run_simulation
    results = run_simulation("VOO", test_ticker="JPM",
                             sim_start="2026-03-01", sim_end="2026-04-01")
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.features import download_price_data, add_indicators, SENTIMENT_FEATURE_COLS
from src.ml.train import load_pretrained
from src.sentiment.news_sentimental_analysis import SentimentScorer
from src.sentiment.finnhub_news import fetch_historical_news


def _compute_daily_sentiment(articles: list[dict]) -> dict:
    """Score a list of articles and return aggregated sentiment features."""
    if not articles:
        return {col: 0.0 for col in SENTIMENT_FEATURE_COLS}

    scorer = SentimentScorer()
    numerics = []
    labels = []

    for a in articles:
        result = scorer.score(a["headline"])
        label = result["label"]
        score = result["score"]
        numeric = score if label == "positive" else (-score if label == "negative" else 0.0)
        numerics.append(numeric)
        labels.append(label)

    n = len(numerics)
    return {
        "sent_mean": np.mean(numerics),
        "sent_std": np.std(numerics) if n > 1 else 0.0,
        "sent_count": float(n),
        "sent_max": max(numerics),
        "sent_min": min(numerics),
        "sent_positive_ratio": sum(1 for l in labels if l == "positive") / n,
        "sent_negative_ratio": sum(1 for l in labels if l == "negative") / n,
        "sent_momentum_3d": np.mean(numerics),
        "sent_momentum_5d": np.mean(numerics),
        "sent_momentum_10d": np.mean(numerics),
        "sent_vol_5d": 0.0,
    }


def _get_news_by_date(news_df: pd.DataFrame, date: str, max_articles: int = 5) -> list[dict]:
    """Extract articles for a specific date from the pre-fetched news DataFrame."""
    if news_df.empty:
        return []
    day_news = news_df[news_df["date"] == date]
    rows = day_news.head(max_articles)
    return [{"headline": row["headline"]} for _, row in rows.iterrows()]


def run_simulation(
    model_ticker: str,
    test_ticker: str,
    sim_start: str = "2026-03-01",
    sim_end: str = "2026-04-01",
    initial_capital: float = 10_000.0,
    articles_per_day: int = 5,
    output_dir: str = "results",
) -> dict:
    """
    Run a day-by-day trading simulation.

    Parameters
    ----------
    model_ticker : str
        Ticker whose pretrained model to load (e.g. "VOO").
    test_ticker : str
        Ticker to trade during the simulation (e.g. "JPM").
    sim_start : str
        Simulation start date (ISO format).
    sim_end : str
        Simulation end date (ISO format). All positions closed on this date.
    initial_capital : float
        Starting cash.
    articles_per_day : int
        Number of news articles to fetch per day.
    output_dir : str
        Where to save results.

    Returns
    -------
    dict with keys: daily_log, metrics, trades
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SIMULATION: model={model_ticker}, trading={test_ticker}")
    print(f"  Period: {sim_start} -> {sim_end}")
    print(f"  Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}")

    # Load pretrained model
    bundle = load_pretrained(model_ticker)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    include_sentiment = bundle["include_sentiment"]

    # Download price data with enough lookback for indicators
    lookback_start = (datetime.strptime(sim_start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    sim_end_dt = datetime.strptime(sim_end, "%Y-%m-%d")
    fetch_end = (sim_end_dt + timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"\n  Downloading price data for {test_ticker} ...")
    price_df = download_price_data(test_ticker, start=lookback_start, end=fetch_end)
    if price_df.empty:
        print("  ERROR: No price data available.")
        return {"error": "No price data"}

    # Pre-fetch all news for the simulation period
    print(f"  Pre-fetching news for {test_ticker} ({sim_start} -> {sim_end}) ...")
    news_df = fetch_historical_news(test_ticker, sim_start, sim_end, save=False, rate_limit_pause=0.5)

    # Compute indicators on full price history
    print("  Computing technical indicators ...")
    df_ta = add_indicators(price_df)
    df_ta.dropna(inplace=True)
    df_ta.index = pd.to_datetime(df_ta.index)
    if df_ta.index.tz is not None:
        df_ta.index = df_ta.index.tz_localize(None)

    # Determine trading days within the simulation window
    sim_start_dt = pd.Timestamp(sim_start)
    sim_end_dt = pd.Timestamp(sim_end)
    trading_days = df_ta.loc[sim_start_dt:sim_end_dt].index.tolist()

    if not trading_days:
        print("  ERROR: No trading days found in the simulation window.")
        return {"error": "No trading days"}

    print(f"  Found {len(trading_days)} trading days\n")

    # Simulation state
    cash = initial_capital
    shares = 0
    daily_log = []
    trades = []
    sentiment_history = []

    signal_map = {0: "HOLD", 1: "SELL", 2: "BUY"}

    for i, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        close_price = float(df_ta.loc[day, "Close"])
        is_last_day = (i == len(trading_days) - 1) or (day >= sim_end_dt)

        # Fetch sentiment for the day
        articles = _get_news_by_date(news_df, day_str, max_articles=articles_per_day)
        sent_features = _compute_daily_sentiment(articles)
        sentiment_history.append(sent_features)

        # Rolling sentiment momentum from history
        if len(sentiment_history) >= 3:
            sent_features["sent_momentum_3d"] = np.mean([s["sent_mean"] for s in sentiment_history[-3:]])
        if len(sentiment_history) >= 5:
            sent_features["sent_momentum_5d"] = np.mean([s["sent_mean"] for s in sentiment_history[-5:]])
        if len(sentiment_history) >= 10:
            sent_features["sent_momentum_10d"] = np.mean([s["sent_mean"] for s in sentiment_history[-10:]])
        if len(sentiment_history) >= 5:
            recent_means = [s["sent_mean"] for s in sentiment_history[-5:]]
            sent_features["sent_vol_5d"] = float(np.std(recent_means))

        # Build feature vector for this day
        today_row = df_ta.loc[[day]].copy()
        if include_sentiment:
            for col in SENTIMENT_FEATURE_COLS:
                if col in feature_cols:
                    today_row[col] = sent_features.get(col, 0.0)

        # Fill any missing feature columns with 0
        for col in feature_cols:
            if col not in today_row.columns:
                today_row[col] = 0.0

        X_today = today_row[feature_cols]
        prediction = int(model.predict(X_today)[0])
        probabilities = model.predict_proba(X_today)[0]
        proba_map = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
        confidence = float(max(probabilities))

        action_taken = "HOLD"
        trade_shares = 0

        # Force close on last day
        if is_last_day and shares > 0:
            cash += shares * close_price
            trade_shares = -shares
            action_taken = "CLOSE_ALL"
            trades.append({
                "date": day_str, "action": "SELL (close)", "shares": abs(trade_shares),
                "price": close_price, "value": abs(trade_shares) * close_price,
            })
            shares = 0
        elif prediction == 2 and shares == 0:
            # BUY: invest all cash
            trade_shares = int(cash // close_price)
            if trade_shares > 0:
                cost = trade_shares * close_price
                cash -= cost
                shares += trade_shares
                action_taken = "BUY"
                trades.append({
                    "date": day_str, "action": "BUY", "shares": trade_shares,
                    "price": close_price, "value": cost,
                })
        elif prediction == 1 and shares > 0:
            # SELL: close position
            revenue = shares * close_price
            cash += revenue
            trade_shares = -shares
            action_taken = "SELL"
            trades.append({
                "date": day_str, "action": "SELL", "shares": shares,
                "price": close_price, "value": revenue,
            })
            shares = 0

        portfolio_value = cash + shares * close_price

        daily_log.append({
            "date": day_str,
            "close": close_price,
            "signal": signal_map.get(prediction, str(prediction)),
            "action": action_taken,
            "confidence": confidence,
            "shares": shares,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "news_count": len(articles),
            "sent_mean": sent_features["sent_mean"],
        })

        arrow = {"BUY": ">>>", "SELL": "<<<", "CLOSE_ALL": "XXX", "HOLD": "   "}
        print(
            f"  {day_str}  ${close_price:>8.2f}  "
            f"signal={signal_map.get(prediction, '?'):<4}  "
            f"{arrow.get(action_taken, '   ')} {action_taken:<10}  "
            f"portfolio=${portfolio_value:>10,.2f}  "
            f"news={len(articles)}  sent={sent_features['sent_mean']:+.3f}"
        )

    # Compute metrics
    log_df = pd.DataFrame(daily_log)
    metrics = _compute_metrics(log_df, initial_capital, trades)

    # Print summary
    _print_summary(model_ticker, test_ticker, sim_start, sim_end, metrics, trades)

    # Save outputs
    _save_results(log_df, metrics, trades, model_ticker, test_ticker, output_dir)
    _plot_capital(log_df, model_ticker, test_ticker, initial_capital, metrics, output_dir)

    return {"daily_log": log_df, "metrics": metrics, "trades": trades}


def _compute_metrics(log_df: pd.DataFrame, initial_capital: float, trades: list) -> dict:
    """Compute trading evaluation metrics."""
    final_value = log_df["portfolio_value"].iloc[-1]
    overall_return = (final_value - initial_capital) / initial_capital

    # Daily returns
    portfolio_values = log_df["portfolio_value"].values
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Maximum drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1 and np.std(downside_returns) > 0:
        sortino = float(np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252))
    else:
        sortino = 0.0

    # Win rate from trades
    trade_returns = []
    buy_price = None
    for t in trades:
        if t["action"] == "BUY":
            buy_price = t["price"]
        elif t["action"] in ("SELL", "SELL (close)") and buy_price is not None:
            trade_returns.append((t["price"] - buy_price) / buy_price)
            buy_price = None

    winning_trades = sum(1 for r in trade_returns if r > 0)
    total_round_trips = len(trade_returns)
    win_rate = winning_trades / total_round_trips if total_round_trips > 0 else 0.0

    # Volatility (annualized)
    volatility = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0

    # Buy-and-hold comparison
    start_price = log_df["close"].iloc[0]
    end_price = log_df["close"].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "overall_return_pct": overall_return * 100,
        "buy_hold_return_pct": buy_hold_return * 100,
        "excess_return_pct": (overall_return - buy_hold_return) * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "volatility_annual_pct": volatility * 100,
        "total_trades": len(trades),
        "round_trips": total_round_trips,
        "win_rate_pct": win_rate * 100,
        "avg_trade_return_pct": float(np.mean(trade_returns) * 100) if trade_returns else 0.0,
        "trading_days": len(log_df),
    }


def _print_summary(
    model_ticker: str,
    test_ticker: str,
    sim_start: str,
    sim_end: str,
    metrics: dict,
    trades: list,
):
    """Print a formatted summary of the simulation."""
    print(f"\n{'='*60}")
    print(f"  SIMULATION RESULTS")
    print(f"  Model: {model_ticker} | Trading: {test_ticker}")
    print(f"  Period: {sim_start} -> {sim_end}")
    print(f"{'='*60}")
    print(f"  Initial Capital:     ${metrics['initial_capital']:>12,.2f}")
    print(f"  Final Value:         ${metrics['final_value']:>12,.2f}")
    print(f"  Overall Return:      {metrics['overall_return_pct']:>+11.2f}%")
    print(f"  Buy & Hold Return:   {metrics['buy_hold_return_pct']:>+11.2f}%")
    print(f"  Excess Return:       {metrics['excess_return_pct']:>+11.2f}%")
    print(f"  {'─'*40}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>11.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>11.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>11.2f}")
    print(f"  Annualized Vol:      {metrics['volatility_annual_pct']:>11.2f}%")
    print(f"  {'─'*40}")
    print(f"  Total Trades:        {metrics['total_trades']:>11}")
    print(f"  Round Trips:         {metrics['round_trips']:>11}")
    print(f"  Win Rate:            {metrics['win_rate_pct']:>11.1f}%")
    print(f"  Avg Trade Return:    {metrics['avg_trade_return_pct']:>+11.2f}%")
    print(f"  Trading Days:        {metrics['trading_days']:>11}")
    print(f"{'='*60}\n")


def _save_results(
    log_df: pd.DataFrame,
    metrics: dict,
    trades: list,
    model_ticker: str,
    test_ticker: str,
    output_dir: str,
):
    """Save simulation CSV outputs."""
    prefix = f"sim_{model_ticker}_on_{test_ticker}"

    log_df.to_csv(os.path.join(output_dir, f"{prefix}_daily_log.csv"), index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False)

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(os.path.join(output_dir, f"{prefix}_trades.csv"), index=False)

    print(f"  Results saved -> {output_dir}/{prefix}_*.csv")


def _plot_capital(
    log_df: pd.DataFrame,
    model_ticker: str,
    test_ticker: str,
    initial_capital: float,
    metrics: dict,
    output_dir: str,
):
    """Plot portfolio value over time with buy/sell markers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)

    dates = pd.to_datetime(log_df["date"])
    portfolio = log_df["portfolio_value"]

    # Buy-and-hold line for comparison
    start_price = log_df["close"].iloc[0]
    buy_hold = initial_capital * (log_df["close"] / start_price)

    # Portfolio value
    ax1.plot(dates, portfolio, "b-", linewidth=2, label="ML Hybrid Strategy")
    ax1.plot(dates, buy_hold, "k--", linewidth=1, alpha=0.6, label="Buy & Hold")
    ax1.axhline(y=initial_capital, color="gray", linestyle=":", alpha=0.5)

    # Mark buy/sell actions
    buys = log_df[log_df["action"] == "BUY"]
    sells = log_df[log_df["action"].isin(["SELL", "CLOSE_ALL"])]
    if not buys.empty:
        ax1.scatter(pd.to_datetime(buys["date"]), buys["portfolio_value"],
                    marker="^", color="green", s=100, zorder=5, label="Buy")
    if not sells.empty:
        ax1.scatter(pd.to_datetime(sells["date"]), sells["portfolio_value"],
                    marker="v", color="red", s=100, zorder=5, label="Sell")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(
        f"Simulation: {model_ticker} model on {test_ticker}  |  "
        f"Return: {metrics['overall_return_pct']:+.2f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown_pct']:.2f}%"
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Sentiment subplot
    ax2.bar(dates, log_df["sent_mean"], color=["green" if s > 0 else "red" for s in log_df["sent_mean"]],
            alpha=0.6, width=0.8)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_ylabel("Sentiment")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"sim_{model_ticker}_on_{test_ticker}_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {path}")


def run_generalization_test(
    model_ticker: str = "VOO",
    test_tickers: list[str] | None = None,
    sim_start: str = "2026-03-01",
    sim_end: str = "2026-04-01",
    initial_capital: float = 10_000.0,
    output_dir: str = "results",
) -> dict[str, dict]:
    """
    Test whether a model trained on one ticker generalizes to others.

    Runs the simulation on each test ticker using the same pretrained model,
    then produces a comparison summary.
    """
    if test_tickers is None:
        test_tickers = ["VOO", "GOOG", "JPM"]

    all_results = {}

    for ticker in test_tickers:
        print(f"\n{'#'*60}")
        print(f"  GENERALIZATION TEST: {model_ticker} model -> {ticker}")
        print(f"{'#'*60}")
        try:
            result = run_simulation(
                model_ticker=model_ticker,
                test_ticker=ticker,
                sim_start=sim_start,
                sim_end=sim_end,
                initial_capital=initial_capital,
                output_dir=output_dir,
            )
            all_results[ticker] = result.get("metrics", {})
        except Exception as e:
            print(f"  ERROR simulating {ticker}: {e}")
            all_results[ticker] = {"error": str(e)}

    # Comparison table
    _save_generalization_summary(model_ticker, all_results, output_dir)
    return all_results


def _save_generalization_summary(model_ticker: str, all_results: dict, output_dir: str):
    """Save and print a comparison table across all test tickers."""
    rows = []
    for ticker, metrics in all_results.items():
        if "error" in metrics:
            continue
        rows.append({"test_ticker": ticker, **metrics})

    if not rows:
        return

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"sim_{model_ticker}_generalization.csv")
    df.to_csv(path, index=False)

    print(f"\n{'='*70}")
    print(f"  GENERALIZATION SUMMARY (model trained on {model_ticker})")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'Return%':>10} {'B&H%':>10} {'Excess%':>10} {'MaxDD%':>10} {'Sharpe':>8} {'WinRate':>8}")
    print(f"  {'─'*64}")
    for _, r in df.iterrows():
        print(
            f"  {r['test_ticker']:<8} "
            f"{r['overall_return_pct']:>+10.2f} "
            f"{r['buy_hold_return_pct']:>+10.2f} "
            f"{r['excess_return_pct']:>+10.2f} "
            f"{r['max_drawdown_pct']:>10.2f} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['win_rate_pct']:>7.1f}%"
        )
    print(f"{'='*70}")
    print(f"  Summary saved -> {path}\n")
