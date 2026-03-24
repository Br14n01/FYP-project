"""
Backtesting strategies: SMA crossover, buy-and-hold, and ML-signal-based.

Usage:
    from src.technical.backtest import run_backtest_comparison
    results = run_backtest_comparison("AAPL")
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

class SmaCross(Strategy):
    """Classic 10/20 SMA crossover."""

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


class BuyAndHold(Strategy):
    """Buy on the first bar and hold forever."""

    def init(self):
        pass

    def next(self):
        if not self.position:
            self.buy()


class MLSignalStrategy(Strategy):
    """
    Use pre-computed ML predictions as trading signals.

    Expects self.data to have a column 'Signal' with values:
      2 = bullish  -> buy
      1 = bearish  -> sell / go flat
      0 = neutral  -> do nothing
    """

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal)

    def next(self):
        if self.signal[-1] == 2:
            if not self.position.is_long:
                self.buy()
        elif self.signal[-1] == 1:
            if self.position.is_long:
                self.position.close()


# ---------------------------------------------------------------------------
# Helper: prepare data for backtesting.py (needs OHLCV with DatetimeIndex)
# ---------------------------------------------------------------------------

def _prepare_bt_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame is in the format backtesting.py expects."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.round({"Open": 2, "High": 2, "Low": 2, "Close": 2})
    return out


# ---------------------------------------------------------------------------
# Run backtest comparison
# ---------------------------------------------------------------------------

def run_backtest_comparison(
    symbol: str,
    df_full: pd.DataFrame | None = None,
    predictions: np.ndarray | None = None,
    output_dir: str = "results",
    cash: float = 100_000,
    commission: float = 0.002,
) -> dict:
    """
    Run three backtests and compare:
      1. Buy-and-hold
      2. SMA crossover
      3. ML hybrid signal (if predictions provided)

    Parameters
    ----------
    symbol : str
        Ticker symbol for labeling.
    df_full : pd.DataFrame
        DataFrame with at least OHLCV columns and a DatetimeIndex.
        Should correspond to the test period.
    predictions : np.ndarray or None
        Array of ML predictions (0/1/2) aligned with df_full rows.
    output_dir : str
        Where to save result plots and CSVs.

    Returns
    -------
    dict  : strategy_name -> key stats dict
    """
    os.makedirs(output_dir, exist_ok=True)
    bt_data = _prepare_bt_data(df_full)
    results = {}

    # 1. Buy-and-hold
    bt_bh = Backtest(bt_data, BuyAndHold, cash=cash, commission=commission)
    stats_bh = bt_bh.run()
    results["buy_and_hold"] = _extract_stats(stats_bh)

    # 2. SMA crossover
    bt_sma = Backtest(
        bt_data, SmaCross, cash=cash, commission=commission, exclusive_orders=True
    )
    stats_sma = bt_sma.run()
    results["sma_crossover"] = _extract_stats(stats_sma)

    # 3. ML signal
    if predictions is not None:
        bt_ml_data = bt_data.copy()
        bt_ml_data["Signal"] = predictions
        bt_ml = Backtest(
            bt_ml_data, MLSignalStrategy, cash=cash, commission=commission
        )
        stats_ml = bt_ml.run()
        results["ml_hybrid"] = _extract_stats(stats_ml)

    # Save comparison
    comp_df = pd.DataFrame(results).T
    comp_df.index.name = "strategy"
    path = os.path.join(output_dir, f"{symbol}_backtest_comparison.csv")
    comp_df.to_csv(path)
    print(f"  Backtest comparison -> {path}")

    # Print summary
    print(f"\n  {'Strategy':<20} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>10} {'#Trades':>8}")
    print(f"  {'-'*56}")
    for name, s in results.items():
        print(
            f"  {name:<20} {s['return_pct']:>10.2f} {s['sharpe']:>8.2f}"
            f" {s['max_drawdown_pct']:>10.2f} {s['n_trades']:>8}"
        )

    return results


def _extract_stats(stats) -> dict:
    return {
        "return_pct": stats["Return [%]"],
        "sharpe": stats.get("Sharpe Ratio", 0) or 0,
        "max_drawdown_pct": stats["Max. Drawdown [%]"],
        "n_trades": stats["# Trades"],
        "win_rate": stats.get("Win Rate [%]", 0) or 0,
        "avg_trade_pct": stats.get("Avg. Trade [%]", 0) or 0,
        "equity_final": stats["Equity Final [$]"],
    }
