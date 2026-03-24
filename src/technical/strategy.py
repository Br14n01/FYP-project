"""
Load historical data from CSV for backtesting.
"""

import os

import pandas as pd
import numpy as np


def get_historical_data(ticker: str, dataset_dir: str = "dataset") -> pd.DataFrame:
    """
    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    dataset_dir : str
        Directory containing {ticker}_historical_data.csv files.
    """
    path = os.path.join(dataset_dir, f"{ticker}_historical_data.csv")
    df = pd.read_csv(path, parse_dates=["Date"])

    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[cols]

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.strftime("%Y-%m-%d")
    df.set_index("Date", inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

    df = df.round({"Open": 2, "High": 2, "Low": 2, "Close": 2})
    return df
