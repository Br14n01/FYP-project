import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os


class TradingStock():

    def __init__(self, ticker: str = None, period: str = '1y', interval: str = '1d', windows: list = None):
        """
        Parameters
        ----------
        ticker: stock ticker symbol
        windows: list of integers for MA window sizes (default: [20, 40, 80, 120])
        """
        self.ticker = ticker
        self.period = period
        self.windows = windows if windows else [20, 40, 80, 120]
        self.data = None



    def fetch(self):
        """
        Load Stock data by Ticker.
        """
        self.data = yf.Ticker(self.ticker).history(period=self.period)
        self.data.drop("Dividends", axis=1, inplace=True)
        self.data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        return self.data


    def download_indicators(self):

        if self.data is not None and not self.data.empty:
            # Calculate and add indicators
            """
            self.data['ATR'] = self.data.ta.atr(length=20)
            self.data['RSI'] = self.data.ta.rsi()
            self.data['MidPrice'] = self.data.ta.midprice(length=1)
            macd = ta.macd(self.data['Close'])
            self.data = self.data.join(macd)

            get_slope = lambda col: linregress(np.arange(len(col)), np.array(col))[0]

            # Flexible MA and slope calculation
            for window in self.windows:
                ma_col = f"MA{window}"
                slope_col = f"slopeMA{window}"
                self.data[ma_col] = self.data.ta.sma(length=window)
                self.data[slope_col] = (
                    self.data[ma_col]
                    .rolling(window=window)
                    .apply(get_slope, raw=True)
                )
            """

            # Ensure directory exists
            os.makedirs("dataset", exist_ok=True)

            # Save to CSV file
            file_path = f"dataset/{self.ticker}_historical_data.csv"
            print(f"Saving data to {file_path}")
            self.data.to_csv(file_path, header=True, index=True)
