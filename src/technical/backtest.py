from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
from backtesting.lib import crossover
from backtesting.test import SMA

from strategy import get_historical_data

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

class MeanReversion(Strategy):

    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        price = self.data.Close
        # Register RSI indicator
        self.rsi = self.I(talib.RSI, price, self.rsi_window)

    def next(self):
        price = self.data.Close[-1]
        # Buy when RSI < lower_bound
        if self.rsi[-1] < self.lower_bound:
            if not self.position.is_long:
                self.buy(tp=price*1.2, sl=price*0.9)
        # Sell when RSI > upper_bound
        elif self.rsi[-1] > self.upper_bound:
            if not self.position.is_short:
                self.sell(tp=price*0.8, sl=price*1.1)

df = get_historical_data('AAPL')
bt = Backtest(df, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
print(stats)
bt.plot()