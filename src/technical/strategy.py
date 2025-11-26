import pandas as pd
import numpy as np

def get_historical_data(ticker):
    """
    Parameters
    ----------
    ticker: stock ticker symbol
    """
    df = pd.read_csv(f'./dataset/{ticker}_historical_data.csv', parse_dates=['Date'])

    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols]
    
    # Convert Date to string if required (to match image)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
    
    df = df.round({
        'Open': 2, 'High': 2, 'Low': 2, 'Close': 2
    })

    return df


