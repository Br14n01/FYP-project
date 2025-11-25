import pandas as pd

def sma(df, period=14, price_col='Close'):
    """
    Calculate Simple Moving Average (SMA)
    """
    return df[price_col].rolling(window=period).mean()

def ema(df, period=14, price_col='Close'):
    """
    Calculate Exponential Moving Average (EMA)
    """
    return df[price_col].ewm(span=period, adjust=False).mean()

def rsi(df, period=14, price_col='Close'):
    """
    Calculate Relative Strength Index (RSI)
    """
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(df, fast_period=12, slow_period=26, signal_period=9, price_col='Close'):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    Returns MACD line, signal line, and MACD histogram
    """
    ema_fast = ema(df, period=fast_period, price_col=price_col)
    ema_slow = ema(df, period=slow_period, price_col=price_col)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist

def vwap(df, price_col_high='High', price_col_low='Low', price_col_close='Close', volume_col='Volume'):
    """
    Calculate Volume Weighted Average Price (VWAP)
    """
    typical_price = (df[price_col_high] + df[price_col_low] + df[price_col_close]) / 3
    cumulative_vp = (typical_price * df[volume_col]).cumsum()
    cumulative_vol = df[volume_col].cumsum()
    vwap = cumulative_vp / cumulative_vol
    return vwap

# Example usage:
# df = pd.read_csv('your_data.csv')
# df['SMA'] = sma(df, period=20)
# df['EMA'] = ema(df, period=20)
# df['RSI'] = rsi(df, period=14)
# df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df)
# df['VWAP'] = vwap(df)
