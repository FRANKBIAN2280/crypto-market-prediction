import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt

# Optional: Add your Binance API key and secret if you need private endpoints
# For public price data, you can leave these blank
API_KEY = ""
API_SECRET = ""

def load_btc_data(start_str="1 Jan, 2020", interval="1d", limit=1000):
    """
    Fetch historical Bitcoin data from Binance.

    Parameters:
        start_str (str): Start date (e.g. '1 Jan, 2020')
        interval (str): Candlestick interval (e.g. '1d', '1h', '15m')
        limit (int): Number of data points to retrieve per request (max 1000)

    Returns:
        pd.DataFrame: Processed BTC/USDT historical data
    """
    client = Client(API_KEY, API_SECRET)

    # Fetch historical klines (candlestick data)
    klines = client.get_historical_klines("BTCUSDT", interval, start_str)

    # Convert to DataFrame
    cols = ["Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

    df = pd.DataFrame(klines, columns=cols)

    # Convert types
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    df["Close time"] = pd.to_datetime(df["Close time"], unit='ms')
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Set index and compute indicators
    df.set_index("Close time", inplace=True)
    df["Return"] = df["Close"].pct_change()
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()

    df.dropna(inplace=True)
    return df
