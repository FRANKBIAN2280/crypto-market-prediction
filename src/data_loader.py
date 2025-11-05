"""
data_loader.py

Fetches historical Bitcoin (BTC/USDT) price data from the Binance API.
Saves and returns a clean pandas DataFrame ready for analysis or modeling.
"""

import os
import pandas as pd
from binance.client import Client

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
API_KEY = os.getenv("BINANCE_API_KEY", "")      # Optional (public data doesn't need auth)
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = "BTCUSDT"                              # Trading pair to fetch
DEFAULT_INTERVAL = "1d"                         # Timeframe: 1m, 15m, 1h, 1d, etc.
DEFAULT_START = "1 Jan, 2020"                   # Binance format for start date


# ─────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────
def load_btc_data(start_str=DEFAULT_START, interval=DEFAULT_INTERVAL, save_path=None):
    """
    Fetch Bitcoin price data from Binance and return a cleaned DataFrame.

    Parameters
    ----------
    start_str : str
        Start date (e.g. "1 Jan, 2020")
    interval : str
        Kline interval, e.g. "1m", "1h", "1d"
    save_path : str or None
        Optional file path to save CSV (e.g. "data/raw/binance_btc.csv")

    Returns
    -------
    pd.DataFrame
        DataFrame containing Open, High, Low, Close, Volume, and timestamps.
    """
    print(f"Fetching BTC/USDT data from Binance ({interval} interval, since {start_str})...")

    client = Client(API_KEY, API_SECRET)

    # Fetch historical klines (candlestick data)
    klines = client.get_historical_klines(symbol=SYMBOL, interval=interval, start_str=start_str)

    # Define DataFrame columns according to Binance’s API response
    cols = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)

    # Convert timestamps and numeric columns
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Keep relevant columns only
    df = df[["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]]
    df.set_index("Close time", inplace=True)

    # Basic derived metric
    df["Return"] = df["Close"].pct_change()

    # Drop missing rows
    df.dropna(inplace=True)

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"✅ Saved to {save_path}")

    print(f"✅ Data loaded: {len(df)} rows from {df.index.min().date()} → {df.index.max().date()}")
    return df


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    btc = load_btc_data(start_str="1 Jan, 2021", interval="1d", save_path="data/raw/btcusdt_daily.csv")
    print(btc.tail())