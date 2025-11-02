import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical Bitcoin data (daily)
btc = yf.download("BTC-USD", start="2020-01-01", end="2025-01-01", interval="1d")

# Display basic info
print(btc.head())
print(btc.describe())

# Drop missing values
btc = btc.dropna()

# Compute daily returns
btc["Return"] = btc["Close"].pct_change()

# Compute simple moving average
btc["SMA_7"] = btc["Close"].rolling(window=7).mean()
btc["SMA_30"] = btc["Close"].rolling(window=30).mean()

# Drop the first few NaNs from rolling calculations
btc = btc.dropna()
