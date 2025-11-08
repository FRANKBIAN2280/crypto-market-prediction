"""
feature_engineering.py

Creates technical indicators and engineered features from raw Bitcoin data.
Designed to work with DataFrames output by `data_loader.py`.
"""

# TODO: fix import errors

import pandas as pd
import numpy as np
import ta  # Technical Analysis library: pip install ta


# ─────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────
def add_technical_indicators(df):
    """
    Add common technical indicators to the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'Open', 'High', 'Low', 'Close', and 'Volume'

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns for indicators
    """
    df = df.copy()

    # --- Moving Averages ---
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_14"] = df["Close"].ewm(span=14, adjust=False).mean()

    # --- Momentum Indicators ---
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD"] = ta.trend.MACD(close=df["Close"]).macd()
    df["MACD_signal"] = ta.trend.MACD(close=df["Close"]).macd_signal()

    # --- Volatility Indicators ---
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["Bollinger_Mavg"] = bollinger.bollinger_mavg()
    df["Bollinger_High"] = bollinger.bollinger_hband()
    df["Bollinger_Low"] = bollinger.bollinger_lband()
    df["Bollinger_Width"] = df["Bollinger_High"] - df["Bollinger_Low"]

    # --- Volume Indicators ---
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()

    # --- Rate of Change ---
    df["ROC_10"] = ta.momentum.ROCIndicator(close=df["Close"], window=10).roc()

    # Drop NaN rows created by rolling calculations
    df.dropna(inplace=True)
    return df


def add_time_features(df):
    """
    Add time-based features that can help models learn seasonality and patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by datetime (from Binance 'Close time')

    Returns
    -------
    pd.DataFrame
        DataFrame with time-derived features
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["hour"] = df.index.hour if hasattr(df.index, "hour") else 0
    return df


def prepare_features(df):
    """
    Complete feature engineering pipeline.

    Combines technical indicators + time features into one processed dataset.
    """
    df = add_technical_indicators(df)
    df = add_time_features(df)
    return df

#TODO: fix error in example code

# ─────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_btc_data

    # Load raw data from Binance
    btc = load_btc_data(start_str="1 Jan, 2021", interval="1d")

    # Add indicators
    btc_feat = prepare_features(btc)

    print("✅ Feature engineering complete!")
    print(btc_feat.tail())

    # Save processed data
    btc_feat.to_csv("data/processed/btc_features.csv")
    print("💾 Saved to data/processed/btc_features.csv")
