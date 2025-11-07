"""
predictor.py

Loads a trained LSTM model and scaler to predict future Bitcoin prices.
Fetches the latest Binance data and runs inference.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
from feature_engineering import prepare_features
import os
import datetime as dt

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Load Binance client
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)


# ─────────────────────────────────────────────
# Data Fetching Helper
# ─────────────────────────────────────────────
def get_recent_btc_data(days=90, interval="1d"):
    """
    Fetch the latest BTC/USDT data from Binance.

    Parameters
    ----------
    days : int
        Number of past days to retrieve.
    interval : str
        Interval string (e.g. '1d', '1h').

    Returns
    -------
    pd.DataFrame
        Recent Bitcoin OHLCV data.
    """
    end_time = dt.datetime.utcnow()
    start_time = end_time - dt.timedelta(days=days)

    klines = client.get_historical_klines(
        "BTCUSDT",
        interval,
        start_str=start_time.strftime("%d %b, %Y"),
        end_str=end_time.strftime("%d %b, %Y")
    )

    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])

    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")

    df.set_index("Close time", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Return"] = df["Close"].pct_change()

    return df.dropna()


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def predict_next_price(model_path="data/models/lstm_btc_model.h5",
                       scaler_path="data/models/scaler.npy",
                       days=90):
    """
    Predict the next closing price for Bitcoin using the trained model.

    Parameters
    ----------
    model_path : str
        Path to trained Keras model.
    scaler_path : str
        Path to saved scaler data.
    days : int
        How many recent days to use for input sequence.

    Returns
    -------
    float
        Predicted next-day closing price.
    """
    # Load model
    model = load_model(model_path)

    # Fetch and process data
    df = get_recent_btc_data(days=days)
    df_feat = prepare_features(df)

    # Load saved scaler
    scaler = MinMaxScaler()
    scaler.data_min_ = np.load(scaler_path)
    scaler.data_max_ = df_feat.max().values  # rough recovery; can refine with pickle if needed

    scaled_data = scaler.fit_transform(df_feat)

    X_input = np.expand_dims(scaled_data[-60:], axis=0)

    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(
        np.concatenate([np.zeros((1, df_feat.shape[1]-1)), [[pred_scaled[0][0]]]], axis=1)
    )[0, -1]

    latest_close = df_feat["Close"].iloc[-1]
    change_pct = (pred_price - latest_close) / latest_close * 100

    print(f"🪙 Latest close: {latest_close:.2f} USDT")
    print(f"🔮 Predicted next close: {pred_price:.2f} USDT")
    print(f"📊 Expected change: {change_pct:+.2f}%")

    return pred_price


# ─────────────────────────────────────────────
# Run Prediction
# ─────────────────────────────────────────────
if __name__ == "__main__":
    predict_next_price()
