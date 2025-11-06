"""
train_model.py

Trains a predictive model (LSTM neural network) on engineered Bitcoin features.
Uses data from `feature_engineering.py`.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# Load and Prepare Data
# ─────────────────────────────────────────────
def load_feature_data(filepath="data/processed/btc_features.csv"):
    """Load pre-engineered feature data."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def prepare_lstm_data(df, target_col="Close", sequence_length=60):
    """
    Converts tabular data into sequences suitable for LSTM input.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data including target column.
    target_col : str
        Column name to predict.
    sequence_length : int
        Number of time steps per input sequence.

    Returns
    -------
    X, y : np.ndarray
        Arrays for LSTM training.
    scaler : fitted MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    target_idx = df.columns.get_loc(target_col)
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, target_idx])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────
def build_lstm_model(input_shape):
    """Build a simple LSTM neural network."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ─────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────
def train_model():
    """Full training routine: load data, train LSTM, evaluate, save model."""
    df = load_feature_data()
    print(f"✅ Loaded feature data with shape: {df.shape}")

    X, y, scaler = prepare_lstm_data(df)
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate model
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"📈 MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # Plot training performance
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training Performance")
    plt.savefig("data/models/training_loss.png", bbox_inches='tight')
    plt.close()

    # Save model and scaler
    os.makedirs("data/models", exist_ok=True)
    model.save("data/models/lstm_btc_model.h5")
    np.save("data/models/scaler.npy", scaler.data_min_)
    print("💾 Model and scaler saved to data/models/")

    return model, scaler


# ─────────────────────────────────────────────
# Run Training
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
