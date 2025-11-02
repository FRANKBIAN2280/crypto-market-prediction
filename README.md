# 🪙 Bitcoin Price Prediction Project

This project aims to **analyze and predict Bitcoin (BTC-USD) prices** using historical data fetched from **Yahoo Finance** via the `yfinance` Python library.  
It provides a foundation for building machine learning and deep learning models (e.g., LSTM, GRU, Transformer) to forecast future Bitcoin trends.

---

## 📂 Project Structure

btc-predictor/
│
├── data/ # Saved datasets (raw and processed)
├── notebooks/ # Jupyter notebooks for exploration and modeling
├── src/
│ ├── data_loader.py # Script to download and preprocess Bitcoin data
│ ├── feature_engineering.py # (optional) Add technical indicators here
│ ├── train_model.py # (optional) Model training script
│ └── predict.py # (optional) Generate forecasts
├── app.py # (optional) Streamlit or FastAPI app
├── requirements.txt # Project dependencies
└── README.md # Project documentation

