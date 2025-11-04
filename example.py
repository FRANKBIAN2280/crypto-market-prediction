from src.data_loader import load_btc_data
import matplotlib.pyplot as plt

# Load data (daily candles)
btc = load_btc_data(start_str="1 Jan, 2020", interval="1d")

# Inspect
print(btc.head())

# Plot
plt.figure(figsize=(10,6))
plt.plot(btc.index, btc["Close"], label="BTC/USDT Close", linewidth=1.5)
plt.plot(btc.index, btc["SMA_7"], label="7-day SMA", linestyle='--')
plt.plot(btc.index, btc["SMA_30"], label="30-day SMA", linestyle='--')
plt.title("Bitcoin Price (Binance) with Moving Averages")
plt.xlabel("Date")
plt.ylabel("USDT Price")
plt.legend()
plt.show()
