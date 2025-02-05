import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch stock data
ticker = "AAPL"  # Change this to any stock symbol
stock_data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Calculate moving averages
stock_data["50_MA"] = stock_data["Close"].rolling(window=50).mean()
stock_data["200_MA"] = stock_data["Close"].rolling(window=200).mean()

# Plot stock price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data["Close"], label="Stock Price", color="blue")
plt.plot(stock_data.index, stock_data["50_MA"], label="50-Day MA", color="orange")
plt.plot(stock_data.index, stock_data["200_MA"], label="200-Day MA", color="red")

# Labels and legend
plt.title(f"{ticker} Stock Price & Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
