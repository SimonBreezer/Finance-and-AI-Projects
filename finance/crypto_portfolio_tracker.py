import requests
import pandas as pd
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Define your crypto holdings
portfolio = {
    "bitcoin": 2.5,  # 2.5 BTC
    "ethereum": 2.0,  # 2 ETH
    "dogecoin": 1000000,  # 1,000,000 DOGE
    "daddy": 1000000,  #(Solana token)
    "xrp": 500,  # 500 XRP
}

# Fetch live prices
prices = cg.get_price(ids=list(portfolio.keys()), vs_currencies="usd")

# Calculate portfolio value
portfolio_value = {coin: portfolio[coin] * prices[coin]["usd"] for coin in portfolio}

# Convert to DataFrame and save
df = pd.DataFrame(portfolio_value.items(), columns=["Crypto", "Value (USD)"])
df.to_csv("finance/crypto_portfolio.csv", index=False)

# Print summary
print(df)
print(f"Total Portfolio Value: ${df['Value (USD)'].sum():,.2f}")
