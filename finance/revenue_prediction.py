# revenue_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_historical_revenue_data(years, start_date=None):
    """
    Generate simulated historical revenue data for a given number of years.
    
    Parameters:
    - years: Number of years of historical data to generate
    - start_date: Starting date for the data (default is today)
    
    Returns:
    - DataFrame with 'Date' and 'Revenue' columns
    """
    if start_date is None:
        start_date = datetime.now().date() - timedelta(days=years*365)
    
    end_date = start_date + timedelta(days=years*365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Simulate revenue with some seasonal trend and randomness
    base_revenue = 100000  # Base revenue
    revenue = base_revenue + np.cumsum(np.random.randn(len(date_range)) * 5000)  # Adding trend and noise
    revenue += np.sin(np.arange(len(date_range)) * (2 * np.pi / 12)) * 10000  # Adding seasonal component
    
    df = pd.DataFrame({'Date': date_range, 'Revenue': revenue})
    return df

def predict_future_revenue(historical_data, forecast_periods, window_size=3):
    """
    Predict future revenue using simple moving average.
    
    Parameters:
    - historical_data: DataFrame with historical revenue data
    - forecast_periods: Number of future periods to forecast
    - window_size: Size of the moving average window
    
    Returns:
    - DataFrame with both historical and predicted data
    """
    # Calculate moving average for historical data
    historical_data['MA_Revenue'] = historical_data['Revenue'].rolling(window=window_size).mean()
    
    # Predict future revenue
    last_date = historical_data['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='M')
    future_revenue = []
    
    for _ in range(forecast_periods):
        last_ma = historical_data['MA_Revenue'].iloc[-1]
        future_revenue.append(last_ma + np.random.normal(0, 1000))  # Adding some randomness to predictions
    
    future_df = pd.DataFrame({'Date': future_dates, 'Revenue': future_revenue})
    future_df['MA_Revenue'] = future_df['Revenue'].rolling(window=window_size).mean()
    
    # Combine historical and future data
    result = pd.concat([historical_data, future_df], ignore_index=True)
    return result

if __name__ == "__main__":
    # Generate 5 years of monthly historical revenue data
    historical_revenue = generate_historical_revenue_data(years=5)
    
    # Predict the next 12 months
    forecast = predict_future_revenue(historical_revenue, forecast_periods=12, window_size=3)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(forecast['Date'], forecast['Revenue'], label='Actual Revenue')
    plt.plot(forecast['Date'], forecast['MA_Revenue'], label='Moving Average Revenue', linestyle='--')
    plt.title('Historical and Predicted Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print last few records to see the predictions
    print(forecast.tail(15))
