import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
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
def arima_forecast(historical_data, forecast_periods):
    """
    Use ARIMA to forecast future revenue.
    
    Parameters:
    - historical_data: DataFrame with historical revenue data
    - forecast_periods: Number of future periods to forecast
    
    Returns:
    - DataFrame with both historical and predicted data
    """
    # Prepare the data
    historical_data = historical_data.set_index('Date')
    revenue_series = historical_data['Revenue']
    
    # Use auto_arima to find the best ARIMA model
    model = auto_arima(revenue_series, start_p=1, start_q=1, test='adf', seasonal=False, 
                       stepwise=True, suppress_warnings=True, error_action="ignore", maxiter=100)
    
    # Fit the model
    fitted_model = model.fit(revenue_series)
    
    # Forecast
    forecast = fitted_model.forecast(forecast_periods)
    
    # Create future dates
    last_date = historical_data.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='M')
    
    # Combine historical data with forecast
    forecast_df = pd.DataFrame({'Date': future_dates, 'Revenue': forecast})
    result = pd.concat([historical_data.reset_index(), forecast_df], ignore_index=True)
    
    return result
if __name__ == "__main__":
    # Generate 5 years of monthly historical revenue data
    historical_revenue = generate_historical_revenue_data(years=5)
    
    # Predict the next 12 months using ARIMA
    forecast = arima_forecast(historical_revenue, forecast_periods=12)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(forecast['Date'], forecast['Revenue'], label='Revenue')
    plt.title('Historical and ARIMA Predicted Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    
    # Highlight the forecast period
    plt.axvline(x=historical_revenue['Date'].max(), color='r', linestyle='--', label='Forecast Start')
    plt.show()
    
    # Print last few records to see the predictions
    print(forecast.tail(15))
