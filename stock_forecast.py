# stock_forecasting.py
"""
A module for forecasting stock prices and analyzing volatility using ARIMA and GARCH models.
Uses a fixed ARIMA order (0, 1, 1) for forecasting.
Supports data loading, stationarity testing, modeling, evaluation, visualization, and monthly rolling window analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 1: DATA LOADING ---

def load_stock_data(ticker, start_date=None, end_date=None):
    """Download stock data from Yahoo Finance and clean it."""
    try:
        start_date = start_date or (datetime.now() - timedelta(days=365))
        end_date = end_date or datetime.now()
        data = yf.download(ticker, start=start_date, end=end_date)
        # Ensure data is a clean DataFrame with single-level columns
        data = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Fill any missing values using forward fill
        data = data.fillna(method='ffill')
        return data
    except Exception as e:
        raise ValueError(f"Failed to download data for {ticker}: {e}")

def split_data(data, test_ratio=0.2):
    """Split data into training and testing sets."""
    train_size = int((1 - test_ratio) * len(data))
    return data[:train_size], data[train_size:]

# --- SECTION 2: STATIONARITY TESTING ---

def check_stationarity(timeseries):
    """Perform ADF test to check stationarity."""
    try:
        result = adfuller(timeseries.dropna(), autolag='AIC')
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        is_stationary = result[1] <= 0.05
        print(f"\nConclusion: The series is {'stationary' if is_stationary else 'non-stationary'}")
        return is_stationary
    except Exception as e:
        print(f"Stationarity test failed: {e}")
        return False

def plot_acf_pacf(timeseries, lags=40):
    """Plot ACF and PACF for ARIMA parameter selection."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_acf(timeseries.dropna(), ax=plt.gca(), lags=lags)
    plt.title("ACF")
    plt.subplot(1, 2, 2)
    plot_pacf(timeseries.dropna(), ax=plt.gca(), lags=lags, method='ywm')
    plt.title("PACF")
    plt.tight_layout()
    plt.show()

# --- SECTION 3: ARIMA AND SARIMA MODELING ---

def fit_arima(data, order=(3, 3, 5), forecast_steps=30):
    """Fit ARIMA model and forecast."""
    try:
        model = ARIMA(data, order=order).fit()
        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return model, forecast_mean, conf_int
    except Exception as e:
        raise ValueError(f"ARIMA fitting failed: {e}")
    
def fit_sarima(data, order=(3, 3, 5), seasonal_order=(1, 1, 1, 12), forecast_steps=30):
    """Fit SARIMA model and forecast."""
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order).fit(disp=False)
        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return model, forecast_mean, conf_int
    except Exception as e:
        raise ValueError(f"SARIMA fitting failed: {e}")

# --- SECTION 4: GARCH MODELING ---

def fit_garch(returns, p=5, q=1, forecast_steps=30):
    """Fit GARCH model and forecast volatility."""
    try:
        model = arch_model(returns, vol='Garch', p=p, q=q)
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=forecast_steps)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :])
        return res, vol_forecast
    except Exception as e:
        raise ValueError(f"GARCH fitting failed: {e}")

# --- SECTION 5: EVALUATION ---

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance."""
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} Evaluation")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}

# --- SECTION 6: VISUALIZATION ---

def plot_forecast(historical, forecast, forecast_index, conf_int=None, title="Stock Price Forecast", ticker="Stock"):
    """Plot historical and forecasted prices with confidence intervals."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical.index, historical, label="Historical Close", color="blue")
    plt.plot(forecast_index, forecast, label="Forecasted Close", color="red")
    if conf_int is not None:
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                         color="red", alpha=0.2, label="95% Confidence Interval")
    plt.title(f"{ticker} {title}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sarima_forecast(historical, forecast, forecast_index, conf_int=None, title="SARIMA Forecast", ticker="Stock"):
    """Plot historical and forecasted prices with confidence intervals for SARIMA."""
    plt.figure(figsize=(12, 6))
    plt.plot(historical.index, historical, label="Historical Close", color="blue")
    plt.plot(forecast_index, forecast, label="Forecasted Close", color="red")
    if conf_int is not None:
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                         color="red", alpha=0.2, label="95% Confidence Interval")
    plt.title(f"{ticker} {title}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_volatility(returns, vol_forecast, forecast_index, ticker="Stock"):
    """Plot recent returns and forecasted volatility."""
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index[-300:], returns[-300:], label="Recent Returns", alpha=0.5, color="blue")
    plt.plot(forecast_index, vol_forecast, label="GARCH Volatility Forecast", color="purple")
    plt.title(f"{ticker} Return Volatility Forecast")
    plt.xlabel("Date")
    plt.ylabel("Volatility / Return (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- SECTION 7: ROLLING WINDOW ANALYSIS ---

def rolling_window_analysis(data, window_size=30):
    """Compute price changes and volatility levels over monthly rolling windows."""
    results = []
    
    def classify_volatility(std_dev):
        # Handle scalar, NaN, or Series values
        if isinstance(std_dev, (pd.Series, pd.DataFrame)):
            std_dev = std_dev.item() if std_dev.size == 1 else np.nan
        if pd.isna(std_dev) or std_dev == 0:
            return 'Unknown'
        if std_dev <= 0.2:
            return 'Low'
        elif std_dev > 0.8:
            return 'High'
        else:
            return 'Medium'

    data = data.sort_values('Date').set_index('Date')
    # Group data by month to get the first trading day of each month
    monthly_starts = data.resample('MS').first().index
    for start_date in monthly_starts:
        # Find the window ending approximately one month later
        end_date = start_date + pd.offsets.MonthEnd(1)
        window = data.loc[start_date:end_date]
        
        if len(window) < 10:  # Skip windows with too few trading days
            continue
            
        try:
            start_date = window.index[0].date()
            end_date = window.index[-1].date()
            # Ensure price_change is a scalar
            close_start = window['Close'].iloc[0]
            close_end = window['Close'].iloc[-1]
            price_change = (close_end - close_start) / close_start
            returns = window['Close'].pct_change().dropna()
            # Compute std_dev only if returns has enough data
            volatility_std = returns.std() if len(returns) > 1 else np.nan
            vol_level = classify_volatility(volatility_std)
            
            # Debug output to inspect std_dev and classification
            print(f"Window {start_date} to {end_date}: len(returns) = {len(returns)}, std_dev = {volatility_std}, Volatility_Level = {vol_level}")
            
            results.append({
                'Window_Start_Date': start_date,
                'Window_End_Date': end_date,
                'Average_Price_Change': round(price_change, 6),
                'Volatility_Level': vol_level
            })
        except Exception as e:
            print(f"Window {start_date} to {end_date} failed: {e}")
            continue
    
    return pd.DataFrame(results)

# --- SECTION 8: MAIN WORKFLOW ---

def forecast_stock(ticker="MSFT", forecast_steps=30, test_ratio=0.2):
    """Main function to forecast stock prices and analyze volatility using ARIMA(0,1,1)."""
    try:
        # Load and prepare data
        data = load_stock_data(ticker)
        train, test = split_data(data, test_ratio)
        close = data.set_index('Date')['Close']
        
        # Check stationarity
        print(f"\nStationarity Test for {ticker} Close Price")
        is_stationary = check_stationarity(close)
        if not is_stationary:
            print("Differencing applied to make series stationary")
            close_diff = close.diff().dropna()
            check_stationarity(close_diff)
        
        # Plot ACF/PACF
        plot_acf_pacf(close)
        
        # Fit ARIMA with fixed order (0, 1, 1)
        train_close = train.set_index('Date')['Close']
        model, forecast, conf_int = fit_arima(train_close, order=(3, 3, 5), forecast_steps=forecast_steps)
        forecast_index = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), 
                                    periods=forecast_steps, freq='D')
        
        # Evaluate ARIMA
        test_close = test.set_index('Date')['Close'][:forecast_steps]
        if len(test_close) == len(forecast):
            evaluate_model(test_close, forecast, f"ARIMA ({ticker})")
        
        # Plot ARIMA forecast
        plot_forecast(close, forecast, forecast_index, conf_int, 
                     title="Price Forecast", ticker=ticker)
        
        plot_sarima_forecast(close, forecast, forecast_index, conf_int,
                             title="SARIMA Forecast", ticker=ticker)
        
        # Fit GARCH
        returns = 100 * close.pct_change().dropna()
        _, vol_forecast = fit_garch(returns[:-forecast_steps])
        plot_volatility(returns, vol_forecast, forecast_index, ticker)
        
        # Rolling window analysis
        summary_df = rolling_window_analysis(data)
        print(f"\nRolling Window Analysis for {ticker}:")
        print(summary_df.to_csv(index=False))
        
        return {
            'data': data,
            'arima_model': model,
            'arima_forecast': forecast,
            'garch_vol_forecast': vol_forecast,
            'summary': summary_df
        }
    except Exception as e:
        print(f"Forecasting failed: {e}")
        return {}

if __name__ == "__main__":
    results = forecast_stock(ticker="MSFT")