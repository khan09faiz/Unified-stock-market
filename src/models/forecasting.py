"""
Stock Forecasting Module using ARIMA and GARCH models.
This module provides comprehensive time series forecasting capabilities for stock prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from arch import arch_model
    FORECASTING_AVAILABLE = True
except ImportError:
    FORECASTING_AVAILABLE = False
    print("Warning: Some forecasting dependencies not available. Install statsmodels and arch for full functionality.")


class StockForecaster:
    """Comprehensive stock price forecasting using time series models."""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.evaluation_metrics = {}
    
    def load_data(self, ticker: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None, period: str = "2y") -> pd.DataFrame:
        """
        Load stock data for forecasting.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Period if dates not specified
            
        Returns:
            DataFrame with stock data
        """
        try:
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date)
            else:
                data = yf.download(ticker, period=period)
            
            # Clean and prepare data
            data = data.reset_index()
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            self.data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load data for {ticker}: {e}")
    
    def check_stationarity(self, series: pd.Series, verbose: bool = True) -> bool:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series: Time series data
            verbose: Whether to print results
            
        Returns:
            Boolean indicating if series is stationary
        """
        if not FORECASTING_AVAILABLE:
            return True  # Assume stationary for fallback
            
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            if verbose:
                print(f"ADF Statistic: {result[0]:.4f}")
                print(f"p-value: {result[1]:.4f}")
                print("Critical Values:")
                for key, value in result[4].items():
                    print(f"\\t{key}: {value:.3f}")
            
            is_stationary = result[1] <= 0.05
            
            if verbose:
                print(f"\\nSeries is {'stationary' if is_stationary else 'non-stationary'}")
            
            return is_stationary
            
        except Exception as e:
            print(f"Stationarity test failed: {e}")
            return True
    
    def plot_diagnostics(self, series: pd.Series, lags: int = 40):
        """
        Plot ACF and PACF for model diagnostics.
        
        Args:
            series: Time series data
            lags: Number of lags to plot
        """
        if not FORECASTING_AVAILABLE:
            print("Diagnostic plots require statsmodels")
            return
            
        plt.figure(figsize=(15, 8))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(series.index, series.values)
        plt.title("Time Series")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # ACF
        plt.subplot(2, 2, 2)
        plot_acf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.title("Autocorrelation Function")
        
        # PACF
        plt.subplot(2, 2, 3)
        plot_pacf(series.dropna(), ax=plt.gca(), lags=lags, method='ywm')
        plt.title("Partial Autocorrelation Function")
        
        # Distribution
        plt.subplot(2, 2, 4)
        plt.hist(series.dropna(), bins=30, alpha=0.7)
        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    
    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), 
                  forecast_steps: int = 30) -> Dict[str, Any]:
        """
        Fit ARIMA model and generate forecasts.
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
            forecast_steps: Number of steps to forecast
            
        Returns:
            Dictionary with model results
        """
        if not FORECASTING_AVAILABLE:
            # Fallback: simple linear trend
            return self._simple_forecast(series, forecast_steps)
        
        try:
            model = ARIMA(series, order=order).fit()
            forecast = model.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            result = {
                'model': model,
                'forecast': forecast_mean,
                'conf_int': conf_int,
                'aic': model.aic,
                'bic': model.bic,
                'order': order
            }
            
            self.models['arima'] = result
            return result
            
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            return self._simple_forecast(series, forecast_steps)
    
    def fit_sarima(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1),
                   seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                   forecast_steps: int = 30) -> Dict[str, Any]:
        """
        Fit SARIMA model and generate forecasts.
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            forecast_steps: Number of steps to forecast
            
        Returns:
            Dictionary with model results
        """
        if not FORECASTING_AVAILABLE:
            return self._simple_forecast(series, forecast_steps)
        
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order).fit(disp=False)
            forecast = model.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            result = {
                'model': model,
                'forecast': forecast_mean,
                'conf_int': conf_int,
                'aic': model.aic,
                'bic': model.bic,
                'order': order,
                'seasonal_order': seasonal_order
            }
            
            self.models['sarima'] = result
            return result
            
        except Exception as e:
            print(f"SARIMA fitting failed: {e}")
            return self._simple_forecast(series, forecast_steps)
    
    def fit_garch(self, returns: pd.Series, p: int = 1, q: int = 1, 
                  forecast_steps: int = 30) -> Dict[str, Any]:
        """
        Fit GARCH model for volatility forecasting.
        
        Args:
            returns: Return series
            p: GARCH p parameter
            q: GARCH q parameter
            forecast_steps: Number of steps to forecast
            
        Returns:
            Dictionary with model results
        """
        if not FORECASTING_AVAILABLE:
            # Simple volatility forecast
            vol_mean = returns.std()
            vol_forecast = np.full(forecast_steps, vol_mean)
            return {'volatility_forecast': vol_forecast}
        
        try:
            model = arch_model(returns * 100, vol='Garch', p=p, q=q)  # Scale returns
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=forecast_steps)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # Scale back
            
            result = {
                'model': res,
                'volatility_forecast': vol_forecast,
                'log_likelihood': res.loglikelihood
            }
            
            self.models['garch'] = result
            return result
            
        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            vol_mean = returns.std()
            vol_forecast = np.full(forecast_steps, vol_mean)
            return {'volatility_forecast': vol_forecast}
    
    def _simple_forecast(self, series: pd.Series, forecast_steps: int) -> Dict[str, Any]:
        """
        Simple linear trend forecast as fallback.
        
        Args:
            series: Time series data
            forecast_steps: Number of steps to forecast
            
        Returns:
            Dictionary with simple forecast
        """
        # Calculate trend
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 2:
            # Constant forecast
            forecast = np.full(forecast_steps, y_valid[-1] if len(y_valid) > 0 else 0)
        else:
            # Linear trend
            trend = np.polyfit(x_valid, y_valid, 1)
            future_x = np.arange(len(series), len(series) + forecast_steps)
            forecast = np.polyval(trend, future_x)
        
        return {
            'forecast': pd.Series(forecast),
            'method': 'simple_trend'
        }
    
    def evaluate_forecast(self, actual: pd.Series, predicted: pd.Series, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate forecast performance.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Align series
            min_len = min(len(actual), len(predicted))
            actual = actual.iloc[:min_len]
            predicted = predicted.iloc[:min_len] if hasattr(predicted, 'iloc') else predicted[:min_len]
            
            # Remove NaN values
            mask = ~(pd.isna(actual) | pd.isna(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return {'error': 'No valid data for evaluation'}
            
            metrics = {
                'MSE': mean_squared_error(actual_clean, predicted_clean),
                'RMSE': np.sqrt(mean_squared_error(actual_clean, predicted_clean)),
                'MAE': mean_absolute_error(actual_clean, predicted_clean),
                'MAPE': np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
            }
            
            try:
                metrics['R2'] = r2_score(actual_clean, predicted_clean)
            except:
                metrics['R2'] = 0.0
            
            self.evaluation_metrics[model_name] = metrics
            return metrics
            
        except Exception as e:
            print(f"Evaluation failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def plot_forecast(self, historical: pd.Series, forecast: pd.Series,
                     conf_int: Optional[pd.DataFrame] = None, 
                     title: str = "Stock Price Forecast"):
        """
        Plot historical data and forecasts.
        
        Args:
            historical: Historical price data
            forecast: Forecasted prices
            conf_int: Confidence intervals (optional)
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical.index, historical.values, label="Historical", color="blue")
        
        # Create forecast index
        last_date = historical.index[-1]
        if hasattr(last_date, 'to_pydatetime'):
            last_date = last_date.to_pydatetime()
        
        forecast_index = pd.date_range(start=last_date, periods=len(forecast)+1, freq='D')[1:]
        
        # Plot forecast
        plt.plot(forecast_index, forecast, label="Forecast", color="red", linestyle='--')
        
        # Plot confidence intervals if available
        if conf_int is not None:
            plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                           color="red", alpha=0.2, label="95% Confidence Interval")
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_rolling_analysis(self, data: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
        """
        Create rolling window analysis for RL environment.
        
        Args:
            data: Stock data DataFrame
            window_size: Rolling window size
            
        Returns:
            DataFrame with rolling analysis
        """
        def classify_volatility(std_dev):
            if pd.isna(std_dev) or std_dev == 0:
                return 'Unknown'
            if std_dev <= 0.02:
                return 'Low'
            elif std_dev > 0.05:
                return 'High'
            else:
                return 'Medium'
        
        def classify_trend(price_change):
            if pd.isna(price_change):
                return 'Stable'
            if price_change > 0.02:
                return 'Uptrend'
            elif price_change < -0.02:
                return 'Downtrend'
            else:
                return 'Stable'
        
        results = []
        data_sorted = data.sort_values('Date').set_index('Date')
        
        # Monthly analysis
        monthly_starts = data_sorted.resample('MS').first().index
        
        for start_date in monthly_starts:
            end_date = start_date + pd.offsets.MonthEnd(1)
            window = data_sorted.loc[start_date:end_date]
            
            if len(window) < 5:  # Need minimum data
                continue
            
            try:
                start_price = window['Close'].iloc[0]
                end_price = window['Close'].iloc[-1]
                price_change = (end_price - start_price) / start_price
                
                returns = window['Close'].pct_change().dropna()
                volatility_std = returns.std() if len(returns) > 1 else 0.0
                
                results.append({
                    'Window_Start_Date': start_date.date(),
                    'Window_End_Date': end_date.date(),
                    'Start_Price': round(start_price, 2),
                    'End_Price': round(end_price, 2),
                    'Average_Price_Change': round(price_change, 6),
                    'Volatility_Level': classify_volatility(volatility_std),
                    'Trend_Classification': classify_trend(price_change),
                    'Volume_Mean': round(window['Volume'].mean(), 0),
                    'Sentiment_Score': 0.0  # Placeholder for sentiment
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(results)


def forecast_stock_prices(ticker: str, forecast_days: int = 30, 
                         models: list = ['arima', 'sarima']) -> Dict[str, Any]:
    """
    Convenience function for stock price forecasting.
    
    Args:
        ticker: Stock ticker symbol
        forecast_days: Number of days to forecast
        models: List of models to use
        
    Returns:
        Dictionary with forecasting results
    """
    forecaster = StockForecaster()
    
    # Load data
    data = forecaster.load_data(ticker)
    price_series = data.set_index('Date')['Close']
    
    results = {'ticker': ticker, 'data': data, 'forecasts': {}}
    
    # Check stationarity
    is_stationary = forecaster.check_stationarity(price_series)
    results['is_stationary'] = is_stationary
    
    # Fit models
    if 'arima' in models:
        arima_result = forecaster.fit_arima(price_series, forecast_steps=forecast_days)
        results['forecasts']['arima'] = arima_result
    
    if 'sarima' in models:
        sarima_result = forecaster.fit_sarima(price_series, forecast_steps=forecast_days)
        results['forecasts']['sarima'] = sarima_result
    
    # GARCH for volatility
    returns = price_series.pct_change().dropna()
    garch_result = forecaster.fit_garch(returns, forecast_steps=forecast_days)
    results['forecasts']['garch'] = garch_result
    
    # Rolling analysis for RL
    rolling_analysis = forecaster.create_rolling_analysis(data)
    results['rolling_analysis'] = rolling_analysis
    
    return results
