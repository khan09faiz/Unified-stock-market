"""
Data processing utilities for stock market analysis.
This module handles data loading, preprocessing, and feature engineering.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StockDataProcessor:
    """Class to handle stock data processing and feature engineering."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_stock_data(self, ticker: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, period: str = "1y") -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Period for data (if start_date and end_date not provided)
        
        Returns:
            DataFrame with stock data
        """
        try:
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date)
            else:
                data = yf.download(ticker, period=period)
            
            # Reset index and clean data
            data = data.reset_index()
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            self.data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to download data for {ticker}: {e}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the stock data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        df = data.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_middle = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Middle'] = bb_middle
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def create_rolling_windows(self, data: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
        """
        Create rolling window analysis for the data - simplified version.
        
        Args:
            data: DataFrame with stock data
            window_size: Size of rolling window in days
            
        Returns:
            DataFrame with rolling window analysis
        """
        def classify_volatility(std_dev):
            if pd.isna(std_dev) or std_dev == 0:
                return 'Low'
            elif std_dev <= 0.02:
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
        data = data.reset_index(drop=True)
        
        # Create rolling windows with 50% overlap
        step_size = max(1, window_size // 2)
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size]
            
            if len(window) < window_size:
                continue
            
            try:
                start_price = window['Close'].iloc[0]
                end_price = window['Close'].iloc[-1]
                price_change = (end_price - start_price) / start_price
                
                returns = window['Close'].pct_change().dropna()
                volatility_std = returns.std() if len(returns) > 1 else 0.0
                
                # Get dates if available
                if 'Date' in window.columns:
                    start_date = window['Date'].iloc[0]
                    end_date = window['Date'].iloc[-1]
                else:
                    start_date = f"Window_{i}_start"
                    end_date = f"Window_{i}_end"
                
                results.append({
                    'Window_Start_Date': start_date,
                    'Window_End_Date': end_date,
                    'Start_Price': round(start_price, 2),
                    'End_Price': round(end_price, 2),
                    'Average_Price_Change': round(price_change, 6),
                    'Volatility_Level': classify_volatility(volatility_std),
                    'Trend_Classification': classify_trend(price_change),
                    'Volume_Mean': round(window['Volume'].mean(), 0) if 'Volume' in window.columns else 0
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def split_data(self, data: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            data: Input DataFrame
            test_ratio: Ratio of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_size = int((1 - test_ratio) * len(data))
        return data[:train_size].copy(), data[train_size:].copy()
    
    def preprocess_for_rl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for reinforcement learning environment.
        
        Args:
            data: DataFrame with stock data and features
            
        Returns:
            Preprocessed DataFrame ready for RL
        """
        df = data.copy()
        
        # If we don't have processed data, create minimal features from raw data
        if 'Average_Price_Change' not in df.columns and 'Close' in df.columns:
            df['Average_Price_Change'] = df['Close'].pct_change().fillna(0)
            df['End_Price'] = df['Close']
            df['Volatility_Level'] = 'Medium'  # Default
            df['Trend_Classification'] = 'Stable'  # Default
        
        # Normalize price changes if they exist
        if 'Average_Price_Change' in df.columns:
            price_change_std = df['Average_Price_Change'].std()
            if price_change_std > 0:
                df['Average_Price_Change'] = (df['Average_Price_Change'] - df['Average_Price_Change'].mean()) / price_change_std
            else:
                df['Average_Price_Change'] = 0.0
        
        # Add sentiment score placeholder
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 0.0
        
        # Ensure we have minimum required columns
        required_columns = ['End_Price', 'Average_Price_Change', 'Sentiment_Score']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'End_Price':
                    df[col] = df['Close'] if 'Close' in df.columns else 100.0
                elif col == 'Average_Price_Change':
                    df[col] = 0.0
                else:
                    df[col] = 0.0
        
        # Add categorical mappings for levels
        if 'Volatility_Level' in df.columns:
            volatility_map = {'Low': 0, 'Medium': 1, 'High': 2}
            df['Volatility_Level_Numeric'] = df['Volatility_Level'].map(volatility_map).fillna(1)
        else:
            df['Volatility_Level_Numeric'] = 1
            
        if 'Trend_Classification' in df.columns:
            trend_map = {'Downtrend': -1, 'Stable': 0, 'Uptrend': 1}
            df['Trend_Classification_Numeric'] = df['Trend_Classification'].map(trend_map).fillna(0)
        else:
            df['Trend_Classification_Numeric'] = 0
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        return df


def load_and_process_stock_data(ticker: str, start_date: str = None, 
                              end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and process stock data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Tuple of (processed_data, rolling_windows_data)
    """
    processor = StockDataProcessor()
    
    # Load data
    raw_data = processor.load_stock_data(ticker, start_date, end_date)
    
    # Add technical indicators
    processed_data = processor.calculate_technical_indicators(raw_data)
    
    # Create rolling windows
    rolling_data = processor.create_rolling_windows(raw_data)
    
    return processed_data, rolling_data
