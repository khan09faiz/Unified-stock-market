"""
Sample data generator for testing when network access is unavailable.
This module creates realistic stock data for demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import random


class SampleDataGenerator:
    """Generate realistic sample stock data for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_stock_data(self, ticker: str = "SAMPLE", 
                          days: int = 250, 
                          initial_price: float = 150.0) -> pd.DataFrame:
        """
        Generate realistic stock market data.
        
        Args:
            ticker: Stock symbol
            days: Number of days of data to generate
            initial_price: Starting price for the stock
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Remove weekends (simplified - doesn't account for holidays)
        dates = [d for d in dates if d.weekday() < 5][:days]
        
        # Generate price data with realistic patterns
        prices = []
        volumes = []
        price = initial_price
        
        # Trend components
        long_term_trend = np.random.choice([-0.0001, 0.0001, 0.0002], p=[0.3, 0.4, 0.3])
        volatility = 0.02
        
        for i in range(len(dates)):
            # Add trend component
            trend_component = long_term_trend * i
            
            # Add cyclical component (simulate market cycles)
            cycle_component = 0.001 * np.sin(2 * np.pi * i / 60)  # ~60 day cycles
            
            # Add random walk
            random_component = np.random.normal(0, volatility)
            
            # Combine components
            daily_return = trend_component + cycle_component + random_component
            price = price * (1 + daily_return)
            
            # Ensure price doesn't go below $10
            price = max(price, 10.0)
            
            # Generate OHLC from close price
            daily_volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + daily_volatility/2)
            low = price * (1 - daily_volatility/2)
            open_price = low + (high - low) * np.random.random()
            
            # Volume (higher volume on volatile days)
            base_volume = 1000000
            volume_multiplier = 1 + 2 * daily_volatility
            volume = int(base_volume * volume_multiplier * (0.5 + np.random.random()))
            
            prices.append([open_price, high, low, price])
            volumes.append(volume)
        
        # Create DataFrame
        df = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close'])
        df['Volume'] = volumes
        df['Date'] = dates[:len(df)]
        df['Adj Close'] = df['Close']  # Simplified - same as close
        
        return df
    
    def generate_sentiment_data(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Generate sample sentiment analysis data.
        
        Args:
            num_samples: Number of sentiment samples to generate
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Sample financial news headlines/text
        sample_texts = [
            "The company reported strong quarterly earnings, beating analyst expectations.",
            "Market volatility continues as investors remain cautious about economic outlook.",
            "New product launch drives investor optimism and stock price surge.",
            "Regulatory concerns weigh on sector performance amid uncertainty.",
            "Technical analysis suggests potential breakout above key resistance levels."
        ]
        
        # Generate realistic sentiment scores (-1 to 1)
        sentiment_scores = []
        for _ in range(num_samples):
            # Bias towards slightly negative sentiment (realistic for financial news)
            score = np.random.normal(-0.1, 0.4)
            score = np.clip(score, -1, 1)
            sentiment_scores.append(score)
        
        return {
            'texts': sample_texts[:num_samples],
            'sentiment_scores': sentiment_scores,
            'sentiment_score': np.mean(sentiment_scores),
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'model_used': 'sample_generator'
        }
    
    def generate_forecast_results(self, current_price: float = 150.0, 
                                forecast_days: int = 30) -> Dict[str, Any]:
        """
        Generate sample forecasting results.
        
        Args:
            current_price: Current stock price
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Generate realistic forecast
        trend = np.random.normal(0.001, 0.0005, forecast_days).cumsum()
        noise = np.random.normal(0, 0.01, forecast_days)
        
        # Create forecast prices
        forecast_prices = []
        price = current_price
        
        for i in range(forecast_days):
            daily_change = trend[i] + noise[i]
            price = price * (1 + daily_change)
            forecast_prices.append(price)
        
        # Generate confidence intervals
        std_dev = np.std(forecast_prices) * 0.1
        upper_bound = [p + 1.96 * std_dev for p in forecast_prices]
        lower_bound = [p - 1.96 * std_dev for p in forecast_prices]
        
        return {
            'forecast_prices': forecast_prices,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'model_performance': {
                'mape': round(np.random.uniform(2.5, 5.0), 2),
                'rmse': round(np.random.uniform(3.0, 8.0), 2),
                'mae': round(np.random.uniform(2.0, 6.0), 2),
                'r2_score': round(np.random.uniform(0.75, 0.95), 3)
            },
            'model_used': 'SARIMA_sample',
            'aic': round(np.random.uniform(2500, 3000), 2)
        }
    
    def generate_trading_results(self, initial_capital: float = 10000, 
                               episodes: int = 20) -> Dict[str, Any]:
        """
        Generate sample RL trading results.
        
        Args:
            initial_capital: Starting capital
            episodes: Number of training episodes
            
        Returns:
            Dictionary with trading results
        """
        # Generate episode rewards (improving over time)
        episode_rewards = []
        net_worths = []
        
        for episode in range(episodes):
            # Simulate learning - gradually improving performance
            base_reward = -100 + (episode * 15)  # Improving trend
            noise = np.random.normal(0, 50)
            reward = base_reward + noise
            episode_rewards.append(reward)
            
            # Net worth should generally increase
            net_worth = initial_capital * (1 + (episode * 0.02) + np.random.normal(0, 0.05))
            net_worth = max(net_worth, initial_capital * 0.8)  # Don't lose more than 20%
            net_worths.append(net_worth)
        
        # Generate trading actions
        actions = [np.random.uniform(-0.5, 0.5) for _ in range(episodes)]
        
        # Calculate performance metrics
        final_return = (net_worths[-1] - initial_capital) / initial_capital
        returns = [(nw - initial_capital) / initial_capital for nw in net_worths]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
        
        return {
            'episode_rewards': episode_rewards,
            'net_worths': net_worths,
            'actions': actions,
            'final_summary': {
                'total_return': round(final_return * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(np.random.uniform(5, 15), 2),
                'win_rate': round(np.random.uniform(0.45, 0.65), 3),
                'final_net_worth': round(net_worths[-1], 2)
            },
            'agent': 'PPO_sample',
            'environment': 'sample_env'
        }


def create_complete_sample_analysis(ticker: str = "SAMPLE_STOCK") -> Dict[str, Any]:
    """
    Create a complete sample analysis with all components.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with complete analysis results
    """
    generator = SampleDataGenerator()
    
    # Generate sample data
    stock_data = generator.generate_stock_data(ticker=ticker, days=250)
    current_price = stock_data['Close'].iloc[-1]
    
    # Generate all analysis components
    sentiment_results = generator.generate_sentiment_data()
    forecast_results = generator.generate_forecast_results(current_price=current_price)
    trading_results = generator.generate_trading_results()
    
    return {
        'ticker': ticker,
        'data_summary': {
            'days_analyzed': len(stock_data),
            'date_range': f"{stock_data['Date'].iloc[0].strftime('%Y-%m-%d')} to {stock_data['Date'].iloc[-1].strftime('%Y-%m-%d')}",
            'current_price': round(current_price, 2),
            'price_change_1d': round(stock_data['Close'].pct_change().iloc[-1] * 100, 2),
            'volatility': round(stock_data['Close'].pct_change().std() * np.sqrt(252) * 100, 2)
        },
        'forecast_analysis': forecast_results,
        'sentiment_analysis': sentiment_results,
        'trading_analysis': trading_results,
        'raw_data': stock_data,
        'data_source': 'sample_generator'
    }
