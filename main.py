"""
Main integration script for unified stock market analysis.
This script combines forecasting, sentiment analysis, and reinforcement learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.data_processor import StockDataProcessor, load_and_process_stock_data
from src.models.forecasting import StockForecaster, forecast_stock_prices
from src.models.sentiment_analyzer import FinancialSentimentAnalyzer
try:
    from src.models.rl_trading import StockTradingEnv, PPOAgent
    RL_AVAILABLE = True
except ImportError:
    print("Warning: RL modules not available. Some features may be limited.")
    RL_AVAILABLE = False
from src.utils.visualization import (plot_stock_price_analysis, plot_trading_performance, 
                                   plot_training_progress, create_performance_report,
                                   calculate_performance_metrics, print_summary_table)


class UnifiedStockAnalyzer:
    """
    Unified stock market analyzer combining multiple AI techniques.
    """
    
    def __init__(self, ticker: str, initial_capital: float = 10000):
        self.ticker = ticker
        self.initial_capital = initial_capital
        
        # Initialize components
        self.data_processor = StockDataProcessor()
        self.forecaster = StockForecaster()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.rolling_data = None
        self.sentiment_data = None
        self.forecast_results = {}
        self.trading_results = {}
        
        print(f"Initialized Unified Stock Analyzer for {ticker}")
    
    def load_and_prepare_data(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Load and prepare all necessary data.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            period: Period if dates not specified
            
        Returns:
            Dictionary with prepared datasets
        """
        print("Loading and preparing stock data...")
        
        # Load raw stock data
        self.raw_data = self.data_processor.load_stock_data(
            self.ticker, start_date, end_date, period
        )
        
        # Add technical indicators
        self.processed_data = self.data_processor.calculate_technical_indicators(self.raw_data)
        
        # Create rolling window analysis
        self.rolling_data = self.data_processor.create_rolling_windows(self.raw_data)
        
        print(f"Loaded {len(self.raw_data)} days of data for {self.ticker}")
        print(f"Created {len(self.rolling_data)} rolling windows for analysis")
        
        return {
            'raw': self.raw_data,
            'processed': self.processed_data,
            'rolling': self.rolling_data
        }
    
    def run_forecasting_analysis(self, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive forecasting analysis.
        
        Args:
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        print(f"Running forecasting analysis for {forecast_days} days...")
        
        if self.raw_data is None:
            raise ValueError("Must load data first using load_and_prepare_data()")
        
        # Run forecasting
        self.forecast_results = forecast_stock_prices(
            self.ticker, forecast_days, models=['arima', 'sarima']
        )
        
        # Print forecast summary
        if 'forecasts' in self.forecast_results:
            for model_name, result in self.forecast_results['forecasts'].items():
                if 'aic' in result:
                    print(f"{model_name.upper()}: AIC = {result['aic']:.2f}")
        
        print("Forecasting analysis completed")
        return self.forecast_results
    
    def run_sentiment_analysis(self, news_texts: Optional[list] = None) -> pd.DataFrame:
        """
        Run sentiment analysis on financial news/social media.
        
        Args:
            news_texts: Optional list of texts to analyze
            
        Returns:
            DataFrame with sentiment results
        """
        print("Running sentiment analysis...")
        
        if news_texts is None:
            # Generate sample financial texts for demonstration
            sample_texts = [
                f"{self.ticker} stock shows strong performance this quarter",
                f"Analysts bullish on {self.ticker} future prospects",
                f"Market volatility affects {self.ticker} trading",
                f"{self.ticker} reports better than expected earnings",
                f"Concerns over {self.ticker} recent market movements"
            ]
            news_texts = sample_texts
            print("Using sample texts for sentiment analysis demonstration")
        
        # Analyze sentiment
        self.sentiment_data = self.sentiment_analyzer.analyze_sentiment_batch(news_texts)
        
        # Calculate average sentiment for RL environment
        avg_sentiment = self.sentiment_data['sentiment_score'].mean()
        
        print(f"Analyzed {len(news_texts)} texts")
        print(f"Average sentiment score: {avg_sentiment:.3f}")
        
        return self.sentiment_data
    
    def prepare_rl_environment(self) -> Any:
        """
        Prepare reinforcement learning environment.
        
        Returns:
            Configured trading environment or None if RL not available
        """
        if not RL_AVAILABLE:
            print("RL environment not available - skipping")
            return None
            
        print("Preparing RL trading environment...")
        
        if self.rolling_data is None or len(self.rolling_data) == 0:
            print("No rolling window data available, using processed stock data directly...")
            if self.processed_data is None:
                raise ValueError("Must load data first using load_and_prepare_data()")
            rl_data = self.data_processor.preprocess_for_rl(self.processed_data)
        else:
            rl_data = self.data_processor.preprocess_for_rl(self.rolling_data)
        
        # Add sentiment if available
        if self.sentiment_data is not None:
            avg_sentiment = self.sentiment_data['sentiment_score'].mean()
            rl_data['Sentiment_Score'] = avg_sentiment
        
        # Ensure minimum data requirements
        if len(rl_data) < 50:
            print(f"Insufficient data for RL training ({len(rl_data)} points). Need at least 50.")
            return None
        
        # Create environment
        env = StockTradingEnv(
            data=rl_data,
            initial_capital=self.initial_capital,
            episode_length=min(200, len(rl_data) - 1),
            trend_reward_coef=0.02,
            sentiment_reward_coef=0.01
        )
        
        print(f"Created RL environment with {len(rl_data)} data points")
        return env
    
    def train_rl_agent(self, episodes: int = 100, save_model: bool = True) -> Dict[str, Any]:
        """
        Train reinforcement learning agent.
        
        Args:
            episodes: Number of training episodes
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training results
        """
        if not RL_AVAILABLE:
            print("RL training not available - returning dummy results")
            return {
                'episode_rewards': [0] * episodes,
                'net_worths': [self.initial_capital] * episodes,
                'actions': [0] * episodes,
                'final_summary': {'total_return': 0, 'sharpe_ratio': 0},
                'agent': None,
                'environment': None
            }
            
        print(f"Training RL agent for {episodes} episodes...")
        
        # Prepare environment
        env = self.prepare_rl_environment()
        if env is None:
            return self.train_rl_agent(episodes, save_model)  # Return dummy results
        
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        
        # Training variables
        episode_rewards = []
        net_worths = []
        all_actions = []
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
        
        # Training loop
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_actions = []
            
            while True:
                action, log_prob = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['log_probs'].append(log_prob)
                memory['rewards'].append(reward)
                
                episode_actions.append(float(action))
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update agent
            if len(memory['states']) > 0:
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
            
            episode_rewards.append(episode_reward)
            net_worths.append(info.get('net_worth', env.net_worth))
            all_actions.extend(episode_actions)
            
            # Progress reporting
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_networth = np.mean(net_worths[-20:])
                print(f"Episode {episode+1}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Net Worth: {avg_networth:.2f}")
        
        # Save model if requested
        if save_model and agent is not None:
            model_path = f"results/{self.ticker}_rl_model.pth"
            os.makedirs("results", exist_ok=True)
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")
        
        # Get final trading summary
        trading_summary = env.get_trade_summary() if env else {}
        
        self.trading_results = {
            'episode_rewards': episode_rewards,
            'net_worths': net_worths,
            'actions': all_actions,
            'final_summary': trading_summary,
            'agent': agent,
            'environment': env
        }
        
        print("RL agent training completed")
        return self.trading_results
    
    def run_complete_analysis(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            forecast_days: int = 30,
                            rl_episodes: int = 100,
                            news_texts: Optional[list] = None) -> Dict[str, Any]:
        """
        Run complete unified analysis.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            forecast_days: Days to forecast
            rl_episodes: RL training episodes
            news_texts: Optional news texts for sentiment
            
        Returns:
            Complete analysis results
        """
        print(f"\\n{'='*60}")
        print(f"UNIFIED STOCK MARKET ANALYSIS FOR {self.ticker}")
        print(f"{'='*60}")
        
        # Step 1: Load and prepare data
        data_results = self.load_and_prepare_data(start_date, end_date)
        
        # Step 2: Forecasting analysis
        forecast_results = self.run_forecasting_analysis(forecast_days)
        
        # Step 3: Sentiment analysis
        sentiment_results = self.run_sentiment_analysis(news_texts)
        
        # Step 4: RL training
        trading_results = self.train_rl_agent(rl_episodes)
        
        # Compile complete results
        complete_results = {
            'ticker': self.ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': data_results,
            'forecasting': forecast_results,
            'sentiment': sentiment_results,
            'trading': trading_results,
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        # Generate visualizations
        self._create_visualizations()
        
        # Create performance report
        report = create_performance_report(complete_results, f"results/{self.ticker}_analysis_report.csv")
        
        print(f"\\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return complete_results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Trading performance
        if self.trading_results and 'final_summary' in self.trading_results:
            summary = self.trading_results['final_summary']
            metrics.update(summary)
        
        # Data metrics
        if self.raw_data is not None:
            price_data = self.raw_data['Close']
            returns = price_data.pct_change().dropna()
            
            stock_metrics = calculate_performance_metrics(returns)
            for key, value in stock_metrics.items():
                metrics[f'Stock_{key}'] = value
        
        return metrics
    
    def _create_visualizations(self):
        """Create all visualizations."""
        print("\\nGenerating visualizations...")
        
        # Stock price analysis
        if self.raw_data is not None:
            plot_stock_price_analysis(self.raw_data, self.ticker)
        
        # Trading performance
        if self.trading_results and 'environment' in self.trading_results:
            env = self.trading_results['environment']
            trade_history = env.trade_history
            if trade_history:
                plot_trading_performance(trade_history, f"{self.ticker} Trading Performance")
        
        # Training progress
        if self.trading_results:
            episode_rewards = self.trading_results.get('episode_rewards', [])
            net_worths = self.trading_results.get('net_worths', [])
            if episode_rewards and net_worths:
                plot_training_progress(episode_rewards, net_worths)
        
        print("Visualizations completed")


def main():
    """Main execution function."""
    # Configuration
    TICKER = "MSFT"  # Microsoft as example
    INITIAL_CAPITAL = 10000
    FORECAST_DAYS = 30
    RL_EPISODES = 50  # Reduced for demo
    
    # Optional: provide custom date range
    # START_DATE = "2023-01-01"
    # END_DATE = "2024-01-01"
    START_DATE = None
    END_DATE = None
    
    # Create analyzer
    analyzer = UnifiedStockAnalyzer(TICKER, INITIAL_CAPITAL)
    
    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis(
            start_date=START_DATE,
            end_date=END_DATE,
            forecast_days=FORECAST_DAYS,
            rl_episodes=RL_EPISODES
        )
        
        # Print summary
        if 'performance_metrics' in results:
            print_summary_table(results['performance_metrics'], f"{TICKER} Performance Summary")
        
        print(f"\\nComplete analysis saved to results/{TICKER}_analysis_report.csv")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
