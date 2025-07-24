"""
Utility functions for visualization and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


def plot_stock_price_analysis(data: pd.DataFrame, ticker: str = "Stock"):
    """
    Plot comprehensive stock price analysis.
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{ticker} Stock Analysis', fontsize=16, fontweight='bold')
    
    # Price and volume
    ax1 = axes[0, 0]
    if 'Date' in data.columns:
        dates = pd.to_datetime(data['Date'])
        ax1.plot(dates, data['Close'], color='blue', linewidth=2)
        ax1.set_title('Stock Price Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # Volume
    ax2 = axes[0, 1]
    if 'Date' in data.columns and 'Volume' in data.columns:
        ax2.bar(dates, data['Volume'], color='orange', alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    # Price distribution
    ax3 = axes[1, 0]
    if 'Close' in data.columns:
        ax3.hist(data['Close'], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax3.set_title('Price Distribution')
        ax3.set_xlabel('Price ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    
    # Returns distribution
    ax4 = axes[1, 1]
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
        ax4.hist(returns, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax4.set_title('Daily Returns Distribution')
        ax4.set_xlabel('Returns')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_trading_performance(trade_history: List[Dict], title: str = "Trading Performance"):
    """
    Plot trading performance metrics.
    
    Args:
        trade_history: List of trade records
        title: Plot title
    """
    if not trade_history:
        print("No trade history to plot")
        return
    
    df = pd.DataFrame(trade_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Portfolio value over time
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['total_assets'], color='green', linewidth=2)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Total Assets ($)')
    ax1.grid(True, alpha=0.3)
    
    # Position over time
    ax2 = axes[0, 1]
    ax2.plot(df['step'], df['position'], color='blue', linewidth=2)
    ax2.set_title('Position Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # Actions distribution
    ax3 = axes[1, 0]
    ax3.hist(df['action'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_title('Action Distribution')
    ax3.set_xlabel('Action Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Rewards over time
    ax4 = axes[1, 1]
    ax4.plot(df['step'], df['reward'], color='red', alpha=0.7)
    ax4.set_title('Rewards Over Time')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Reward')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_progress(episode_rewards: List[float], net_worths: List[float]):
    """
    Plot training progress over episodes.
    
    Args:
        episode_rewards: List of episode rewards
        net_worths: List of final net worths per episode
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Episode rewards
    ax1 = axes[0]
    episodes = range(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, color='blue', alpha=0.7)
    
    # Moving average
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards) // 4)
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'MA({window})')
        ax1.legend()
    
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Net worth progression
    ax2 = axes[1]
    ax2.plot(episodes, net_worths, color='green', alpha=0.7)
    
    # Moving average
    if len(net_worths) > 10:
        window = min(50, len(net_worths) // 4)
        moving_avg = pd.Series(net_worths).rolling(window=window).mean()
        ax2.plot(episodes, moving_avg, color='red', linewidth=2, label=f'MA({window})')
        ax2.legend()
    
    ax2.set_title('Net Worth Progression')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Net Worth ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(historical: pd.Series, forecasts: Dict[str, pd.Series],
                           conf_intervals: Optional[Dict[str, pd.DataFrame]] = None,
                           title: str = "Forecast Comparison"):
    """
    Plot comparison of different forecasting models.
    
    Args:
        historical: Historical price data
        forecasts: Dictionary of forecasts by model name
        conf_intervals: Optional confidence intervals
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical.index, historical.values, label="Historical", color="black", linewidth=2)
    
    # Plot forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
        # Create forecast index
        last_date = historical.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=len(forecast)+1, freq='D')[1:]
        
        plt.plot(forecast_index, forecast, label=f"{model_name} Forecast", 
                color=color, linestyle='--', linewidth=2)
        
        # Plot confidence intervals if available
        if conf_intervals and model_name in conf_intervals:
            conf_int = conf_intervals[model_name]
            plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                           color=color, alpha=0.2, label=f"{model_name} 95% CI")
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sentiment_analysis(sentiment_data: pd.DataFrame):
    """
    Plot sentiment analysis results.
    
    Args:
        sentiment_data: DataFrame with sentiment analysis results
    """
    if sentiment_data.empty:
        print("No sentiment data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
    
    # Sentiment distribution
    ax1 = axes[0, 0]
    if 'sentiment' in sentiment_data.columns:
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        pie_colors = [colors.get(label, 'gray') for label in sentiment_counts.index]
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=pie_colors)
        ax1.set_title('Sentiment Distribution')
    
    # Confidence distribution
    ax2 = axes[0, 1]
    if 'confidence' in sentiment_data.columns:
        ax2.hist(sentiment_data['confidence'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    
    # Sentiment score over time
    ax3 = axes[1, 0]
    if 'sentiment_score' in sentiment_data.columns and len(sentiment_data) > 1:
        ax3.plot(range(len(sentiment_data)), sentiment_data['sentiment_score'], 
                color='purple', alpha=0.7)
        ax3.set_title('Sentiment Score Over Time')
        ax3.set_xlabel('Text Index')
        ax3.set_ylabel('Sentiment Score')
        ax3.grid(True, alpha=0.3)
    
    # Financial relevance
    ax4 = axes[1, 1]
    if 'is_financial' in sentiment_data.columns:
        financial_counts = sentiment_data['is_financial'].value_counts()
        ax4.bar(['Financial', 'Non-Financial'], 
               [financial_counts.get(True, 0), financial_counts.get(False, 0)],
               color=['green', 'red'], alpha=0.7)
        ax4.set_title('Financial Relevance')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}
    
    returns_clean = returns.dropna()
    
    # Convert to float if needed
    std_val = float(returns_clean.std()) if hasattr(returns_clean.std(), 'item') else returns_clean.std()
    mean_val = float(returns_clean.mean()) if hasattr(returns_clean.mean(), 'item') else returns_clean.mean()
    
    # Calculate max drawdown
    drawdown = (returns_clean.cumsum() - returns_clean.cumsum().cummax()).min()
    max_drawdown_val = float(drawdown) if hasattr(drawdown, 'item') else drawdown
    
    metrics = {
        'Total Return': (1 + returns_clean).prod() - 1,
        'Annualized Return': mean_val * 252,
        'Volatility': std_val * np.sqrt(252),
        'Sharpe Ratio': (mean_val / std_val) * np.sqrt(252) if std_val > 0 else 0,
        'Max Drawdown': max_drawdown_val,
        'Calmar Ratio': (mean_val * 252) / abs(max_drawdown_val) if max_drawdown_val != 0 else 0
    }
    
    # Win rate
    positive_returns = returns_clean[returns_clean > 0]
    metrics['Win Rate'] = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns_clean[returns_clean < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    metrics['Sortino Ratio'] = (returns_clean.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # Beta and alpha if benchmark provided
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_clean = benchmark_returns.dropna()
        
        # Align returns
        min_len = min(len(returns_clean), len(benchmark_clean))
        returns_aligned = returns_clean.iloc[:min_len]
        benchmark_aligned = benchmark_clean.iloc[:min_len]
        
        if len(returns_aligned) > 1 and benchmark_aligned.var() > 0:
            covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = benchmark_aligned.var()
            beta = covariance / benchmark_variance
            alpha = returns_aligned.mean() - beta * benchmark_aligned.mean()
            
            metrics['Beta'] = beta
            metrics['Alpha'] = alpha * 252  # Annualized
            metrics['Information Ratio'] = (returns_aligned - benchmark_aligned).mean() / (returns_aligned - benchmark_aligned).std() * np.sqrt(252) if (returns_aligned - benchmark_aligned).std() > 0 else 0
    
    return metrics


def create_performance_report(results: Dict[str, Any], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comprehensive performance report.
    
    Args:
        results: Dictionary with all analysis results
        save_path: Optional path to save the report
        
    Returns:
        DataFrame with performance summary
    """
    report_data = []
    
    # Basic information
    report_data.append(['Analysis Date', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    if 'ticker' in results:
        report_data.append(['Ticker', results['ticker']])
    
    # Data summary
    if 'data' in results:
        data = results['data']
        report_data.append(['Data Points', len(data)])
        if 'Date' in data.columns:
            report_data.append(['Date Range', f"{data['Date'].min()} to {data['Date'].max()}"])
    
    # Forecasting results
    if 'forecasts' in results:
        forecasts = results['forecasts']
        for model_name, forecast_result in forecasts.items():
            if 'aic' in forecast_result:
                report_data.append([f'{model_name.upper()} AIC', f"{forecast_result['aic']:.2f}"])
            if 'bic' in forecast_result:
                report_data.append([f'{model_name.upper()} BIC', f"{forecast_result['bic']:.2f}"])
    
    # Trading performance
    if 'trading_summary' in results:
        summary = results['trading_summary']
        for key, value in summary.items():
            if isinstance(value, float):
                report_data.append([key.replace('_', ' ').title(), f"{value:.4f}"])
            else:
                report_data.append([key.replace('_', ' ').title(), str(value)])
    
    # Performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                report_data.append([key, f"{value:.4f}"])
            else:
                report_data.append([key, str(value)])
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data, columns=['Metric', 'Value'])
    
    # Save if path provided
    if save_path:
        report_df.to_csv(save_path, index=False)
        print(f"Report saved to {save_path}")
    
    return report_df


def print_summary_table(data: Dict[str, Any], title: str = "Analysis Summary"):
    """
    Print a formatted summary table.
    
    Args:
        data: Dictionary with summary data
        title: Table title
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    for key, value in data.items():
        if isinstance(value, float):
            print(f"{key:<30}: {value:>15.4f}")
        elif isinstance(value, (int, str)):
            print(f"{key:<30}: {value:>15}")
        elif isinstance(value, bool):
            print(f"{key:<30}: {str(value):>15}")
    
    print(f"{'='*50}\n")
