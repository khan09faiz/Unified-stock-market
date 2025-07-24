"""
Configuration settings for the Unified Stock Market Analysis System.
"""

# Default stock ticker
DEFAULT_TICKER = "MSFT"

# Trading parameters
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.002  # 0.2%
STOP_LOSS_PCT = 0.02      # 2%

# Forecasting parameters
FORECAST_DAYS = 30
ARIMA_ORDER = (1, 1, 1)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
GARCH_P = 1
GARCH_Q = 1

# Reinforcement Learning parameters
RL_EPISODES = 100
RL_EPISODE_LENGTH = 300
LEARNING_RATE = 3e-4
VALUE_LEARNING_RATE = 1e-3
GAMMA = 0.99
EPS_CLIP = 0.2
TREND_REWARD_COEF = 0.02
SENTIMENT_REWARD_COEF = 0.01

# Sentiment Analysis parameters
CONFIDENCE_THRESHOLD = 0.6
FINANCIAL_RELEVANCE_THRESHOLD = 0.5

# Data parameters
DEFAULT_PERIOD = "1y"
ROLLING_WINDOW_SIZE = 30

# Visualization parameters
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = 'default'

# File paths
RESULTS_DIR = "results"
MODELS_DIR = "models"
DATA_DIR = "data"

# Financial lexicon for sentiment analysis
FINANCIAL_LEXICON = {
    'bullish': 'very_positive',
    'bearish': 'very_negative', 
    'short': 'negative_position',
    'long': 'positive_position',
    'pump': 'artificial_positive',
    'dump': 'artificial_negative'
}
