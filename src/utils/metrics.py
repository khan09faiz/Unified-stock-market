"""
Performance metrics calculation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union


def calculate_metrics(predictions: Union[List, np.ndarray], 
                     actual: Union[List, np.ndarray]) -> Dict[str, float]:
    """
    Calculate various performance metrics.
    
    Args:
        predictions: Predicted values
        actual: Actual values
        
    Returns:
        Dictionary with performance metrics
    """
    if len(predictions) == 0 or len(actual) == 0:
        return {
            'mape': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'r2_score': 0.0
        }
    
    pred = np.array(predictions)
    act = np.array(actual)
    
    # Ensure same length
    min_len = min(len(pred), len(act))
    pred = pred[:min_len]
    act = act[:min_len]
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((act - pred) / (act + 1e-8))) * 100
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((act - pred) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(act - pred))
    
    # R-squared Score
    ss_res = np.sum((act - pred) ** 2)
    ss_tot = np.sum((act - np.mean(act)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mape': float(mape),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2_score)
    }


def calculate_trading_metrics(returns: List[float], 
                            benchmark_returns: List[float] = None) -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    
    Args:
        returns: List of portfolio returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary with trading metrics
    """
    if not returns:
        return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}
    
    returns_array = np.array(returns)
    
    # Total return
    total_return = (np.prod(1 + returns_array) - 1) * 100
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(abs(max_drawdown))
    }


def calculate_forecast_accuracy(forecast: List[float], 
                              actual: List[float]) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        forecast: Forecasted values
        actual: Actual values
        
    Returns:
        Dictionary with accuracy metrics
    """
    return calculate_metrics(forecast, actual)
