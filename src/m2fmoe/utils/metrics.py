"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
import torch


def mse(pred, true):
    """
    Mean Squared Error.
    
    Args:
        pred: Predictions
        true: Ground truth
        
    Returns:
        MSE value
    """
    return np.mean((pred - true) ** 2)


def mae(pred, true):
    """
    Mean Absolute Error.
    
    Args:
        pred: Predictions
        true: Ground truth
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    """
    Root Mean Squared Error.
    
    Args:
        pred: Predictions
        true: Ground truth
        
    Returns:
        RMSE value
    """
    return np.sqrt(mse(pred, true))


def mape(pred, true, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    
    Args:
        pred: Predictions
        true: Ground truth
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (in percentage)
    """
    return np.mean(np.abs((true - pred) / (true + epsilon))) * 100


def smape(pred, true, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error.
    
    Args:
        pred: Predictions
        true: Ground truth
        epsilon: Small value to avoid division by zero
        
    Returns:
        SMAPE value (in percentage)
    """
    numerator = np.abs(pred - true)
    denominator = (np.abs(pred) + np.abs(true)) / 2 + epsilon
    return np.mean(numerator / denominator) * 100


def r2_score(pred, true):
    """
    R-squared (coefficient of determination).
    
    Args:
        pred: Predictions
        true: Ground truth
        
    Returns:
        R2 score
    """
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def calculate_metrics(pred, true):
    """
    Calculate all metrics.
    
    Args:
        pred: Predictions (numpy array or torch tensor)
        true: Ground truth (numpy array or torch tensor)
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    metrics = {
        'MSE': mse(pred, true),
        'MAE': mae(pred, true),
        'RMSE': rmse(pred, true),
        'MAPE': mape(pred, true),
        'SMAPE': smape(pred, true),
        'R2': r2_score(pred, true)
    }
    
    return metrics


def print_metrics(metrics, prefix=''):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the print statement
    """
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"{name:10s}: {value:.6f}")
    print("-" * 50)
