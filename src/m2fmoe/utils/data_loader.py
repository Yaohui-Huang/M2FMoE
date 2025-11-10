"""
Data loading utilities for time series forecasting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset for forecasting tasks.
    
    Args:
        data (np.ndarray): Time series data of shape (num_samples, num_features)
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction sequence
        stride (int): Stride for sliding window
    """
    
    def __init__(self, data, seq_len=96, pred_len=96, stride=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        # Create sliding windows
        self.indices = []
        for i in range(0, len(data) - seq_len - pred_len + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len
        
        # Input sequence
        x = self.data[start_idx:end_idx]
        # Target sequence
        y = self.data[end_idx:pred_end_idx]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_dataloader(
    data,
    seq_len=96,
    pred_len=96,
    batch_size=32,
    stride=1,
    shuffle=True,
    num_workers=0
):
    """
    Create a DataLoader for time series data.
    
    Args:
        data (np.ndarray): Time series data
        seq_len (int): Length of input sequence
        pred_len (int): Length of prediction sequence
        batch_size (int): Batch size
        stride (int): Stride for sliding window
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader object
    """
    dataset = TimeSeriesDataset(
        data=data,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def normalize_data(data, method='zscore'):
    """
    Normalize time series data.
    
    Args:
        data (np.ndarray): Time series data
        method (str): Normalization method ('zscore' or 'minmax')
        
    Returns:
        Normalized data and normalization parameters
    """
    if method == 'zscore':
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True) + 1e-8
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data, params, method='zscore'):
    """
    Denormalize time series data.
    
    Args:
        data (np.ndarray): Normalized time series data
        params (dict): Normalization parameters
        method (str): Normalization method ('zscore' or 'minmax')
        
    Returns:
        Denormalized data
    """
    if method == 'zscore':
        denormalized = data * params['std'] + params['mean']
    elif method == 'minmax':
        denormalized = data * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return denormalized
