"""
M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts Model
for Extreme-Adaptive Time Series Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.multi_resolution import MultiResolutionLayer
from ..layers.multi_view import MultiViewLayer
from ..layers.frequency_moe import FrequencyMoE


class M2FMoE(nn.Module):
    """
    M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts
    
    This model combines:
    1. Multi-Resolution processing for capturing patterns at different temporal scales
    2. Multi-View analysis for different perspectives of time series
    3. Frequency domain Mixture-of-Experts for adaptive frequency component modeling
    
    Args:
        input_size (int): Number of input features
        d_model (int): Model dimension
        num_layers (int): Number of M2FMoE layers
        num_resolutions (int): Number of resolution levels in multi-resolution layer
        num_views (int): Number of views in multi-view layer
        num_experts (int): Number of experts in MoE
        top_k (int): Number of experts to activate
        pred_len (int): Prediction length
        seq_len (int): Input sequence length
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_size=1,
        d_model=512,
        num_layers=3,
        num_resolutions=3,
        num_views=3,
        num_experts=8,
        top_k=2,
        pred_len=96,
        seq_len=96,
        dropout=0.1
    ):
        super(M2FMoE, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.seq_len = seq_len
        
        # Input embedding
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # M2FMoE layers
        self.layers = nn.ModuleList([
            M2FMoEBlock(
                d_model=d_model,
                num_resolutions=num_resolutions,
                num_views=num_views,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_size)
        )
        
        # Prediction head
        self.prediction_head = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        """
        Forward pass of M2FMoE model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, pred_len, input_size)
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply M2FMoE blocks
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        x = self.output_projection(x)  # (batch_size, seq_len, input_size)
        
        # Prediction
        # Transpose to apply linear on sequence length dimension
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.prediction_head(x)  # (batch_size, input_size, pred_len)
        x = x.transpose(1, 2)  # (batch_size, pred_len, input_size)
        
        return x


class M2FMoEBlock(nn.Module):
    """
    Single M2FMoE block combining all three components.
    
    Args:
        d_model (int): Model dimension
        num_resolutions (int): Number of resolution levels
        num_views (int): Number of views
        num_experts (int): Number of experts
        top_k (int): Number of experts to activate
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        d_model,
        num_resolutions=3,
        num_views=3,
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super(M2FMoEBlock, self).__init__()
        
        # Multi-Resolution Layer
        self.multi_resolution = MultiResolutionLayer(
            d_model=d_model,
            num_resolutions=num_resolutions
        )
        
        # Multi-View Layer
        self.multi_view = MultiViewLayer(
            d_model=d_model,
            num_views=num_views,
            dropout=dropout
        )
        
        # Frequency MoE Layer
        self.freq_moe = FrequencyMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of M2FMoE block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-Resolution processing
        x = self.norm1(self.multi_resolution(x) + x)
        
        # Multi-View processing
        x = self.norm2(self.multi_view(x) + x)
        
        # Frequency MoE processing
        x = self.norm3(self.freq_moe(x) + x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-like architectures.
    
    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
