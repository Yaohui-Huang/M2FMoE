"""
Multi-View Layer for analyzing time series from different perspectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewLayer(nn.Module):
    """
    Multi-View Layer that processes time series from different analytical perspectives.
    
    This layer creates multiple views of the input data:
    - Temporal view: Sequential patterns
    - Statistical view: Statistical features
    - Trend view: Long-term trends
    
    Args:
        d_model (int): Dimension of the model
        num_views (int): Number of different views
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model, num_views=3, dropout=0.1):
        super(MultiViewLayer, self).__init__()
        self.d_model = d_model
        self.num_views = num_views
        
        # Temporal view - attention-based
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Statistical view - MLP-based
        self.stat_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Trend view - moving average-based
        self.trend_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=25,
            padding=12,
            groups=d_model
        )
        
        # View fusion
        self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
        self.fusion = nn.Linear(d_model * num_views, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of Multi-View Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Temporal view
        temporal_out, _ = self.temporal_attn(x, x, x)
        
        # Statistical view
        stat_out = self.stat_encoder(x)
        
        # Trend view
        x_t = x.transpose(1, 2)
        trend_out = self.trend_conv(x_t).transpose(1, 2)
        
        # Weighted fusion of views
        views = [temporal_out, stat_out, trend_out]
        weighted_views = [
            view * self.view_weights[i].sigmoid()
            for i, view in enumerate(views)
        ]
        
        # Concatenate and fuse
        concat_views = torch.cat(weighted_views, dim=-1)
        output = self.fusion(concat_views)
        output = self.norm(output)
        output = self.dropout(output)
        
        # Residual connection
        output = output + x
        
        return output
