"""
Multi-Resolution Layer for capturing temporal patterns at different scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionLayer(nn.Module):
    """
    Multi-Resolution Layer that processes time series data at different temporal resolutions.
    
    This layer applies multi-scale convolutions to capture patterns at various time scales,
    which is crucial for extreme-adaptive time series forecasting.
    
    Args:
        d_model (int): Dimension of the model
        num_resolutions (int): Number of different resolution levels
        kernel_sizes (list): List of kernel sizes for different resolutions
    """
    
    def __init__(self, d_model, num_resolutions=3, kernel_sizes=None):
        super(MultiResolutionLayer, self).__init__()
        self.d_model = d_model
        self.num_resolutions = num_resolutions
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]  # Default kernel sizes
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=k,
                padding=k // 2,
                groups=d_model  # Depthwise convolution
            )
            for k in kernel_sizes[:num_resolutions]
        ])
        
        # Projection layers for each resolution
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_resolutions)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * num_resolutions, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of Multi-Resolution Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Transpose for Conv1d: (batch, channels, length)
        x_t = x.transpose(1, 2)
        
        # Process at different resolutions
        multi_res_outputs = []
        for conv, proj in zip(self.convs, self.projections):
            # Apply convolution
            out = conv(x_t)
            # Transpose back
            out = out.transpose(1, 2)
            # Apply projection
            out = proj(out)
            multi_res_outputs.append(out)
        
        # Concatenate all resolution outputs
        concat_out = torch.cat(multi_res_outputs, dim=-1)
        
        # Fuse multi-resolution features
        output = self.fusion(concat_out)
        output = self.norm(output)
        
        # Residual connection
        output = output + x
        
        return output
