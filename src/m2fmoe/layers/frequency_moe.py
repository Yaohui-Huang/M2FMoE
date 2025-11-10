"""
Frequency Mixture-of-Experts Layer for frequency domain processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Expert(nn.Module):
    """
    Individual expert in the MoE architecture.
    
    Args:
        d_model (int): Dimension of the model
        d_ff (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Expert, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class FrequencyMoE(nn.Module):
    """
    Frequency Mixture-of-Experts Layer.
    
    This layer performs frequency domain transformation and applies
    a mixture of experts to different frequency components.
    
    Args:
        d_model (int): Dimension of the model
        num_experts (int): Number of experts
        top_k (int): Number of experts to activate per token
        d_ff (int): Dimension of feedforward network in experts
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None, dropout=0.1):
        super(FrequencyMoE, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # Router/Gating network
        self.router = nn.Linear(d_model, num_experts)
        
        # Frequency embedding
        self.freq_embed = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of Frequency MoE Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # Apply FFT to transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        freq_len = x_freq.shape[1]
        
        # Convert complex to real representation
        x_freq_real = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # (batch, freq_len, 2*d_model)
        freq_d = x_freq_real.shape[-1]
        
        # Create temporary projection layer if needed
        if not hasattr(self, 'freq_proj_in') or self.freq_proj_in.in_features != freq_d:
            self.freq_proj_in = nn.Linear(freq_d, d_model).to(x.device)
        if not hasattr(self, 'freq_proj_out') or self.freq_proj_out.out_features != freq_d:
            self.freq_proj_out = nn.Linear(d_model, freq_d).to(x.device)
        
        # Project to d_model
        x_freq_proj = self.freq_proj_in(x_freq_real)  # (batch, freq_len, d_model)
        
        # Router: compute gating weights
        router_logits = self.router(x_freq_proj)  # (batch_size, freq_len, num_experts)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (batch_size, freq_len, top_k)
        
        # Apply softmax to get routing weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_freq_proj)
        
        # Apply selected experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # (batch_size, freq_len)
            expert_weight = top_k_weights[:, :, i:i+1]  # (batch_size, freq_len, 1)
            
            # Process each expert
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id).unsqueeze(-1).float()
                if mask.sum() > 0:
                    expert_out = self.experts[expert_id](x_freq_proj)
                    output += expert_out * expert_weight * mask
        
        # Project back to frequency dimension
        output = self.freq_proj_out(output)  # (batch, freq_len, freq_d)
        
        # Inverse FFT to transform back to time domain
        output_complex = torch.complex(
            output[:, :, :d_model],
            output[:, :, d_model:]
        )
        
        output_time = torch.fft.irfft(output_complex, n=seq_len, dim=1, norm='ortho')
        
        output_time = self.norm(output_time)
        
        # Residual connection
        output_time = output_time + residual
        
        return output_time
