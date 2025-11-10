"""
M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts
for Extreme-Adaptive Time Series Forecasting

AAAI 2026
"""

__version__ = "1.0.0"
__author__ = "Yaohui Huang"

from .models.m2fmoe import M2FMoE
from .layers.multi_resolution import MultiResolutionLayer
from .layers.multi_view import MultiViewLayer
from .layers.frequency_moe import FrequencyMoE

__all__ = [
    "M2FMoE",
    "MultiResolutionLayer",
    "MultiViewLayer",
    "FrequencyMoE",
]
