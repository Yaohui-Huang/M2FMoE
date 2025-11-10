# M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

## Paper Information

- **Conference**: AAAI 2026 (38th AAAI Conference on Artificial Intelligence)
- **Status**: Accepted
- **Title**: M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting
- **Authors**: Yaohui Huang

## Abstract

Time series forecasting is a fundamental task in various domains, including finance, weather prediction, and energy management. However, existing methods often struggle with extreme adaptability—the ability to handle diverse patterns across different temporal scales and characteristics. We propose M²FMoE (Multi-Resolution Multi-View Frequency Mixture-of-Experts), a novel architecture that addresses these challenges through three key innovations:

1. **Multi-Resolution Processing**: Captures temporal patterns at different scales simultaneously
2. **Multi-View Analysis**: Analyzes time series from complementary perspectives (temporal, statistical, and trend)
3. **Frequency Mixture-of-Experts**: Applies adaptive expert selection in the frequency domain for enhanced modeling capacity

Our extensive experiments on multiple benchmark datasets demonstrate that M²FMoE achieves state-of-the-art performance while maintaining computational efficiency through sparse expert activation.

## Key Contributions

1. **Novel Architecture**: First work to combine multi-resolution, multi-view, and frequency-domain MoE for time series forecasting
2. **Extreme Adaptability**: Handles diverse time series patterns across different scales and characteristics
3. **Frequency Domain Learning**: Leverages FFT-based frequency representations for improved pattern recognition
4. **Sparse Activation**: Efficient expert selection mechanism reducing computational overhead
5. **State-of-the-Art Results**: Outperforms existing methods on multiple benchmark datasets

## Architecture Details

### 1. Multi-Resolution Layer

The Multi-Resolution Layer employs depthwise separable convolutions with varying kernel sizes to capture patterns at different temporal scales:

- **Short-term patterns**: Small kernel (e.g., 3)
- **Medium-term patterns**: Medium kernel (e.g., 5)
- **Long-term patterns**: Large kernel (e.g., 7)

Each resolution path processes the input independently, and the outputs are fused through a learned projection.

### 2. Multi-View Layer

The Multi-View Layer creates three complementary representations:

- **Temporal View**: Uses self-attention to capture sequential dependencies
- **Statistical View**: Employs MLPs to extract statistical features
- **Trend View**: Applies moving averages to identify long-term trends

These views are weighted and combined to provide a comprehensive understanding of the time series.

### 3. Frequency MoE Layer

The Frequency MoE Layer operates in the frequency domain:

1. **Forward FFT**: Transforms input to frequency domain
2. **Expert Routing**: Gating network selects top-k experts for each frequency component
3. **Expert Processing**: Selected experts process frequency components
4. **Inverse FFT**: Transforms result back to time domain

This design enables adaptive modeling of different frequency components, crucial for handling diverse time series patterns.

## Experimental Setup

### Datasets

We evaluate M²FMoE on standard time series forecasting benchmarks:

1. **ETT (Electricity Transformer Temperature)**
   - ETTh1, ETTh2: Hourly data
   - ETTm1, ETTm2: 15-minute data
   
2. **Weather**: Weather forecasting with 21 indicators

3. **Electricity**: Electricity consumption from 321 clients

4. **Traffic**: Road occupancy rates from 862 sensors

5. **ILI**: Influenza-like illness cases

### Baselines

We compare against state-of-the-art methods:
- Transformer variants (Informer, Autoformer, FEDformer)
- CNN-based methods (TCN, TimesNet)
- MLP-based methods (DLinear, NLinear)
- Hybrid methods (PatchTST, Crossformer)

### Metrics

- **MSE (Mean Squared Error)**: Primary metric
- **MAE (Mean Absolute Error)**: Secondary metric

## Results Summary

M²FMoE consistently achieves state-of-the-art performance across all datasets and forecasting horizons:

### Key Findings

1. **Multi-Resolution Benefits**: Improves performance by capturing both short and long-term patterns
2. **Multi-View Advantages**: Complementary views provide robust representations
3. **Frequency MoE Effectiveness**: Adaptive expert selection in frequency domain enhances modeling capacity
4. **Scalability**: Maintains efficiency through sparse expert activation

### Ablation Studies

Component-wise analysis demonstrates the importance of each module:

1. **Without Multi-Resolution**: Performance drops significantly on datasets with varying temporal scales
2. **Without Multi-View**: Loss of robustness to different time series characteristics
3. **Without Frequency MoE**: Reduced capacity for handling complex frequency patterns

## Computational Efficiency

Despite its sophisticated architecture, M²FMoE maintains computational efficiency:

- **Sparse Activation**: Only top-k experts activated per token
- **Efficient FFT**: Leverages fast Fourier transform implementations
- **Depthwise Convolutions**: Reduces parameters in multi-resolution layer

## Future Directions

1. **Long-term Forecasting**: Extend to longer prediction horizons
2. **Multimodal Learning**: Incorporate external information sources
3. **Online Learning**: Adapt to distribution shifts in real-time
4. **Uncertainty Quantification**: Provide confidence intervals for predictions
5. **Interpretability**: Enhance understanding of expert specialization

## Implementation Notes

The official implementation includes:

- Modular architecture for easy customization
- Comprehensive training and evaluation scripts
- Extensive documentation and examples
- Pre-configured hyperparameters for each dataset
- Visualization tools for analysis

## Reproducibility

To ensure reproducibility:

1. **Fixed Random Seeds**: All experiments use fixed random seeds
2. **Detailed Hyperparameters**: Complete configuration provided
3. **Standard Datasets**: Uses publicly available benchmarks
4. **Code Release**: Full implementation available on GitHub

## Citation

```bibtex
@inproceedings{huang2026m2fmoe,
  title={M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting},
  author={Huang, Yaohui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Contact

For questions about the paper or implementation:
- GitHub Issues: [https://github.com/Yaohui-Huang/M2FMoE/issues](https://github.com/Yaohui-Huang/M2FMoE/issues)
- Email: See paper for contact information

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback and suggestions that helped improve this work.
