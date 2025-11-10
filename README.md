# M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official PyTorch implementation of **M²FMoE** accepted at **AAAI 2026**.

## 📋 Overview

M²FMoE is a novel deep learning architecture designed for extreme-adaptive time series forecasting. It combines three key innovations:

1. **Multi-Resolution Processing**: Captures temporal patterns at different scales using multi-scale convolutions
2. **Multi-View Analysis**: Analyzes time series from temporal, statistical, and trend perspectives
3. **Frequency Mixture-of-Experts**: Applies adaptive expert selection in the frequency domain for enhanced modeling capacity

## 🏗️ Architecture

```
Input Time Series
      ↓
[Input Projection + Positional Encoding]
      ↓
[M²FMoE Block] × N layers
    ├── Multi-Resolution Layer
    ├── Multi-View Layer  
    └── Frequency MoE Layer
      ↓
[Output Projection + Prediction Head]
      ↓
Forecasted Time Series
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Yaohui-Huang/M2FMoE.git
cd M2FMoE

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom parameters
python scripts/train.py \
    --data_path data/your_data.npy \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 512 \
    --num_layers 3 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path data/test_data.npy \
    --output_dir results
```

## 📊 Usage Example

```python
import torch
from m2fmoe import M2FMoE

# Initialize model
model = M2FMoE(
    input_size=1,          # Number of input features
    d_model=512,           # Model dimension
    num_layers=3,          # Number of M²FMoE layers
    num_resolutions=3,     # Number of resolution levels
    num_views=3,           # Number of views
    num_experts=8,         # Number of experts in MoE
    top_k=2,               # Number of experts to activate
    pred_len=96,           # Prediction length
    seq_len=96,            # Input sequence length
    dropout=0.1            # Dropout rate
)

# Prepare input
x = torch.randn(32, 96, 1)  # (batch_size, seq_len, input_size)

# Forward pass
predictions = model(x)  # (batch_size, pred_len, input_size)
```

## 📁 Project Structure

```
M2FMoE/
├── src/
│   └── m2fmoe/
│       ├── models/
│       │   └── m2fmoe.py              # Main M²FMoE model
│       ├── layers/
│       │   ├── multi_resolution.py    # Multi-Resolution Layer
│       │   ├── multi_view.py          # Multi-View Layer
│       │   └── frequency_moe.py       # Frequency MoE Layer
│       └── utils/
│           ├── data_loader.py         # Data loading utilities
│           ├── metrics.py             # Evaluation metrics
│           └── visualization.py       # Visualization tools
├── scripts/
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluation script
├── configs/
│   └── default_config.json            # Default configuration
├── data/                              # Data directory
├── checkpoints/                       # Model checkpoints
├── results/                           # Evaluation results
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🔧 Key Components

### Multi-Resolution Layer
Processes time series at different temporal resolutions using depthwise convolutions with varying kernel sizes, enabling the capture of both short-term and long-term patterns.

### Multi-View Layer
Analyzes time series from three complementary perspectives:
- **Temporal View**: Sequential patterns via self-attention
- **Statistical View**: Statistical features via MLPs
- **Trend View**: Long-term trends via moving averages

### Frequency MoE Layer
Applies FFT to transform data to frequency domain, uses a gating network to select top-k experts adaptively, and transforms results back to time domain via inverse FFT.

## 📈 Performance

M²FMoE achieves state-of-the-art performance on multiple time series forecasting benchmarks:

| Dataset | MSE ↓ | MAE ↓ | 
|---------|-------|-------|
| ETTh1   | -     | -     |
| ETTm1   | -     | -     |
| Weather | -     | -     |
| Electricity | - | -     |

*Note: Full benchmark results will be updated upon paper publication.*

## 🔬 Ablation Study

Each component of M²FMoE contributes to the overall performance:

| Configuration | MSE ↓ |
|---------------|-------|
| Full Model    | -     |
| w/o Multi-Resolution | - |
| w/o Multi-View | - |
| w/o Frequency MoE | - |

## 💡 Key Features

- ✅ **Extreme Adaptability**: Handles diverse time series patterns
- ✅ **Multi-Scale Processing**: Captures patterns at different temporal scales
- ✅ **Frequency Domain Learning**: Leverages frequency representations
- ✅ **Sparse Activation**: Efficient expert selection via top-k routing
- ✅ **End-to-End Training**: Fully differentiable architecture
- ✅ **Flexible Configuration**: Easy to customize for different tasks

## 🎯 Supported Tasks

- Long-term time series forecasting
- Short-term time series forecasting
- Multivariate time series forecasting
- Univariate time series forecasting

## 📊 Datasets

The model has been evaluated on standard time series forecasting benchmarks:

- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: Weather forecasting dataset
- **Electricity**: Electricity consumption dataset
- **Traffic**: Traffic flow dataset
- **ILI (Influenza-Like Illness)**: Disease prediction dataset

*Data preprocessing scripts and dataset links will be provided.*

## 🛠️ Hyperparameters

Key hyperparameters and their recommended ranges:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| d_model | 512 | [256, 1024] | Model dimension |
| num_layers | 3 | [2, 6] | Number of M²FMoE blocks |
| num_experts | 8 | [4, 16] | Number of experts |
| top_k | 2 | [1, 4] | Active experts per token |
| learning_rate | 1e-4 | [1e-5, 1e-3] | Learning rate |
| batch_size | 32 | [16, 128] | Batch size |

## 📄 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{huang2026m2fmoe,
  title={M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting},
  author={Huang, Yaohui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please contact:
- **Yaohui Huang** - [GitHub](https://github.com/Yaohui-Huang)

## 🙏 Acknowledgments

We thank the reviewers and the time series forecasting community for their valuable feedback and support.

## 🔗 Related Work

- [Autoformer](https://github.com/thuml/Autoformer)
- [FEDformer](https://github.com/MAZiqing/FEDformer)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [Transformer](https://arxiv.org/abs/1706.03762)
- [Mixture-of-Experts](https://arxiv.org/abs/1701.06538)

---

⭐ If you find this repository helpful, please consider giving it a star!