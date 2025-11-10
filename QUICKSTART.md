# M²FMoE Quick Start Guide

This guide will help you get started with M²FMoE in just a few minutes.

## Installation

### Option 1: Install from source
```bash
git clone https://github.com/Yaohui-Huang/M2FMoE.git
cd M2FMoE
pip install -r requirements.txt
```

### Option 2: Install as package
```bash
git clone https://github.com/Yaohui-Huang/M2FMoE.git
cd M2FMoE
pip install -e .
```

## Quick Demo

Run the demo script to see M²FMoE in action:

```bash
python scripts/demo.py
```

This will:
1. Generate synthetic time series data
2. Initialize the M²FMoE model
3. Run a forward pass
4. Display model architecture and metrics
5. Create a visualization of predictions

## Basic Usage

### 1. Import the model

```python
from m2fmoe import M2FMoE
import torch

# Create model
model = M2FMoE(
    input_size=1,      # Number of features
    d_model=512,       # Model dimension
    num_layers=3,      # Number of layers
    seq_len=96,        # Input sequence length
    pred_len=96        # Prediction length
)

# Prepare input
x = torch.randn(32, 96, 1)  # (batch_size, seq_len, input_size)

# Forward pass
predictions = model(x)  # (32, 96, 1)
```

### 2. Training

Train on your own data:

```bash
python scripts/train.py \
    --data_path data/your_data.npy \
    --seq_len 96 \
    --pred_len 96 \
    --epochs 100 \
    --batch_size 32
```

Your data should be a numpy array of shape `(num_samples, num_features)`.

### 3. Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path data/test_data.npy \
    --output_dir results
```

## Understanding the Architecture

M²FMoE consists of three main components:

### 1. Multi-Resolution Layer
- Captures patterns at different temporal scales
- Uses convolutions with varying kernel sizes (3, 5, 7)
- Fuses multi-scale features

### 2. Multi-View Layer
- **Temporal View**: Sequential patterns via self-attention
- **Statistical View**: Statistical features via MLPs
- **Trend View**: Long-term trends via moving averages

### 3. Frequency MoE Layer
- Transforms to frequency domain using FFT
- Routes to top-k experts adaptively
- Transforms back to time domain using inverse FFT

## Common Use Cases

### Case 1: Univariate Forecasting

```python
model = M2FMoE(input_size=1, seq_len=96, pred_len=96)
```

### Case 2: Multivariate Forecasting

```python
model = M2FMoE(input_size=7, seq_len=96, pred_len=96)
```

### Case 3: Long-term Forecasting

```python
model = M2FMoE(
    input_size=1,
    seq_len=192,
    pred_len=192,
    d_model=512,
    num_layers=4
)
```

### Case 4: Resource-Constrained

```python
model = M2FMoE(
    input_size=1,
    seq_len=96,
    pred_len=96,
    d_model=256,
    num_layers=2,
    num_experts=4,
    top_k=1
)
```

## Data Preparation

### Format Requirements

Your data should be:
- **Format**: Numpy array (`.npy`) or can be loaded as numpy
- **Shape**: `(num_timesteps, num_features)`
- **Type**: Float32 or Float64

### Example Data Preparation

```python
import numpy as np
import pandas as pd

# From CSV
df = pd.read_csv('your_data.csv')
data = df.values  # Convert to numpy
np.save('data/your_data.npy', data)

# From time series
time_series = [...]  # Your time series list
data = np.array(time_series).reshape(-1, 1)
np.save('data/your_data.npy', data)
```

## Configuration

### Using Config Files

```bash
# Copy default config
cp configs/default_config.json configs/my_config.json

# Edit your config
# vim configs/my_config.json

# Use in training
python scripts/train.py --config configs/my_config.json
```

### Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `d_model` | Model dimension | 128-1024 |
| `num_layers` | Number of blocks | 2-6 |
| `num_experts` | Experts in MoE | 4-16 |
| `top_k` | Active experts | 1-4 |
| `learning_rate` | Learning rate | 1e-5 to 1e-3 |
| `batch_size` | Batch size | 16-128 |

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or model size
```bash
python scripts/train.py --batch_size 16 --d_model 256
```

### Issue: Training too slow

**Solution**: Reduce model complexity
```bash
python scripts/train.py --num_layers 2 --num_experts 4
```

### Issue: Poor performance

**Solution**: Try different hyperparameters
- Increase model capacity: `--d_model 1024 --num_layers 4`
- Adjust learning rate: `--learning_rate 5e-5`
- Change prediction length: `--pred_len 48`

## Next Steps

1. **Read the Paper**: Check `PAPER.md` for detailed architecture explanation
2. **Explore Code**: Look at `src/m2fmoe/` for implementation details
3. **Run Tests**: Execute `python tests/test_model.py`
4. **Customize**: Modify layers in `src/m2fmoe/layers/`
5. **Experiment**: Try different datasets and configurations

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Yaohui-Huang/M2FMoE/issues)
- **Documentation**: See `README.md` and `PAPER.md`
- **Examples**: Check `scripts/` directory

## Tips for Best Results

1. **Normalize your data**: The model works best with normalized inputs
2. **Tune sequence length**: Match it to your data's periodicity
3. **Start simple**: Begin with default configs, then tune
4. **Monitor training**: Watch for overfitting with validation loss
5. **Use appropriate metrics**: Choose metrics that match your use case

Happy forecasting! 🚀
