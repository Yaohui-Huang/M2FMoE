"""
Quick demonstration of M2FMoE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from m2fmoe import M2FMoE
from m2fmoe.utils.data_loader import normalize_data
from m2fmoe.utils.metrics import calculate_metrics
from m2fmoe.utils.visualization import plot_predictions

def main():
    """Run a simple demonstration of the M2FMoE model."""
    
    print("="*60)
    print("M²FMoE Demonstration")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic time series data
    print("\n1. Generating synthetic time series data...")
    t = np.linspace(0, 100, 1000)
    data = np.sin(t) + 0.5 * np.sin(3*t) + 0.1 * np.random.randn(len(t))
    data = data.reshape(-1, 1)
    print(f"   Data shape: {data.shape}")
    
    # Normalize data
    data_norm, norm_params = normalize_data(data, method='zscore')
    
    # Prepare input and target
    seq_len = 96
    pred_len = 96
    
    input_seq = torch.FloatTensor(data_norm[:seq_len]).unsqueeze(0)  # (1, seq_len, 1)
    target_seq = data_norm[seq_len:seq_len+pred_len]
    
    print(f"   Input shape: {input_seq.shape}")
    print(f"   Target shape: {target_seq.shape}")
    
    # Initialize model
    print("\n2. Initializing M²FMoE model...")
    model = M2FMoE(
        input_size=1,
        d_model=128,  # Smaller for demo
        num_layers=2,
        num_resolutions=3,
        num_views=3,
        num_experts=4,
        top_k=2,
        pred_len=pred_len,
        seq_len=seq_len,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Model components
    print("\n3. Model components:")
    print("   ✓ Multi-Resolution Layer (captures patterns at different scales)")
    print("   ✓ Multi-View Layer (temporal, statistical, trend views)")
    print("   ✓ Frequency MoE Layer (adaptive frequency domain modeling)")
    
    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(input_seq)
    
    print(f"   Prediction shape: {predictions.shape}")
    
    # Calculate metrics (on random predictions since model is not trained)
    predictions_np = predictions.squeeze(0).numpy()
    metrics = calculate_metrics(predictions_np, target_seq)
    
    print("\n5. Metrics (untrained model - random baseline):")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"   {name:10s}: {value:.6f}")
    print("-" * 50)
    
    # Visualize
    print("\n6. Generating visualization...")
    try:
        plot_predictions(
            true=target_seq.flatten(),
            pred=predictions_np.flatten(),
            history=input_seq.squeeze(0).numpy().flatten(),
            title='M²FMoE Demo (Untrained Model)',
            save_path='demo_prediction.png'
        )
        print("   ✓ Visualization saved to 'demo_prediction.png'")
    except Exception as e:
        print(f"   Note: Visualization skipped (matplotlib display not available)")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train the model: python scripts/train.py")
    print("2. Evaluate: python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("3. Check README.md for more details")
    print()

if __name__ == '__main__':
    main()
