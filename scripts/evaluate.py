"""
Evaluation script for M2FMoE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import argparse
import json

from m2fmoe.models.m2fmoe import M2FMoE
from m2fmoe.utils.data_loader import create_dataloader, normalize_data, denormalize_data
from m2fmoe.utils.metrics import calculate_metrics, print_metrics
from m2fmoe.utils.visualization import plot_predictions, plot_multiple_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate M2FMoE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/sample_data.npy',
                        help='Path to test data')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file (auto-detected if not provided)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config_path is None:
        config_dir = os.path.dirname(args.checkpoint)
        args.config_path = os.path.join(config_dir, 'config.json')
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    print("\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading test data...")
    data = np.load(args.data_path)
    print(f"Data shape: {data.shape}")
    
    # Normalize data
    data_norm, norm_params = normalize_data(data, method='zscore')
    
    # Create dataloader
    test_loader = create_dataloader(
        data_norm,
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = M2FMoE(
        input_size=config['input_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_resolutions=config['num_resolutions'],
        num_views=config['num_views'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        pred_len=config['pred_len'],
        seq_len=config['seq_len'],
        dropout=config['dropout']
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Checkpoint loaded (Epoch {checkpoint['epoch']})")
    
    # Evaluate
    print("\nEvaluating...")
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_x)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
            all_inputs.append(batch_x.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_inputs = torch.cat(all_inputs, dim=0).numpy()
    
    # Calculate metrics on normalized data
    metrics_norm = calculate_metrics(all_predictions, all_targets)
    print_metrics(metrics_norm, prefix='Normalized ')
    
    # Denormalize for real-scale metrics
    all_predictions_denorm = denormalize_data(all_predictions, norm_params, method='zscore')
    all_targets_denorm = denormalize_data(all_targets, norm_params, method='zscore')
    
    metrics_real = calculate_metrics(all_predictions_denorm, all_targets_denorm)
    print_metrics(metrics_real, prefix='Real-scale ')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    results = {
        'normalized_metrics': metrics_norm,
        'real_scale_metrics': metrics_real,
        'num_samples': len(all_predictions)
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nMetrics saved to {args.output_dir}/metrics.json")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    
    # Plot single prediction with history
    idx = 0
    plot_predictions(
        true=all_targets_denorm[idx, :, 0],
        pred=all_predictions_denorm[idx, :, 0],
        history=all_inputs[idx, :, 0] * norm_params['std'][0] + norm_params['mean'][0],
        title='Sample Prediction with History',
        save_path=os.path.join(args.output_dir, 'sample_prediction.png')
    )
    
    # Plot multiple predictions
    num_samples = min(args.num_samples, len(all_predictions))
    plot_multiple_predictions(
        trues=[all_targets_denorm[i, :, 0] for i in range(num_samples)],
        preds=[all_predictions_denorm[i, :, 0] for i in range(num_samples)],
        num_samples=num_samples,
        title='Multiple Prediction Samples',
        save_path=os.path.join(args.output_dir, 'multiple_predictions.png')
    )
    
    print(f"Visualizations saved to {args.output_dir}/")
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
