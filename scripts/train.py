"""
Training script for M2FMoE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from tqdm import tqdm

from m2fmoe.models.m2fmoe import M2FMoE
from m2fmoe.utils.data_loader import create_dataloader, normalize_data
from m2fmoe.utils.metrics import calculate_metrics, print_metrics
from m2fmoe.utils.visualization import plot_loss_curve


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train M2FMoE model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/sample_data.npy',
                        help='Path to training data')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction sequence length')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=1,
                        help='Number of input features')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of M2FMoE layers')
    parser.add_argument('--num_resolutions', type=int, default=3,
                        help='Number of resolution levels')
    parser.add_argument('--num_views', type=int, default=3,
                        help='Number of views')
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts in MoE')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of experts to activate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in tqdm(dataloader, desc='Training'):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc='Validation'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return total_loss / len(dataloader), metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load and prepare data
    print("\nLoading data...")
    if os.path.exists(args.data_path):
        data = np.load(args.data_path)
        print(f"Data shape: {data.shape}")
    else:
        print(f"Data file not found: {args.data_path}")
        print("Generating synthetic data for demonstration...")
        # Generate synthetic data
        t = np.linspace(0, 100, 10000)
        data = np.sin(t) + 0.5 * np.sin(3*t) + 0.1 * np.random.randn(len(t))
        data = data.reshape(-1, 1)
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        np.save(args.data_path, data)
    
    # Normalize data
    data_norm, norm_params = normalize_data(data, method='zscore')
    
    # Split into train and validation
    train_size = int(0.8 * len(data_norm))
    train_data = data_norm[:train_size]
    val_data = data_norm[train_size:]
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = create_dataloader(
        val_data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = M2FMoE(
        input_size=args.input_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_resolutions=args.num_resolutions,
        num_views=args.num_views,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print_metrics(val_metrics, prefix='Validation ')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("Saved best model!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Plot and save loss curve
    plot_loss_curve(
        train_losses,
        val_losses,
        save_path=os.path.join(args.save_dir, 'loss_curve.png')
    )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
