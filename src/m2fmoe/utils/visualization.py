"""
Visualization utilities for time series forecasting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(
    true,
    pred,
    history=None,
    title='Time Series Forecasting',
    save_path=None,
    figsize=(15, 5)
):
    """
    Plot time series predictions against ground truth.
    
    Args:
        true: Ground truth values
        pred: Predicted values
        history: Historical values (optional)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Convert to numpy if torch tensor
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if history is not None and isinstance(history, torch.Tensor):
        history = history.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Plot history if provided
    if history is not None:
        hist_len = len(history)
        plt.plot(range(hist_len), history, 'b-', label='History', linewidth=2)
        offset = hist_len
    else:
        offset = 0
    
    # Plot predictions and ground truth
    pred_len = len(pred)
    x_pred = range(offset, offset + pred_len)
    
    plt.plot(x_pred, true, 'g-', label='Ground Truth', linewidth=2)
    plt.plot(x_pred, pred, 'r--', label='Prediction', linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return plt


def plot_multiple_predictions(
    trues,
    preds,
    num_samples=4,
    title='Multiple Predictions',
    save_path=None,
    figsize=(15, 10)
):
    """
    Plot multiple prediction samples in a grid.
    
    Args:
        trues: List or array of ground truth values
        preds: List or array of predicted values
        num_samples: Number of samples to plot
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    num_samples = min(num_samples, len(trues))
    rows = (num_samples + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        true = trues[i]
        pred = preds[i]
        
        # Convert to numpy if needed
        if isinstance(true, torch.Tensor):
            true = true.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        axes[i].plot(true, 'g-', label='Ground Truth', linewidth=2)
        axes[i].plot(pred, 'r--', label='Prediction', linewidth=2)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_attention_weights(
    attention_weights,
    title='Attention Weights',
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return plt


def plot_loss_curve(
    train_losses,
    val_losses=None,
    title='Training Loss Curve',
    save_path=None,
    figsize=(10, 6)
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    return plt
