"""
Unit tests for M2FMoE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from m2fmoe import M2FMoE
from m2fmoe.layers import MultiResolutionLayer, MultiViewLayer, FrequencyMoE


def test_multi_resolution_layer():
    """Test Multi-Resolution Layer."""
    print("Testing Multi-Resolution Layer...")
    
    batch_size, seq_len, d_model = 2, 96, 64
    layer = MultiResolutionLayer(d_model=d_model, num_resolutions=3)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Multi-Resolution Layer test passed")


def test_multi_view_layer():
    """Test Multi-View Layer."""
    print("Testing Multi-View Layer...")
    
    batch_size, seq_len, d_model = 2, 96, 64
    layer = MultiViewLayer(d_model=d_model, num_views=3)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Multi-View Layer test passed")


def test_frequency_moe_layer():
    """Test Frequency MoE Layer."""
    print("Testing Frequency MoE Layer...")
    
    batch_size, seq_len, d_model = 2, 96, 64
    layer = FrequencyMoE(d_model=d_model, num_experts=4, top_k=2)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Frequency MoE Layer test passed")


def test_m2fmoe_model():
    """Test full M2FMoE model."""
    print("Testing M2FMoE model...")
    
    batch_size = 2
    seq_len = 96
    pred_len = 96
    input_size = 1
    
    model = M2FMoE(
        input_size=input_size,
        d_model=64,
        num_layers=2,
        num_resolutions=3,
        num_views=3,
        num_experts=4,
        top_k=2,
        pred_len=pred_len,
        seq_len=seq_len,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    expected_shape = (batch_size, pred_len, input_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("✓ M2FMoE model test passed")


def test_model_forward_backward():
    """Test forward and backward pass."""
    print("Testing forward and backward pass...")
    
    model = M2FMoE(
        input_size=1,
        d_model=32,
        num_layers=1,
        seq_len=48,
        pred_len=24
    )
    
    x = torch.randn(4, 48, 1)
    y = torch.randn(4, 24, 1)
    
    # Forward pass
    pred = model(x)
    
    # Loss and backward
    loss = torch.nn.MSELoss()(pred, y)
    loss.backward()
    
    # Check gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed"
    
    print("✓ Forward and backward pass test passed")


def test_different_input_sizes():
    """Test model with different input sizes."""
    print("Testing different input sizes...")
    
    for input_size in [1, 5, 10]:
        model = M2FMoE(
            input_size=input_size,
            d_model=32,
            num_layers=1,
            seq_len=48,
            pred_len=24
        )
        
        x = torch.randn(2, 48, input_size)
        output = model(x)
        
        assert output.shape == (2, 24, input_size), f"Failed for input_size={input_size}"
    
    print("✓ Different input sizes test passed")


def test_different_sequence_lengths():
    """Test model with different sequence lengths."""
    print("Testing different sequence lengths...")
    
    for seq_len, pred_len in [(24, 12), (48, 24), (96, 48), (192, 96)]:
        model = M2FMoE(
            input_size=1,
            d_model=32,
            num_layers=1,
            seq_len=seq_len,
            pred_len=pred_len
        )
        
        x = torch.randn(2, seq_len, 1)
        output = model(x)
        
        assert output.shape == (2, pred_len, 1), f"Failed for seq_len={seq_len}, pred_len={pred_len}"
    
    print("✓ Different sequence lengths test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running M2FMoE Tests")
    print("="*60)
    print()
    
    tests = [
        test_multi_resolution_layer,
        test_multi_view_layer,
        test_frequency_moe_layer,
        test_m2fmoe_model,
        test_model_forward_backward,
        test_different_input_sizes,
        test_different_sequence_lengths,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            failed += 1
        print()
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
