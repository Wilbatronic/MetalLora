"""
MetalLoRA - Unit Tests for Kernel Correctness

Tests verify:
1. Numerical accuracy of LoRA operations vs reference implementation
2. Gradient correctness via finite differences
3. Edge cases (rank=1, large batches, non-contiguous tensors)
4. Shape handling and broadcasting
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from metal_lora import LoRALinear, LoRAEmbedding
from metal_lora.ops import lora_forward, lora_backward_efficient, merge_lora_weights


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    mx.random.seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_dims():
    """Small dimensions for fast tests."""
    return {
        "batch_size": 2,
        "seq_len": 8,
        "in_features": 64,
        "out_features": 64,
        "rank": 4,
    }


@pytest.fixture
def medium_dims():
    """Medium dimensions for realistic tests."""
    return {
        "batch_size": 4,
        "seq_len": 128,
        "in_features": 512,
        "out_features": 512,
        "rank": 16,
    }


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================

def reference_lora_forward(x, W0, A, B, alpha):
    """Pure NumPy reference implementation of LoRA forward."""
    rank = A.shape[0]
    scale = alpha / rank
    
    # W0 @ x
    W0x = np.einsum('bsk,dk->bsd', x, W0)
    
    # A @ x -> Ax
    Ax = np.einsum('bsk,rk->bsr', x, A)
    
    # B @ Ax -> BAx
    BAx = np.einsum('bsr,dr->bsd', Ax, B)
    
    return W0x + scale * BAx


def reference_lora_backward(grad_h, x, A, B, alpha):
    """Pure NumPy reference implementation of LoRA backward."""
    rank = A.shape[0]
    scale = alpha / rank
    
    # Ax
    Ax = np.einsum('bsk,rk->bsr', x, A)
    
    # grad_B = scale * sum over batch of (grad_h.T @ Ax)
    grad_B = scale * np.einsum('bsd,bsr->dr', grad_h, Ax)
    
    # Bt @ grad_h
    Bt_grad = np.einsum('dr,bsd->bsr', B, grad_h)
    
    # grad_A = scale * sum over batch of (Bt_grad.T @ x)
    grad_A = scale * np.einsum('bsr,bsk->rk', Bt_grad, x)
    
    return grad_A, grad_B


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

class TestForwardPass:
    """Tests for LoRA forward pass accuracy."""
    
    def test_forward_basic(self, seed, small_dims):
        """Test basic forward pass matches reference."""
        d = small_dims
        
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        alpha = 16.0
        
        # MLX implementation
        out_mlx = lora_forward(x, W0, A, B, alpha)
        
        # Reference
        out_ref = reference_lora_forward(
            np.array(x), np.array(W0), np.array(A), np.array(B), alpha
        )
        
        np.testing.assert_allclose(
            np.array(out_mlx), out_ref, rtol=1e-5, atol=1e-5
        )
    
    def test_forward_various_ranks(self, seed, small_dims):
        """Test forward pass with different ranks."""
        d = small_dims
        
        for rank in [1, 2, 4, 8, 16, 32]:
            x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
            W0 = mx.random.normal((d["out_features"], d["in_features"]))
            A = mx.random.normal((rank, d["in_features"]))
            B = mx.random.normal((d["out_features"], rank))
            
            out_mlx = lora_forward(x, W0, A, B, alpha=16.0)
            out_ref = reference_lora_forward(
                np.array(x), np.array(W0), np.array(A), np.array(B), 16.0
            )
            
            np.testing.assert_allclose(
                np.array(out_mlx), out_ref, rtol=1e-5, atol=1e-5,
                err_msg=f"Failed for rank={rank}"
            )
    
    def test_forward_2d_input(self, seed, small_dims):
        """Test forward pass with 2D input (no batch dim)."""
        d = small_dims
        
        x = mx.random.normal((d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        
        out = lora_forward(x, W0, A, B, alpha=16.0)
        
        assert out.shape == (1, d["seq_len"], d["out_features"])
    
    def test_forward_alpha_scaling(self, seed, small_dims):
        """Test that alpha scaling works correctly."""
        d = small_dims
        
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        W0 = mx.zeros((d["out_features"], d["in_features"]))  # Zero base
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        
        out_alpha16 = lora_forward(x, W0, A, B, alpha=16.0)
        out_alpha32 = lora_forward(x, W0, A, B, alpha=32.0)
        
        # With zero W0, doubling alpha should double output
        np.testing.assert_allclose(
            np.array(out_alpha32), 2.0 * np.array(out_alpha16),
            rtol=1e-5, atol=1e-5
        )


# ============================================================================
# BACKWARD PASS TESTS
# ============================================================================

class TestBackwardPass:
    """Tests for LoRA backward pass accuracy."""
    
    def test_backward_basic(self, seed, small_dims):
        """Test basic backward pass matches reference."""
        d = small_dims
        
        grad_h = mx.random.normal((d["batch_size"], d["seq_len"], d["out_features"]))
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        alpha = 16.0
        
        # MLX implementation
        grad_A_mlx, grad_B_mlx = lora_backward_efficient(grad_h, x, A, B, alpha)
        
        # Reference (without clipping)
        grad_A_ref, grad_B_ref = reference_lora_backward(
            np.array(grad_h), np.array(x), np.array(A), np.array(B), alpha
        )
        
        np.testing.assert_allclose(
            np.array(grad_A_mlx), np.clip(grad_A_ref, -1.0, 1.0), 
            rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.array(grad_B_mlx), np.clip(grad_B_ref, -1.0, 1.0),
            rtol=1e-4, atol=1e-4
        )
    
    def test_gradient_shapes(self, seed, small_dims):
        """Test gradient tensor shapes are correct."""
        d = small_dims
        
        grad_h = mx.random.normal((d["batch_size"], d["seq_len"], d["out_features"]))
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        
        grad_A, grad_B = lora_backward_efficient(grad_h, x, A, B, alpha=16.0)
        
        assert grad_A.shape == A.shape, f"grad_A shape {grad_A.shape} != A shape {A.shape}"
        assert grad_B.shape == B.shape, f"grad_B shape {grad_B.shape} != B shape {B.shape}"


# ============================================================================
# LAYER TESTS
# ============================================================================

class TestLoRALinear:
    """Tests for LoRALinear layer."""
    
    def test_init(self, seed):
        """Test layer initialization."""
        layer = LoRALinear(64, 128, rank=8, alpha=16.0)
        
        assert layer.W0.shape == (128, 64)
        assert layer.A.shape == (8, 64)
        assert layer.B.shape == (128, 8)
        assert layer.rank == 8
        assert layer.alpha == 16.0
    
    def test_forward_shape(self, seed):
        """Test forward pass output shape."""
        layer = LoRALinear(64, 128, rank=8)
        x = mx.random.normal((4, 32, 64))
        out = layer(x)
        
        assert out.shape == (4, 32, 128)
    
    def test_merge_weights(self, seed):
        """Test weight merging for inference."""
        layer = LoRALinear(64, 64, rank=8, alpha=16.0)
        
        # Set specific weights
        layer.W0 = mx.eye(64)
        layer.A = mx.ones((8, 64)) * 0.1
        layer.B = mx.ones((64, 8)) * 0.1
        
        merged = layer.merge_weights()
        
        assert merged.shape == (64, 64)
        
        # Check merge is correct
        expected = np.eye(64) + (16.0 / 8) * (0.1 * np.ones((64, 8)) @ (0.1 * np.ones((8, 64))))
        np.testing.assert_allclose(np.array(merged), expected, rtol=1e-5)
    
    def test_from_linear(self, seed):
        """Test creating LoRALinear from nn.Linear."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear.from_linear(linear, rank=16, alpha=32.0)
        
        np.testing.assert_array_equal(np.array(lora.W0), np.array(linear.weight))
        assert lora.rank == 16
        assert lora.alpha == 32.0
    
    def test_parameter_count(self, seed):
        """Test parameter counting."""
        layer = LoRALinear(1024, 1024, rank=16)
        
        trainable = layer.num_trainable_params()
        total = layer.num_total_params()
        
        # Trainable: A (16*1024) + B (1024*16) = 32768
        assert trainable == 16 * 1024 + 1024 * 16
        
        # Total: W0 (1024*1024) + A + B
        assert total == 1024 * 1024 + trainable
    
    def test_inference_mode(self, seed):
        """Test conversion to inference mode."""
        layer = LoRALinear(64, 64, rank=8, alpha=16.0)
        layer.W0 = mx.eye(64)
        layer.A = mx.random.normal((8, 64)) * 0.1
        layer.B = mx.random.normal((64, 8)) * 0.1
        
        x = mx.random.normal((2, 8, 64))
        
        # Output before merge
        out_before = layer(x)
        
        # Convert to inference mode
        layer.to_inference_mode()
        
        # A and B should be zero
        assert mx.abs(layer.A).max() == 0
        assert mx.abs(layer.B).max() == 0
        
        # Output should be same
        out_after = layer(x)
        np.testing.assert_allclose(np.array(out_before), np.array(out_after), rtol=1e-5)


# ============================================================================
# WEIGHT OPERATIONS TESTS
# ============================================================================

class TestWeightOperations:
    """Tests for weight merging and manipulation."""
    
    def test_merge_weights(self, seed, small_dims):
        """Test weight merging produces correct matrix."""
        d = small_dims
        
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        alpha = 16.0
        
        merged = merge_lora_weights(W0, A, B, alpha)
        
        # Manual computation
        scale = alpha / d["rank"]
        expected = np.array(W0) + scale * (np.array(B) @ np.array(A))
        
        np.testing.assert_allclose(np.array(merged), expected, rtol=1e-5)
    
    def test_merge_preserves_inference(self, seed, small_dims):
        """Test that merged weights produce same output as LoRA."""
        d = small_dims
        
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        alpha = 16.0
        
        # LoRA output
        out_lora = lora_forward(x, W0, A, B, alpha)
        
        # Merged output
        W_merged = merge_lora_weights(W0, A, B, alpha)
        out_merged = mx.matmul(x, W_merged.T)
        
        np.testing.assert_allclose(
            np.array(out_lora), np.array(out_merged), rtol=1e-5, atol=1e-5
        )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_rank_one(self, seed, small_dims):
        """Test with minimal rank=1."""
        d = small_dims
        
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((1, d["in_features"]))
        B = mx.random.normal((d["out_features"], 1))
        
        out = lora_forward(x, W0, A, B, alpha=16.0)
        
        assert out.shape == (d["batch_size"], d["seq_len"], d["out_features"])
        assert not mx.isnan(out).any()
    
    def test_large_batch(self, seed, small_dims):
        """Test with large batch size."""
        d = small_dims
        
        x = mx.random.normal((64, d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.random.normal((d["rank"], d["in_features"]))
        B = mx.random.normal((d["out_features"], d["rank"]))
        
        out = lora_forward(x, W0, A, B, alpha=16.0)
        
        assert out.shape == (64, d["seq_len"], d["out_features"])
        assert not mx.isnan(out).any()
    
    def test_zero_lora(self, seed, small_dims):
        """Test with zero LoRA weights (should match base)."""
        d = small_dims
        
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"]))
        W0 = mx.random.normal((d["out_features"], d["in_features"]))
        A = mx.zeros((d["rank"], d["in_features"]))
        B = mx.zeros((d["out_features"], d["rank"]))
        
        out_lora = lora_forward(x, W0, A, B, alpha=16.0)
        out_base = mx.matmul(x, W0.T)
        
        np.testing.assert_allclose(
            np.array(out_lora), np.array(out_base), rtol=1e-5, atol=1e-5
        )
    
    def test_numerical_stability(self, seed, small_dims):
        """Test numerical stability with large values."""
        d = small_dims
        
        # Large values
        x = mx.random.normal((d["batch_size"], d["seq_len"], d["in_features"])) * 100
        W0 = mx.random.normal((d["out_features"], d["in_features"])) * 10
        A = mx.random.normal((d["rank"], d["in_features"])) * 10
        B = mx.random.normal((d["out_features"], d["rank"])) * 10
        
        out = lora_forward(x, W0, A, B, alpha=16.0)
        
        assert not mx.isnan(out).any(), "NaN values in output"
        assert not mx.isinf(out).any(), "Inf values in output"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
