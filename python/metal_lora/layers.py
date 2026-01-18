"""
MetalLoRA - High-level LoRA layers

Drop-in replacements for nn.Linear that use LoRA for efficient fine-tuning.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Callable
import math

from .ops import lora_forward, lora_backward_efficient, merge_lora_weights


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).
    
    This layer wraps a base linear transformation and adds trainable low-rank
    matrices A and B. During training, only A and B are updated while the
    base weights remain frozen.
    
    The forward pass computes:
        h = W₀x + (α/r) * B @ A @ x
    
    where:
        - W₀ is the frozen base weight [out_features, in_features]
        - A is the trainable down-projection [rank, in_features]
        - B is the trainable up-projection [out_features, rank]
        - α is a scaling hyperparameter
        - r is the rank
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (smaller = fewer parameters, larger = more capacity)
        alpha: Scaling factor (typically alpha = 2 * rank)
        dropout: Dropout probability applied to LoRA path
        use_bias: Whether to include bias term
        freeze_base: Whether to freeze base weights (True for fine-tuning)
    
    Example:
        >>> layer = LoRALinear(4096, 4096, rank=16, alpha=32)
        >>> x = mx.random.normal((4, 512, 4096))
        >>> out = layer(x)
        >>> print(out.shape)
        (4, 512, 4096)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_bias: bool = False,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.freeze_base = freeze_base
        
        # Base weight matrix (typically frozen)
        self.W0 = mx.zeros((out_features, in_features))
        
        # LoRA matrices with proper initialization
        # A: Kaiming uniform initialization
        # B: Zero initialization (so initial LoRA contribution is zero)
        bound = 1.0 / math.sqrt(in_features)
        self.A = mx.random.uniform(-bound, bound, (rank, in_features))
        self.B = mx.zeros((out_features, rank))
        
        # Optional bias
        if use_bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
        
        # Mark base weights as non-trainable if freezing
        if freeze_base:
            self.W0 = mx.stop_gradient(self.W0)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the LoRA layer.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        # Store original shape for restoration
        original_shape = x.shape
        
        # Reshape to 3D for batched ops if needed
        if x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim > 3:
            batch_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        
        # Use optimized forward
        out = lora_forward(
            x=x,
            W0=self.W0,
            A=self.A,
            B=self.B,
            alpha=self.alpha,
            dropout=self.dropout,
            training=self.training,
        )
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        
        # Restore original batch dimensions
        if len(original_shape) == 2:
            out = out.squeeze(0)
        elif len(original_shape) > 3:
            out = out.reshape(*batch_dims, out.shape[-1])
        
        return out
    
    def merge_weights(self) -> mx.array:
        """
        Merge LoRA weights into base weights for inference.
        
        Returns:
            Merged weight matrix [out_features, in_features]
        """
        return merge_lora_weights(self.W0, self.A, self.B, self.alpha)
    
    def to_inference_mode(self) -> "LoRALinear":
        """
        Convert to inference mode by merging weights.
        
        After calling this, the layer uses a single merged weight matrix
        for maximum inference speed.
        
        Returns:
            Self (modified in-place)
        """
        self.W0 = self.merge_weights()
        self.A = mx.zeros_like(self.A)
        self.B = mx.zeros_like(self.B)
        return self
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Create LoRALinear from an existing nn.Linear layer.
        
        Args:
            linear: Base linear layer to wrap
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
        
        Returns:
            New LoRALinear layer with copied base weights
        """
        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        
        lora = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_bias=linear.bias is not None,
            freeze_base=True,
        )
        
        # Copy base weights
        lora.W0 = linear.weight
        if linear.bias is not None:
            lora.bias = linear.bias
        
        return lora
    
    def trainable_parameters(self) -> dict:
        """
        Get only the trainable LoRA parameters.
        
        Returns:
            Dictionary with A, B, and optionally bias
        """
        params = {"A": self.A, "B": self.B}
        if self.bias is not None and not self.freeze_base:
            params["bias"] = self.bias
        return params
    
    def num_trainable_params(self) -> int:
        """
        Count number of trainable parameters.
        
        Returns:
            Total number of trainable parameters
        """
        count = self.A.size + self.B.size
        if self.bias is not None and not self.freeze_base:
            count += self.bias.size
        return count
    
    def num_total_params(self) -> int:
        """
        Count total number of parameters (including frozen).
        
        Returns:
            Total number of parameters
        """
        count = self.W0.size + self.A.size + self.B.size
        if self.bias is not None:
            count += self.bias.size
        return count
    
    def compression_ratio(self) -> float:
        """
        Compute compression ratio vs full fine-tuning.
        
        Returns:
            Ratio of trainable params to total params
        """
        return self.num_trainable_params() / self.num_total_params()


class LoRAEmbedding(nn.Module):
    """
    Embedding layer with LoRA adaptation.
    
    Similar to LoRALinear but for embedding lookups. The base embedding
    table is frozen while low-rank adapters learn task-specific adjustments.
    
    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
        rank: LoRA rank
        alpha: Scaling factor
        freeze_base: Whether to freeze base embeddings
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha
        
        # Base embeddings
        self.embeddings = mx.random.normal((num_embeddings, embedding_dim)) * 0.02
        
        # LoRA adapters
        bound = 1.0 / math.sqrt(embedding_dim)
        self.A = mx.random.uniform(-bound, bound, (rank, embedding_dim))
        self.B = mx.zeros((num_embeddings, rank))
        
        if freeze_base:
            self.embeddings = mx.stop_gradient(self.embeddings)
    
    def __call__(self, indices: mx.array) -> mx.array:
        """
        Lookup embeddings with LoRA adaptation.
        
        Args:
            indices: Token indices [...any shape]
        
        Returns:
            Embeddings [..., embedding_dim]
        """
        # Base embedding lookup
        base_emb = self.embeddings[indices]
        
        # LoRA contribution: B[indices] @ A
        scale = self.alpha / self.rank
        B_selected = self.B[indices]  # [..., rank]
        lora_emb = mx.matmul(B_selected, self.A)  # [..., embedding_dim]
        
        return base_emb + scale * lora_emb
    
    def merge_weights(self) -> mx.array:
        """Merge LoRA into base embeddings."""
        scale = self.alpha / self.rank
        return self.embeddings + scale * mx.matmul(self.B, self.A)


class LoRAAttention(nn.Module):
    """
    Multi-head attention with LoRA on Q, K, V projections.
    
    Applies LoRA to selected projections (typically Q and V for efficiency).
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        rank: LoRA rank for each projection
        alpha: LoRA scaling factor
        lora_targets: Which projections to apply LoRA ("q", "k", "v", "o")
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int = 8,
        alpha: float = 16.0,
        lora_targets: tuple = ("q", "v"),
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q, K, V, O projections
        self.q_proj = LoRALinear(dim, dim, rank, alpha) if "q" in lora_targets else nn.Linear(dim, dim)
        self.k_proj = LoRALinear(dim, dim, rank, alpha) if "k" in lora_targets else nn.Linear(dim, dim)
        self.v_proj = LoRALinear(dim, dim, rank, alpha) if "v" in lora_targets else nn.Linear(dim, dim)
        self.o_proj = LoRALinear(dim, dim, rank, alpha) if "o" in lora_targets else nn.Linear(dim, dim)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through attention with LoRA projections.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask [batch, seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            scores = scores + mask
        
        attn = mx.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = mx.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        
        # Output projection
        return self.o_proj(out)
