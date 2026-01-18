"""
MetalLoRA - Optimized LoRA kernels for MLX on Apple Silicon

This package provides heavily optimized Metal kernels for LoRA (Low-Rank Adaptation)
training and inference on Apple Silicon, integrated with the MLX framework.

Quick Start:
    # For training:
    from metal_lora import LoRATrainer, TrainingConfig
    trainer = LoRATrainer(model, config=TrainingConfig(rank=16))
    trainer.train(train_data, num_epochs=3)
    
    # For inference:
    from metal_lora import LoRALinear
    layer = LoRALinear(4096, 4096, rank=16)
    output = layer(x)
"""

__version__ = "0.1.0"

# Core layers
from .layers import LoRALinear, LoRAEmbedding

# Low-level operations
from .ops import (
    lora_forward,
    lora_backward,
    merge_lora_weights,
)

# Utilities
from .utils import (
    save_adapter,
    load_adapter,
    apply_lora_to_model,
    count_lora_parameters,
)

# Training
from .trainer import (
    LoRATrainer,
    TrainingConfig,
    TrainingState,
    train_lora,
    quick_finetune,
)

__all__ = [
    # Core
    "LoRALinear",
    "LoRAEmbedding",
    # Operations
    "lora_forward",
    "lora_backward",
    "merge_lora_weights",
    # Utilities
    "save_adapter",
    "load_adapter",
    "apply_lora_to_model",
    "count_lora_parameters",
    # Training
    "LoRATrainer",
    "TrainingConfig",
    "TrainingState",
    "train_lora",
    "quick_finetune",
]
