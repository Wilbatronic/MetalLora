"""
MetalLoRA - Optimized LoRA kernels for MLX on Apple Silicon

This package provides heavily optimized Metal kernels for LoRA (Low-Rank Adaptation)
training and inference on Apple Silicon, integrated with the MLX framework.

Example:
    >>> from metal_lora import LoRATrainer, TrainingConfig
    >>> trainer = LoRATrainer(model, config=TrainingConfig(rank=16))
    >>> trainer.train(train_data, num_epochs=3)
"""

__version__ = "0.1.0"

from .layers import LoRALinear, LoRAEmbedding
from .ops import (
    lora_forward,
    lora_backward,
    merge_lora_weights,
)
from .utils import (
    save_adapter,
    load_adapter,
    apply_lora_to_model,
    count_lora_parameters,
)

# Advanced Apple Silicon optimizations
from .advanced_ops import (
    FP16LoRALinear,
    QLoRALinear,
    StreamingLoRALinear,
    MultiAdapterLoRA,
    quantize_nf4,
    dequantize_nf4,
)

# Extreme performance optimizations
from .extreme_ops import (
    detect_hardware,
    get_hardware,
    select_optimal_kernel,
    GradientCheckpointedLoRA,
    TensorParallelLoRA,
    SpeculativeLoRADecoder,
    FusedLoRAAdamW,
    FusedLoRABlock,
)

# Training package
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
    "lora_forward",
    "lora_backward",
    "merge_lora_weights",
    "save_adapter",
    "load_adapter",
    "apply_lora_to_model",
    "count_lora_parameters",
    # Advanced
    "FP16LoRALinear",
    "QLoRALinear",
    "StreamingLoRALinear", 
    "MultiAdapterLoRA",
    "quantize_nf4",
    "dequantize_nf4",
    # Extreme
    "detect_hardware",
    "get_hardware",
    "select_optimal_kernel",
    "GradientCheckpointedLoRA",
    "TensorParallelLoRA",
    "SpeculativeLoRADecoder",
    "FusedLoRAAdamW",
    "FusedLoRABlock",
    # Training
    "LoRATrainer",
    "TrainingConfig",
    "TrainingState",
    "train_lora",
    "quick_finetune",
]
