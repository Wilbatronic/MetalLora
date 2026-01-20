"""MetalLoRA - Optimized LoRA for MLX on Apple Silicon."""

__version__ = "0.1.0"

from .exceptions import (
    AdapterError,
    ConfigurationError,
    DeviceError,
    MetalLoRAError,
    ShapeError,
    validate_alpha,
    validate_probability,
    validate_rank,
)
from .kernels import is_metal_available, lora_backward_metal, lora_forward_metal
from .layers import LoRAEmbedding, LoRALinear
from .logging import disable_logging, enable_debug, get_logger, logger, set_log_level
from .ops import lora_backward, lora_forward, merge_lora_weights
from .optimizations import (
    FusedGradAccumulator,
    LazyLoRA,
    LoRAKVCache,
    MemoryPool,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    MultiAdapterManager,
    SpeculativeDecoder,
    compress_weights,
    decompress_weights,
    get_memory_pool,
    load_compressed,
    save_compressed,
)
from .trainer import LoRATrainer, TrainingConfig, TrainingState, quick_finetune, train_lora
from .utils import apply_lora_to_model, count_lora_parameters, load_adapter, save_adapter

__all__ = [
    # Core
    "LoRALinear", "LoRAEmbedding",
    "lora_forward", "lora_backward", "merge_lora_weights",
    "is_metal_available", "lora_forward_metal", "lora_backward_metal",
    "save_adapter", "load_adapter", "apply_lora_to_model", "count_lora_parameters",
    # Training
    "LoRATrainer", "TrainingConfig", "TrainingState", "train_lora", "quick_finetune",
    # Optimizations
    "compress_weights", "decompress_weights", "save_compressed", "load_compressed",
    "MemoryPool", "get_memory_pool",
    "MultiAdapterManager", "SpeculativeDecoder", "LoRAKVCache", "LazyLoRA",
    "MixedPrecisionConfig", "MixedPrecisionTrainer", "FusedGradAccumulator",
    # Exceptions
    "MetalLoRAError", "ConfigurationError", "ShapeError", "DeviceError", "AdapterError",
    "validate_rank", "validate_alpha", "validate_probability",
    # Logging
    "logger", "get_logger", "set_log_level", "enable_debug", "disable_logging",
]
