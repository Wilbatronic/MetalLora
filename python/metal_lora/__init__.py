"""MetalLoRA - Optimized LoRA for MLX on Apple Silicon."""

import platform

__version__ = "0.1.0"

# Check platform - MLX only works on macOS with Apple Silicon
_IS_MACOS = platform.system() == "Darwin"

# Always-available imports (exceptions, kernels availability check)
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
from .kernels import is_metal_available

# MLX-dependent imports (only available on macOS with MLX)
if _IS_MACOS:
    try:
        from .kernels import lora_backward_metal, lora_forward_metal
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
        _FULL_IMPORT = True
    except ImportError as e:
        _FULL_IMPORT = False
        _IMPORT_ERROR = str(e)
else:
    _FULL_IMPORT = False
    _IMPORT_ERROR = "MetalLoRA requires macOS with Apple Silicon"

__all__ = [
    # Always available
    "is_metal_available",
    "MetalLoRAError", "ConfigurationError", "ShapeError", "DeviceError", "AdapterError",
    "validate_rank", "validate_alpha", "validate_probability",
]

# Extend __all__ if full import succeeded
if _FULL_IMPORT:
    __all__.extend([
        # Core
        "LoRALinear", "LoRAEmbedding",
        "lora_forward", "lora_backward", "merge_lora_weights",
        "lora_forward_metal", "lora_backward_metal",
        "save_adapter", "load_adapter", "apply_lora_to_model", "count_lora_parameters",
        # Training
        "LoRATrainer", "TrainingConfig", "TrainingState", "train_lora", "quick_finetune",
        # Optimizations
        "compress_weights", "decompress_weights", "save_compressed", "load_compressed",
        "MemoryPool", "get_memory_pool",
        "MultiAdapterManager", "SpeculativeDecoder", "LoRAKVCache", "LazyLoRA",
        "MixedPrecisionConfig", "MixedPrecisionTrainer", "FusedGradAccumulator",
        # Logging
        "logger", "get_logger", "set_log_level", "enable_debug", "disable_logging",
    ])

