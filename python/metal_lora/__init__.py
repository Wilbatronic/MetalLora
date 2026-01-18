"""MetalLoRA - Optimized LoRA for MLX on Apple Silicon."""

__version__ = "0.1.0"

from .layers import LoRALinear, LoRAEmbedding
from .ops import lora_forward, lora_backward, merge_lora_weights
from .utils import save_adapter, load_adapter, apply_lora_to_model, count_lora_parameters
from .trainer import LoRATrainer, TrainingConfig, TrainingState, train_lora, quick_finetune
from .optimizations import (
    compress_weights, decompress_weights, save_compressed, load_compressed,
    MemoryPool, get_memory_pool,
    MultiAdapterManager,
    SpeculativeDecoder,
    LoRAKVCache,
    LazyLoRA,
    MixedPrecisionConfig, MixedPrecisionTrainer,
    FusedGradAccumulator,
)

__all__ = [
    # Core
    "LoRALinear", "LoRAEmbedding",
    "lora_forward", "lora_backward", "merge_lora_weights",
    "save_adapter", "load_adapter", "apply_lora_to_model", "count_lora_parameters",
    # Training
    "LoRATrainer", "TrainingConfig", "TrainingState", "train_lora", "quick_finetune",
    # Optimizations
    "compress_weights", "decompress_weights", "save_compressed", "load_compressed",
    "MemoryPool", "get_memory_pool",
    "MultiAdapterManager",
    "SpeculativeDecoder",
    "LoRAKVCache",
    "LazyLoRA",
    "MixedPrecisionConfig", "MixedPrecisionTrainer",
    "FusedGradAccumulator",
]
