"""
MetalLoRA - Utility functions

Helper functions for saving/loading adapters, applying LoRA to models,
and other common operations.
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json

from .layers import LoRALinear, LoRAEmbedding


def save_adapter(
    model: nn.Module,
    path: Union[str, Path],
    adapter_name: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save LoRA adapter weights to disk.
    
    Only saves the trainable LoRA parameters (A and B matrices),
    not the frozen base weights.
    
    Args:
        model: Model containing LoRA layers
        path: Directory to save adapter to
        adapter_name: Name for the adapter
        metadata: Optional metadata to include
    
    Example:
        >>> model = create_model_with_lora(base_model, rank=16)
        >>> train(model, data)
        >>> save_adapter(model, "adapters/my_task")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Extract LoRA parameters
    lora_params = {}
    lora_config = {}
    
    def extract_lora(module: nn.Module, prefix: str = ""):
        for name, child in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                lora_params[f"{full_name}.A"] = child.A
                lora_params[f"{full_name}.B"] = child.B
                lora_config[full_name] = {
                    "type": "LoRALinear",
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "in_features": child.in_features,
                    "out_features": child.out_features,
                }
            elif isinstance(child, LoRAEmbedding):
                lora_params[f"{full_name}.A"] = child.A
                lora_params[f"{full_name}.B"] = child.B
                lora_config[full_name] = {
                    "type": "LoRAEmbedding",
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "num_embeddings": child.num_embeddings,
                    "embedding_dim": child.embedding_dim,
                }
    
    extract_lora(model)
    
    # Save weights
    weights_path = path / f"{adapter_name}.safetensors"
    mx.save_safetensors(str(weights_path), lora_params)
    
    # Save config
    config = {
        "adapter_name": adapter_name,
        "lora_layers": lora_config,
        "metadata": metadata or {},
    }
    config_path = path / f"{adapter_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved adapter to {path}")
    print(f"  - Weights: {weights_path}")
    print(f"  - Config: {config_path}")
    print(f"  - Total parameters: {sum(p.size for p in lora_params.values()):,}")


def load_adapter(
    model: nn.Module,
    path: Union[str, Path],
    adapter_name: str = "default",
) -> nn.Module:
    """
    Load LoRA adapter weights from disk into model.
    
    Args:
        model: Model with LoRA layers to load into
        path: Directory containing adapter files
        adapter_name: Name of the adapter to load
    
    Returns:
        Model with loaded adapter weights
    
    Example:
        >>> model = create_model_with_lora(base_model, rank=16)
        >>> model = load_adapter(model, "adapters/my_task")
    """
    path = Path(path)
    
    # Load weights
    weights_path = path / f"{adapter_name}.safetensors"
    lora_params = mx.load(str(weights_path))
    
    # Load config
    config_path = path / f"{adapter_name}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Apply weights to model
    def apply_lora(module: nn.Module, prefix: str = ""):
        for name, child in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, (LoRALinear, LoRAEmbedding)):
                a_key = f"{full_name}.A"
                b_key = f"{full_name}.B"
                
                if a_key in lora_params and b_key in lora_params:
                    child.A = lora_params[a_key]
                    child.B = lora_params[b_key]
                else:
                    print(f"Warning: No weights found for {full_name}")
    
    apply_lora(model)
    
    print(f"Loaded adapter from {path}")
    print(f"  - Loaded {len(lora_params)} weight tensors")
    
    return model


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to specified linear layers in a model.
    
    This function replaces nn.Linear layers with LoRALinear layers,
    copying the original weights as frozen base weights.
    
    Args:
        model: Model to modify
        rank: LoRA rank for all layers
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all linear layers.
        dropout: Dropout probability for LoRA paths
    
    Returns:
        Modified model with LoRA layers
    
    Example:
        >>> base_model = load_pretrained("llama-7b")
        >>> lora_model = apply_lora_to_model(
        ...     base_model,
        ...     rank=16,
        ...     target_modules=["q_proj", "v_proj"]
        ... )
    """
    def should_apply(name: str) -> bool:
        if target_modules is None:
            return True
        return any(t in name for t in target_modules)
    
    def replace_linear(module: nn.Module, name: str = ""):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear) and should_apply(full_name):
                # Replace with LoRALinear
                lora_layer = LoRALinear.from_linear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, child_name, lora_layer)
                print(f"Applied LoRA to: {full_name}")
            else:
                # Recurse into children
                replace_linear(child, full_name)
    
    replace_linear(model)
    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model with LoRA.
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        Dictionary with parameter counts
    """
    trainable = 0
    frozen = 0
    lora_layers = 0
    
    def count(module: nn.Module):
        nonlocal trainable, frozen, lora_layers
        
        for child in module.children():
            if isinstance(child, (LoRALinear, LoRAEmbedding)):
                lora_layers += 1
                trainable += child.num_trainable_params()
                frozen += child.num_total_params() - child.num_trainable_params()
            else:
                # Count regular parameters
                for param in child.parameters():
                    frozen += param.size
                count(child)
    
    count(model)
    
    total = trainable + frozen
    
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "lora_layers": lora_layers,
        "trainable_ratio": trainable / total if total > 0 else 0,
    }


def merge_and_save(
    model: nn.Module,
    path: Union[str, Path],
    format: str = "safetensors",
) -> None:
    """
    Merge LoRA weights into base weights and save full model.
    
    Use this for deployment when you want a single model file
    without separate adapter weights.
    
    Args:
        model: Model with LoRA layers
        path: Path to save merged model
        format: Save format ("safetensors" or "npz")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge all LoRA layers
    def merge_lora(module: nn.Module):
        for child in module.children():
            if isinstance(child, LoRALinear):
                child.to_inference_mode()
            elif isinstance(child, LoRAEmbedding):
                child.embeddings = child.merge_weights()
                child.A = mx.zeros_like(child.A)
                child.B = mx.zeros_like(child.B)
            else:
                merge_lora(child)
    
    merge_lora(model)
    
    # Save model
    weights = dict(model.parameters())
    
    if format == "safetensors":
        mx.save_safetensors(str(path), weights)
    else:
        mx.savez(str(path), **weights)
    
    print(f"Saved merged model to {path}")


def get_lora_summary(model: nn.Module) -> str:
    """
    Get a human-readable summary of LoRA configuration.
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        Formatted summary string
    """
    params = count_lora_parameters(model)
    
    lines = [
        "=" * 50,
        "LoRA Configuration Summary",
        "=" * 50,
        f"Total parameters:      {params['total']:>15,}",
        f"Trainable parameters:  {params['trainable']:>15,}",
        f"Frozen parameters:     {params['frozen']:>15,}",
        f"LoRA layers:           {params['lora_layers']:>15}",
        f"Trainable ratio:       {params['trainable_ratio']:>14.2%}",
        "=" * 50,
    ]
    
    return "\n".join(lines)
