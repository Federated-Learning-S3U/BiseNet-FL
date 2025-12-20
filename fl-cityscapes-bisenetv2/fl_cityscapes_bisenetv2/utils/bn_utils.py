"""
Batch Normalization utilities for SiloBN and other BN-aware FL strategies.

This module provides utilities for identifying and handling BatchNorm parameters
in federated learning settings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def get_bn_layer_names(model: nn.Module) -> List[str]:
    """
    Get names of all BatchNorm layers in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of BatchNorm layer names
    """
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(name)
    return bn_layers


def get_bn_param_names(model: nn.Module) -> List[str]:
    """
    Get names of all BatchNorm parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of BatchNorm parameter names (including buffers like running_mean)
    """
    bn_param_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Add trainable parameters
            bn_param_names.append(f"{name}.weight")
            bn_param_names.append(f"{name}.bias")
            # Add buffers (running statistics)
            bn_param_names.append(f"{name}.running_mean")
            bn_param_names.append(f"{name}.running_var")
            bn_param_names.append(f"{name}.num_batches_tracked")
    
    return bn_param_names


def split_state_dict_by_bn(
    state_dict: Dict[str, torch.Tensor], 
    model: nn.Module
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split a state dict into BN and non-BN parameters.
    
    Args:
        state_dict: Model state dictionary
        model: PyTorch model (used to identify BN layers)
        
    Returns:
        Tuple of (non_bn_state_dict, bn_state_dict)
    """
    bn_param_names = set(get_bn_param_names(model))
    
    non_bn_params = {}
    bn_params = {}
    
    for key, value in state_dict.items():
        if key in bn_param_names:
            bn_params[key] = value
        else:
            non_bn_params[key] = value
    
    return non_bn_params, bn_params


def count_bn_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the number of BN and non-BN parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (num_non_bn_params, num_bn_params)
    """
    bn_param_names = set(get_bn_param_names(model))
    
    num_bn = 0
    num_non_bn = 0
    
    for name, param in model.state_dict().items():
        if name in bn_param_names:
            num_bn += param.numel()
        else:
            num_non_bn += param.numel()
    
    return num_non_bn, num_bn


def freeze_bn_layers(model: nn.Module) -> None:
    """
    Freeze all BatchNorm layers in the model (set to eval mode).
    
    This prevents BN statistics from being updated during training.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            # Also freeze the parameters
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_bn_layers(model: nn.Module) -> None:
    """
    Unfreeze all BatchNorm layers in the model.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()
            for param in module.parameters():
                param.requires_grad = True


def reset_bn_stats(model: nn.Module) -> None:
    """
    Reset running statistics of all BatchNorm layers.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
