"""Utilities for extracting and analyzing model weights and architecture components."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def get_device_from_state_dict(state_dict: Dict) -> torch.device:
    """
    Infer device from a state dictionary (returns 'cpu').
    """
    return torch.device('cpu')


def extract_weights_by_layer(model_state_dict: Dict, exclude_bn_stats: bool = True) -> Dict[str, torch.Tensor]:
    """
    Extract model weights organized by layer name.
    
    Args:
        model_state_dict: Model state dictionary
        exclude_bn_stats: If True, exclude BN running_mean and running_var
    
    Returns:
        Dictionary mapping layer names to weight tensors
    """
    weights_dict = {}
    
    for layer_name, param in model_state_dict.items():
        if exclude_bn_stats:
            # Skip BN running statistics
            if 'running_mean' in layer_name or 'running_var' in layer_name:
                continue
        
        weights_dict[layer_name] = param
    
    return weights_dict


def extract_bn_statistics(model_state_dict: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract BatchNorm layer statistics (running_mean, running_var, weight, bias).
    
    Args:
        model_state_dict: Model state dictionary
    
    Returns:
        Dictionary mapping BN layer names to their statistics
        {
            'bn_layer_name': {
                'running_mean': tensor,
                'running_var': tensor,
                'weight': tensor,
                'bias': tensor
            }
        }
    """
    bn_stats = {}
    
    for layer_name, param in model_state_dict.items():
        # Extract base layer name (without .running_mean, .weight, etc.)
        if 'bn' in layer_name.lower() or 'batchnorm' in layer_name.lower():
            base_name = layer_name.rsplit('.', 1)[0]
            
            if base_name not in bn_stats:
                bn_stats[base_name] = {}
            
            # Extract the parameter type
            param_type = layer_name.split('.')[-1]
            bn_stats[base_name][param_type] = param
    
    return bn_stats


def flatten_model_weights(model_state_dict: Dict, exclude_bn_stats: bool = True) -> torch.Tensor:
    """
    Flatten all model weights into a single vector.
    
    Args:
        model_state_dict: Model state dictionary
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Flattened weight tensor
    """
    weights = extract_weights_by_layer(model_state_dict, exclude_bn_stats)
    
    flattened = []
    for param in weights.values():
        flattened.append(param.flatten())
    
    return torch.cat(flattened)


def get_layer_names_by_component(model: nn.Module) -> Dict[str, List[str]]:
    """
    Map architecture components to their layer names.
    
    For BiSeNetV2, this includes:
    - detail_branch: DetailBranch layers
    - semantic_branch: StemBlock + SemanticBranch layers
    - decoder: Decoder layers
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping component names to lists of layer names
    """
    components = {
        'detail_branch': [],
        'semantic_branch': [],
        'decoder': [],
        'auxiliary_heads': [],
        'other': []
    }
    
    for name, module in model.named_modules():
        if 'detail' in name.lower():
            components['detail_branch'].append(name)
        elif 'stem' in name.lower() or 'semantic' in name.lower():
            components['semantic_branch'].append(name)
        elif 'decoder' in name.lower() or 'decode' in name.lower():
            components['decoder'].append(name)
        elif 'aux' in name.lower():
            components['auxiliary_heads'].append(name)
        elif name != '':  # Skip root module
            components['other'].append(name)
    
    return components


def get_branch_state_dict(
    model_state_dict: Dict,
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract only one branch's parameters from model state dict.
    
    Args:
        model_state_dict: Full model state dictionary
        branch_name: Name of branch ('detail_branch', 'semantic_branch', 'decoder', etc.)
        model: Optional model to get layer mapping (if None, uses heuristic matching)
    
    Returns:
        State dict containing only the specified branch's parameters
    """
    if model is not None:
        components = get_layer_names_by_component(model)
        target_layers = components.get(branch_name, [])
    else:
        # Use heuristic matching based on layer names
        target_keywords = {
            'detail_branch': ['detail'],
            'semantic_branch': ['stem', 'semantic'],
            'decoder': ['decoder', 'decode'],
            'auxiliary_heads': ['aux']
        }
        target_keywords_list = target_keywords.get(branch_name, [])
    
    branch_state_dict = {}
    
    for layer_name, param in model_state_dict.items():
        if model is not None:
            # Check if this layer belongs to the target component
            for target_layer in target_layers:
                if layer_name.startswith(target_layer):
                    branch_state_dict[layer_name] = param
                    break
        else:
            # Use keyword matching
            if any(keyword in layer_name.lower() for keyword in target_keywords_list):
                branch_state_dict[layer_name] = param
    
    return branch_state_dict


def compute_layer_wise_l2_distance(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute L2 distance between two models, per layer.
    
    Args:
        state_dict1: First model state dictionary
        state_dict2: Second model state dictionary
    
    Returns:
        Dictionary mapping layer names to L2 distances
    """
    distances = {}
    
    for layer_name in state_dict1.keys():
        if layer_name not in state_dict2:
            continue
        
        param1 = state_dict1[layer_name]
        param2 = state_dict2[layer_name]
        
        if param1.shape != param2.shape:
            continue
        
        l2_dist = torch.norm(param1 - param2).item()
        distances[layer_name] = l2_dist
    
    return distances


def compute_layer_wise_cosine_similarity(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute cosine similarity between two models, per layer.
    
    Args:
        state_dict1: First model state dictionary
        state_dict2: Second model state dictionary
    
    Returns:
        Dictionary mapping layer names to cosine similarities ([-1, 1])
    """
    similarities = {}
    
    for layer_name in state_dict1.keys():
        if layer_name not in state_dict2:
            continue
        
        param1 = state_dict1[layer_name].flatten()
        param2 = state_dict2[layer_name].flatten()
        
        if param1.shape != param2.shape:
            continue
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(param1.unsqueeze(0), param2.unsqueeze(0)).item()
        similarities[layer_name] = cos_sim
    
    return similarities


def compute_weight_magnitude(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute magnitude (L2 norm) of all weights.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Dictionary mapping layer names to their L2 magnitudes
    """
    magnitudes = {}
    
    for layer_name, param in state_dict.items():
        magnitude = torch.norm(param).item()
        magnitudes[layer_name] = magnitude
    
    return magnitudes


def get_model_layer_names_from_state_dict(state_dict: Dict) -> List[str]:
    """
    Extract unique base layer names from state dictionary.
    Removes parameter suffixes (.weight, .bias, .running_mean, etc.)
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        List of unique layer names
    """
    layer_names = set()
    
    for param_name in state_dict.keys():
        # Remove parameter suffixes
        parts = param_name.split('.')
        # Find where the parameter type starts (usually at the end)
        # Keep all but the last part if it's a known parameter type
        
        param_types = {'weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'}
        
        if parts[-1] in param_types:
            layer_name = '.'.join(parts[:-1])
        else:
            layer_name = param_name
        
        if layer_name:
            layer_names.add(layer_name)
    
    return sorted(list(layer_names))


def compute_weight_statistics(
    model_state_dicts: Dict[int, Dict],
    layer_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute statistics of weights across multiple models.
    Useful for understanding weight distribution across clients.
    
    Args:
        model_state_dicts: Dictionary mapping client IDs to their state dicts
        layer_name: Optional specific layer to analyze (if None, analyzes all)
    
    Returns:
        Dictionary with statistics
    """
    weights_list = []
    
    for client_id, state_dict in model_state_dicts.items():
        weights = extract_weights_by_layer(state_dict)
        
        if layer_name:
            # Only analyze specific layer
            for name, w in weights.items():
                if layer_name in name:
                    weights_list.append(w.flatten().numpy())
        else:
            # Analyze all weights
            for w in weights.values():
                weights_list.append(w.flatten().numpy())
    
    if not weights_list:
        return {}
    
    all_weights = np.concatenate(weights_list)
    
    return {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'median': float(np.median(all_weights))
    }
