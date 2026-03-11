"""Utilities for analyzing BatchNorm layer statistics in federated models."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def extract_bn_statistics_from_model(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract BatchNorm layer statistics from a live model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping BN layer names to their statistics
        {
            'layer.bn': {
                'running_mean': tensor,
                'running_var': tensor,
                'weight': tensor,
                'bias': tensor,
                'momentum': float
            }
        }
    """
    bn_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_stats[name] = {
                'running_mean': module.running_mean.clone().detach().cpu(),
                'running_var': module.running_var.clone().detach().cpu(),
                'weight': module.weight.data.clone().detach().cpu() if module.weight is not None else None,
                'bias': module.bias.data.clone().detach().cpu() if module.bias is not None else None,
                'momentum': module.momentum,
                'eps': module.eps,
                'num_batches_tracked': module.num_batches_tracked.clone().detach().cpu() if hasattr(module, 'num_batches_tracked') else None
            }
    
    return bn_stats


def extract_bn_statistics_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract BatchNorm statistics from a model state dictionary.
    
    Uses 'running_mean' as the anchor to identify all BN layers, ensuring no BN layer is missed.
    This is more robust than substring matching because running_mean is unique to BN layers.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Dictionary mapping BN layer names to their statistics
        {
            'layer.name.bn': {
                'running_mean': tensor,
                'running_var': tensor,
                'weight': tensor,
                'bias': tensor,
                'num_batches_tracked': tensor (optional)
            }
        }
    """
    bn_stats = {}
    
    # STEP 1: Find all BN layers by identifying keys with 'running_mean'
    # This is the most reliable way since running_mean only exists in BN layers
    bn_layer_names = set()
    
    for param_name in state_dict.keys():
        if 'running_mean' in param_name:
            # Extract base BN layer name
            # e.g., "module.detail_branch.conv_cur.0.bn.running_mean" 
            # -> "module.detail_branch.conv_cur.0.bn"
            base_name = param_name.rsplit('.running_mean', 1)[0]
            bn_layer_names.add(base_name)
    
    # STEP 2: For each BN layer, collect ALL its parameters
    for bn_layer in bn_layer_names:
        bn_stats[bn_layer] = {}
        
        # Look for all BN-related parameters
        param_suffixes = [
            'running_mean',
            'running_var',
            'weight',
            'bias',
            'num_batches_tracked'
        ]
        
        for suffix in param_suffixes:
            param_key = f"{bn_layer}.{suffix}"
            if param_key in state_dict:
                bn_stats[bn_layer][suffix] = state_dict[param_key]
    
    return bn_stats


def compute_bn_divergence(
    local_bn_stats: Dict[str, Dict[str, torch.Tensor]],
    global_bn_stats: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute divergence (L2 distance) between local and global BN statistics.
    
    Args:
        local_bn_stats: BN statistics from a local model
        global_bn_stats: BN statistics from the global model
    
    Returns:
        Dictionary mapping BN layer names to divergence metrics
        {
            'layer.bn': {
                'running_mean_l2': float,
                'running_var_l2': float,
                'weight_l2': float,
                'bias_l2': float,
                'mean_l2': float  # Average of all metrics
            }
        }
    """
    divergence = {}
    
    for layer_name in local_bn_stats.keys():
        if layer_name not in global_bn_stats:
            continue
        
        local_stats = local_bn_stats[layer_name]
        global_stats = global_bn_stats[layer_name]
        
        layer_divergence = {}
        values = []
        
        # Running mean divergence
        if 'running_mean' in local_stats and 'running_mean' in global_stats:
            running_mean_l2 = torch.norm(local_stats['running_mean'] - global_stats['running_mean']).item()
            layer_divergence['running_mean_l2'] = running_mean_l2
            values.append(running_mean_l2)
        
        # Running variance divergence
        if 'running_var' in local_stats and 'running_var' in global_stats:
            running_var_l2 = torch.norm(local_stats['running_var'] - global_stats['running_var']).item()
            layer_divergence['running_var_l2'] = running_var_l2
            values.append(running_var_l2)
        
        # Weight divergence
        if 'weight' in local_stats and 'weight' in global_stats:
            if local_stats['weight'] is not None and global_stats['weight'] is not None:
                weight_l2 = torch.norm(local_stats['weight'] - global_stats['weight']).item()
                layer_divergence['weight_l2'] = weight_l2
                values.append(weight_l2)
        
        # Bias divergence
        if 'bias' in local_stats and 'bias' in global_stats:
            if local_stats['bias'] is not None and global_stats['bias'] is not None:
                bias_l2 = torch.norm(local_stats['bias'] - global_stats['bias']).item()
                layer_divergence['bias_l2'] = bias_l2
                values.append(bias_l2)
        
        # Mean divergence
        if values:
            layer_divergence['mean_l2'] = float(np.mean(values))
            divergence[layer_name] = layer_divergence
    
    return divergence


def compute_bn_stability(
    bn_stats_per_round: Dict[int, Dict[str, Dict[str, torch.Tensor]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute stability (variance over rounds) of BN statistics.
    Measures how much BN statistics change across communication rounds.
    
    Args:
        bn_stats_per_round: Dictionary mapping round numbers to BN statistics
    
    Returns:
        Dictionary mapping BN layer names to stability metrics
    """
    stability = {}
    
    for layer_name in next(iter(bn_stats_per_round.values())).keys():
        layer_stats = {}
        
        # Collect running_mean across rounds
        running_means = []
        for round_num, stats in bn_stats_per_round.items():
            if layer_name in stats and 'running_mean' in stats[layer_name]:
                running_means.append(stats[layer_name]['running_mean'].numpy())
        
        if running_means:
            running_means = np.stack(running_means)
            layer_stats['running_mean_std'] = float(np.std(running_means, axis=0).mean())
        
        # Collect running_var across rounds
        running_vars = []
        for round_num, stats in bn_stats_per_round.items():
            if layer_name in stats and 'running_var' in stats[layer_name]:
                running_vars.append(stats[layer_name]['running_var'].numpy())
        
        if running_vars:
            running_vars = np.stack(running_vars)
            layer_stats['running_var_std'] = float(np.std(running_vars, axis=0).mean())
        
        if layer_stats:
            stability[layer_name] = layer_stats
    
    return stability


def compute_bn_shift_magnitude(
    bn_stats_before: Dict[str, Dict[str, torch.Tensor]],
    bn_stats_after: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute the magnitude of shift in BN statistics (useful for before/after aggregation).
    
    Args:
        bn_stats_before: BN statistics before some event (e.g., before aggregation)
        bn_stats_after: BN statistics after the event
    
    Returns:
        Dictionary mapping layer names to shift magnitudes
    """
    shifts = {}
    
    for layer_name in bn_stats_before.keys():
        if layer_name not in bn_stats_after:
            continue
        
        before = bn_stats_before[layer_name]
        after = bn_stats_after[layer_name]
        
        layer_shifts = {}
        
        # Running mean shift
        if 'running_mean' in before and 'running_mean' in after:
            mean_shift = torch.norm(after['running_mean'] - before['running_mean']).item()
            layer_shifts['running_mean_shift'] = mean_shift
        
        # Running variance shift
        if 'running_var' in before and 'running_var' in after:
            var_shift = torch.norm(after['running_var'] - before['running_var']).item()
            layer_shifts['running_var_shift'] = var_shift
        
        if layer_shifts:
            shifts[layer_name] = layer_shifts
    
    return shifts


def get_bn_layer_names(model: nn.Module) -> List[str]:
    """
    Get all BatchNorm layer names in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of BN layer names
    """
    bn_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(name)
    
    return bn_layers


def compute_bn_variance_per_client(
    client_bn_stats_dict: Dict[int, Dict[str, Dict[str, torch.Tensor]]]
) -> Dict[str, float]:
    """
    Compute variance of BN statistics across all clients (for a given round).
    Shows how much clients diverge in their BN statistics.
    
    Args:
        client_bn_stats_dict: Dictionary mapping client IDs to their BN statistics
    
    Returns:
        Dictionary mapping layer names to their cross-client variance
    """
    variance_dict = {}
    
    for layer_name in next(iter(client_bn_stats_dict.values())).keys():
        client_running_means = []
        
        for client_id, stats in client_bn_stats_dict.items():
            if layer_name in stats and 'running_mean' in stats[layer_name]:
                client_running_means.append(stats[layer_name]['running_mean'].numpy())
        
        if client_running_means:
            client_running_means = np.stack(client_running_means)
            # Compute variance across clients for each element, then average
            variance = float(np.var(client_running_means, axis=0).mean())
            variance_dict[layer_name] = variance
    
    return variance_dict


def normalize_bn_statistics(
    bn_stats: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Normalize BN statistics to [0, 1] range for comparison.
    
    Args:
        bn_stats: BN statistics dictionary
    
    Returns:
        Normalized BN statistics
    """
    normalized = {}
    
    for layer_name, stats in bn_stats.items():
        normalized[layer_name] = {}
        
        for stat_name, value in stats.items():
            if isinstance(value, torch.Tensor):
                # Normalize to [0, 1]
                min_val = value.min()
                max_val = value.max()
                if max_val > min_val:
                    normalized[layer_name][stat_name] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[layer_name][stat_name] = torch.zeros_like(value)
            else:
                # Keep non-tensor values as is
                normalized[layer_name][stat_name] = value
    
    return normalized
