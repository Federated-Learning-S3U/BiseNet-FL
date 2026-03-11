"""Utilities for analyzing branch-specific learning dynamics in BiSeNetV2."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from fl_cityscapes_bisenetv2.utils.weight_utils import (
    get_branch_state_dict,
    compute_layer_wise_cosine_similarity,
    compute_layer_wise_l2_distance,
    flatten_model_weights,
    compute_weight_magnitude
)


def compare_branch_learning_speed(
    local_models_per_round: Dict[int, Dict[int, Dict]],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[int, float]:
    """
    Compare learning speed (weight change magnitude) of a specific branch across rounds.
    
    Args:
        local_models_per_round: Dictionary mapping round -> client_id -> state_dict
        branch_name: Name of branch ('detail_branch', 'semantic_branch', etc.)
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping rounds to average weight change magnitude for the branch
    """
    learning_speed = {}
    
    rounds = sorted(local_models_per_round.keys())
    if len(rounds) < 2:
        return learning_speed
    
    for i in range(1, len(rounds)):
        prev_round = rounds[i - 1]
        curr_round = rounds[i]
        
        if prev_round not in local_models_per_round or curr_round not in local_models_per_round:
            continue
        
        prev_models = local_models_per_round[prev_round]
        curr_models = local_models_per_round[curr_round]
        
        # Get common clients
        common_clients = set(prev_models.keys()) & set(curr_models.keys())
        if not common_clients:
            continue
        
        changes = []
        for client_id in common_clients:
            prev_state = get_branch_state_dict(prev_models[client_id], branch_name, model)
            curr_state = get_branch_state_dict(curr_models[client_id], branch_name, model)
            
            # Compute L2 distance for this branch
            distances = compute_layer_wise_l2_distance(prev_state, curr_state)
            if distances:
                avg_distance = float(np.mean(list(distances.values())))
                changes.append(avg_distance)
        
        if changes:
            learning_speed[curr_round] = float(np.mean(changes))
    
    return learning_speed


def branch_aggregation_impact(
    local_models_before_agg: Dict[int, Dict],
    global_model_after_agg: Dict,
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[int, Dict[str, float]]:
    """
    Measure how much each client's branch weights change due to aggregation.
    
    Args:
        local_models_before_agg: Dictionary mapping client_id -> state_dict (before aggregation)
        global_model_after_agg: Global model state dict (after aggregation)
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping client_id to impact metrics
    """
    impact = {}
    
    for client_id, local_state in local_models_before_agg.items():
        local_branch = get_branch_state_dict(local_state, branch_name, model)
        global_branch = get_branch_state_dict(global_model_after_agg, branch_name, model)
        
        distances = compute_layer_wise_l2_distance(local_branch, global_branch)
        
        if distances:
            impact[client_id] = {
                'mean_distance': float(np.mean(list(distances.values()))),
                'max_distance': float(np.max(list(distances.values()))),
                'min_distance': float(np.min(list(distances.values())))
            }
    
    return impact


def branch_client_variance(
    models_per_client: Dict[int, Dict],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Compute how much a specific branch's weights vary across different clients.
    Higher variance = more diverse learning across clients.
    
    Args:
        models_per_client: Dictionary mapping client_id -> state_dict
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary with variance metrics for the branch
    """
    # Get branch weights for all clients
    branch_weights_list = []
    
    for client_id, state_dict in models_per_client.items():
        branch_state = get_branch_state_dict(state_dict, branch_name, model)
        
        # Flatten and collect
        flattened = []
        for param in branch_state.values():
            flattened.append(param.flatten())
        
        if flattened:
            all_weights = torch.cat(flattened)
            branch_weights_list.append(all_weights.numpy())
    
    if not branch_weights_list:
        return {}
    
    # Stack all client weights
    all_weights = np.concatenate(branch_weights_list)
    
    return {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'variance': float(np.var(all_weights)),
        'max': float(np.max(all_weights)),
        'min': float(np.min(all_weights))
    }


def branch_weight_magnitude(
    models_dict: Dict[int, Dict],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[int, float]:
    """
    Compute weight magnitude (L2 norm) of a specific branch for each client.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping client_id to branch weight magnitude
    """
    magnitudes = {}
    
    for client_id, state_dict in models_dict.items():
        branch_state = get_branch_state_dict(state_dict, branch_name, model)
        
        total_magnitude = 0.0
        for param in branch_state.values():
            total_magnitude += torch.norm(param).item()
        
        magnitudes[client_id] = total_magnitude
    
    return magnitudes


def compare_branch_cosine_similarity(
    models_dict1: Dict[int, Dict],
    models_dict2: Dict[int, Dict],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Compare cosine similarity of a specific branch between two sets of models.
    Useful for comparing IID vs Non-IID branches.
    
    Args:
        models_dict1: First set of models (e.g., IID clients at round 5)
        models_dict2: Second set of models (e.g., Non-IID clients at round 5)
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary with similarity metrics
    """
    similarities = {}
    
    # Get common clients
    common_clients = set(models_dict1.keys()) & set(models_dict2.keys())
    
    if not common_clients:
        return similarities
    
    cosine_sims = []
    for client_id in common_clients:
        state1 = get_branch_state_dict(models_dict1[client_id], branch_name, model)
        state2 = get_branch_state_dict(models_dict2[client_id], branch_name, model)
        
        cos_sims = compute_layer_wise_cosine_similarity(state1, state2)
        if cos_sims:
            cosine_sims.extend(cos_sims.values())
    
    if cosine_sims:
        similarities['mean_cosine_similarity'] = float(np.mean(cosine_sims))
        similarities['min_cosine_similarity'] = float(np.min(cosine_sims))
        similarities['max_cosine_similarity'] = float(np.max(cosine_sims))
        similarities['std_cosine_similarity'] = float(np.std(cosine_sims))
    
    return similarities


def branch_convergence_rate(
    local_models_per_round: Dict[int, Dict[int, Dict]],
    global_models_per_round: Dict[int, Dict],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[int, float]:
    """
    Measure convergence rate of a branch by tracking average distance to global model.
    Faster convergence = distance decreases faster.
    
    Args:
        local_models_per_round: Dictionary mapping round -> client_id -> state_dict
        global_models_per_round: Dictionary mapping round -> global_state_dict
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping round to average distance from clients to global model
    """
    convergence = {}
    
    for round_num in sorted(local_models_per_round.keys()):
        if round_num not in global_models_per_round:
            continue
        
        local_models = local_models_per_round[round_num]
        global_state = global_models_per_round[round_num]
        
        distances = []
        for client_id, local_state in local_models.items():
            local_branch = get_branch_state_dict(local_state, branch_name, model)
            global_branch = get_branch_state_dict(global_state, branch_name, model)
            
            dist_dict = compute_layer_wise_l2_distance(local_branch, global_branch)
            if dist_dict:
                distances.append(float(np.mean(list(dist_dict.values()))))
        
        if distances:
            convergence[round_num] = float(np.mean(distances))
    
    return convergence


def branch_layer_wise_convergence(
    local_models_per_round: Dict[int, Dict[int, Dict]],
    global_models_per_round: Dict[int, Dict],
    branch_name: str,
    model: Optional[nn.Module] = None
) -> Dict[int, Dict[str, float]]:
    """
    Measure convergence rate per layer within a branch.
    Shows which layers converge fastest/slowest.
    
    Args:
        local_models_per_round: Dictionary mapping round -> client_id -> state_dict
        global_models_per_round: Dictionary mapping round -> global_state_dict
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping round -> layer_name -> average distance
    """
    layer_convergence = {}
    
    for round_num in sorted(local_models_per_round.keys()):
        if round_num not in global_models_per_round:
            continue
        
        local_models = local_models_per_round[round_num]
        global_state = global_models_per_round[round_num]
        
        layer_distances = {}
        
        for client_id, local_state in local_models.items():
            local_branch = get_branch_state_dict(local_state, branch_name, model)
            global_branch = get_branch_state_dict(global_state, branch_name, model)
            
            dist_dict = compute_layer_wise_l2_distance(local_branch, global_branch)
            
            for layer, dist in dist_dict.items():
                if layer not in layer_distances:
                    layer_distances[layer] = []
                layer_distances[layer].append(dist)
        
        # Average across clients for each layer
        round_layer_convergence = {}
        for layer, dists in layer_distances.items():
            round_layer_convergence[layer] = float(np.mean(dists))
        
        layer_convergence[round_num] = round_layer_convergence
    
    return layer_convergence


def get_branch_parameter_count(
    model_state_dict: Dict,
    branch_name: str,
    model: Optional[nn.Module] = None
) -> int:
    """
    Count total number of parameters in a specific branch.
    
    Args:
        model_state_dict: Model state dictionary
        branch_name: Name of branch to analyze
        model: Optional model for layer mapping
    
    Returns:
        Total parameter count for the branch
    """
    branch_state = get_branch_state_dict(model_state_dict, branch_name, model)
    
    total_params = 0
    for param in branch_state.values():
        total_params += param.numel()
    
    return total_params
