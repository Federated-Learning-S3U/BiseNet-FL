"""Utilities for analyzing model similarity and divergence across clients."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from fl_cityscapes_bisenetv2.utils.weight_utils import (
    compute_layer_wise_cosine_similarity,
    compute_layer_wise_l2_distance,
    flatten_model_weights,
    extract_weights_by_layer
)


def cosine_similarity_matrices(
    models_dict: Dict[int, Dict],
    exclude_bn_stats: bool = True
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between all models.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        NxN similarity matrix where N is number of models
    """
    client_ids = sorted(models_dict.keys())
    n_clients = len(client_ids)
    
    # Compute flattened weights for all clients
    flattened_weights = {}
    for client_id in client_ids:
        weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
        flattened = []
        for param in weights.values():
            flattened.append(param.flatten())
        flattened_weights[client_id] = torch.cat(flattened)
    
    # Compute similarity matrix
    similarity_matrix = np.zeros((n_clients, n_clients))
    
    for i, client_i in enumerate(client_ids):
        for j, client_j in enumerate(client_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    flattened_weights[client_i].unsqueeze(0),
                    flattened_weights[client_j].unsqueeze(0)
                ).item()
                similarity_matrix[i, j] = cos_sim
    
    return similarity_matrix, client_ids


def branch_similarity_matrices(
    models_dict: Dict[int, Dict],
    branch_names: Optional[List[str]] = None,
    model: Optional[nn.Module] = None
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    """
    Compute pairwise cosine similarity matrices per branch.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        branch_names: List of branch names to analyze (default: detail_branch, semantic_branch, decoder)
        model: Optional model for layer mapping
    
    Returns:
        Dictionary mapping branch_name -> (similarity_matrix, client_ids)
    """
    from fl_cityscapes_bisenetv2.utils.weight_utils import get_branch_state_dict
    
    if branch_names is None:
        branch_names = ['detail_branch', 'semantic_branch', 'decoder']
    
    client_ids = sorted(models_dict.keys())
    results = {}
    
    for branch_name in branch_names:
        # Get branch weights for all clients
        branch_weights = {}
        for client_id in client_ids:
            branch_state = get_branch_state_dict(models_dict[client_id], branch_name, model)
            
            flattened = []
            for param in branch_state.values():
                flattened.append(param.flatten())
            
            if flattened:
                branch_weights[client_id] = torch.cat(flattened)
        
        # Compute similarity matrix for this branch
        n_clients = len(client_ids)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif client_i in branch_weights and client_j in branch_weights:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        branch_weights[client_i].unsqueeze(0),
                        branch_weights[client_j].unsqueeze(0)
                    ).item()
                    similarity_matrix[i, j] = cos_sim
        
        results[branch_name] = (similarity_matrix, client_ids)
    
    return results


def l2_distance_matrix(
    models_dict: Dict[int, Dict],
    exclude_bn_stats: bool = True
) -> np.ndarray:
    """
    Compute pairwise L2 distance matrix between all models.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        NxN distance matrix where N is number of models
    """
    client_ids = sorted(models_dict.keys())
    n_clients = len(client_ids)
    
    # Compute flattened weights for all clients
    flattened_weights = {}
    for client_id in client_ids:
        weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
        flattened = []
        for param in weights.values():
            flattened.append(param.flatten())
        flattened_weights[client_id] = torch.cat(flattened)
    
    # Compute distance matrix
    distance_matrix = np.zeros((n_clients, n_clients))
    
    for i, client_i in enumerate(client_ids):
        for j, client_j in enumerate(client_ids):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                # L2 distance
                l2_dist = torch.norm(flattened_weights[client_i] - flattened_weights[client_j]).item()
                distance_matrix[i, j] = l2_dist
    
    return distance_matrix, client_ids


def layer_wise_divergence(
    models_dict: Dict[int, Dict],
    exclude_bn_stats: bool = True
) -> Dict[str, float]:
    """
    Compute average divergence per layer across all client models.
    Higher divergence = more distinct learning across clients.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Dictionary mapping layer_name -> average divergence score
    """
    client_ids = sorted(models_dict.keys())
    n_clients = len(client_ids)
    
    if n_clients < 2:
        return {}
    
    # Collect all layer names
    all_layers = set()
    for state_dict in models_dict.values():
        for layer_name in extract_weights_by_layer(state_dict, exclude_bn_stats).keys():
            all_layers.add(layer_name)
    
    layer_divergence = {}
    
    for layer_name in all_layers:
        # Collect weights for this layer across all clients
        layer_weights = []
        for client_id in client_ids:
            weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
            if layer_name in weights:
                layer_weights.append(weights[layer_name].flatten().numpy())
        
        if not layer_weights:
            continue
        
        # Compute divergence: average pairwise distance
        divergences = []
        for i in range(len(layer_weights)):
            for j in range(i + 1, len(layer_weights)):
                dist = np.linalg.norm(layer_weights[i] - layer_weights[j])
                divergences.append(dist)
        
        if divergences:
            layer_divergence[layer_name] = float(np.mean(divergences))
    
    return layer_divergence


def model_to_global_distance(
    local_models: Dict[int, Dict],
    global_model: Dict,
    exclude_bn_stats: bool = True
) -> Dict[int, float]:
    """
    Compute L2 distance from each local model to the global model.
    
    Args:
        local_models: Dictionary mapping client_id -> state_dict
        global_model: Global model state_dict
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Dictionary mapping client_id -> distance to global model
    """
    distances = {}
    
    for client_id, local_state in local_models.items():
        local_weights = extract_weights_by_layer(local_state, exclude_bn_stats)
        global_weights = extract_weights_by_layer(global_model, exclude_bn_stats)
        
        dist_dict = compute_layer_wise_l2_distance(local_weights, global_weights)
        
        if dist_dict:
            # L2 of all layer distances
            total_dist = float(np.sqrt(sum(d**2 for d in dist_dict.values())))
            distances[client_id] = total_dist
    
    return distances


def identify_outlier_clients(
    models_dict: Dict[int, Dict],
    threshold_std: float = 2.0,
    exclude_bn_stats: bool = True
) -> Dict[int, float]:
    """
    Identify client models that diverge significantly from the cluster.
    Uses distance to centroid as the outlier metric.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        threshold_std: Number of standard deviations for outlier detection
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Dictionary mapping client_id -> distance score (only outliers)
    """
    client_ids = sorted(models_dict.keys())
    
    if len(client_ids) < 3:
        return {}
    
    # Compute flattened weights
    flattened_weights = {}
    for client_id in client_ids:
        weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
        flattened = []
        for param in weights.values():
            flattened.append(param.flatten())
        flattened_weights[client_id] = torch.cat(flattened).numpy()
    
    # Compute centroid
    centroid = np.mean([w for w in flattened_weights.values()], axis=0)
    
    # Compute distances to centroid
    distances = {}
    for client_id in client_ids:
        dist = np.linalg.norm(flattened_weights[client_id] - centroid)
        distances[client_id] = dist
    
    # Find outliers
    distances_array = np.array(list(distances.values()))
    mean_dist = np.mean(distances_array)
    std_dist = np.std(distances_array)
    
    outliers = {}
    for client_id, dist in distances.items():
        if dist > mean_dist + threshold_std * std_dist:
            outliers[client_id] = dist
    
    return outliers


def cluster_clients_by_similarity(
    models_dict: Dict[int, Dict],
    n_clusters: Optional[int] = None,
    exclude_bn_stats: bool = True
) -> Dict[int, int]:
    """
    Cluster clients based on model weight similarity using k-means.
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        n_clusters: Number of clusters (default: sqrt(n_clients))
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Dictionary mapping client_id -> cluster_id
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("sklearn required for clustering. Skipping.")
        return {}
    
    client_ids = sorted(models_dict.keys())
    
    if len(client_ids) < 2:
        return {client_ids[0]: 0} if client_ids else {}
    
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(len(client_ids))))
    
    # Compute flattened weights
    flattened_list = []
    for client_id in client_ids:
        weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
        flattened = []
        for param in weights.values():
            flattened.append(param.flatten().numpy())
        flattened_list.append(np.concatenate(flattened))
    
    # Cluster
    kmeans = KMeans(n_clusters=min(n_clusters, len(client_ids)), random_state=42)
    labels = kmeans.fit_predict(np.array(flattened_list))
    
    clustering = {}
    for client_id, label in zip(client_ids, labels):
        clustering[client_id] = int(label)
    
    return clustering


def compute_model_consensus_distance(
    models_dict: Dict[int, Dict],
    exclude_bn_stats: bool = True
) -> float:
    """
    Compute average distance of all models from their centroid.
    Lower value = higher consensus (clients agree more).
    
    Args:
        models_dict: Dictionary mapping client_id -> state_dict
        exclude_bn_stats: If True, exclude BN running statistics
    
    Returns:
        Average distance to centroid
    """
    client_ids = sorted(models_dict.keys())
    
    if len(client_ids) < 2:
        return 0.0
    
    # Compute flattened weights
    flattened_weights = {}
    for client_id in client_ids:
        weights = extract_weights_by_layer(models_dict[client_id], exclude_bn_stats)
        flattened = []
        for param in weights.values():
            flattened.append(param.flatten().numpy())
        flattened_weights[client_id] = np.concatenate(flattened)
    
    # Compute centroid
    centroid = np.mean([w for w in flattened_weights.values()], axis=0)
    
    # Average distance to centroid
    distances = [np.linalg.norm(w - centroid) for w in flattened_weights.values()]
    
    return float(np.mean(distances))
