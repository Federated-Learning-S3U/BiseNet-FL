"""Utilities for saving and managing local and global model checkpoints."""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional


def create_checkpoint_dirs(base_path: str, partition_name: str) -> Dict[str, str]:
    """
    Create checkpoint directory structure for a partition.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition (e.g., 'iid_partitions', 'non_iid_partitions')
    
    Returns:
        Dictionary with paths to local_models, global_models, and metadata directories
    """
    partition_path = os.path.join(base_path, partition_name)
    paths = {
        'local_models': os.path.join(partition_path, 'local_models'),
        'global_models': os.path.join(partition_path, 'global_models'),
        'metadata': partition_path
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def save_local_model(
    model_state_dict: Dict,
    base_path: str,
    partition_name: str,
    server_round: int,
    client_id: int
) -> str:
    """
    Save a local model checkpoint.
    
    Args:
        model_state_dict: Model state dictionary
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
        client_id: Client partition ID
    
    Returns:
        Path where model was saved
    """
    paths = create_checkpoint_dirs(base_path, partition_name)
    
    # Create round directory
    round_dir = os.path.join(paths['local_models'], str(server_round))
    os.makedirs(round_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(round_dir, f'{client_id}.pt')
    torch.save(model_state_dict, model_path)
    
    return model_path


def save_global_model(
    model_state_dict: Dict,
    base_path: str,
    partition_name: str,
    server_round: int
) -> str:
    """
    Save the global model checkpoint after aggregation.
    
    Args:
        model_state_dict: Global model state dictionary
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
    
    Returns:
        Path where model was saved
    """
    paths = create_checkpoint_dirs(base_path, partition_name)
    
    # Save global model
    model_path = os.path.join(paths['global_models'], f'{server_round}.pt')
    torch.save(model_state_dict, model_path)
    
    return model_path


def update_client_metadata(
    base_path: str,
    partition_name: str,
    server_round: int,
    client_ids: List[int]
) -> None:
    """
    Update metadata file tracking which clients participated in each round.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
        client_ids: List of client IDs that participated in this round
    """
    paths = create_checkpoint_dirs(base_path, partition_name)
    
    metadata_file = os.path.join(paths['metadata'], 'client_metadata.json')
    
    # Load existing metadata or create new
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update with current round
    metadata[str(server_round)] = client_ids
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_local_model(
    base_path: str,
    partition_name: str,
    server_round: int,
    client_id: int
) -> Dict:
    """
    Load a locally saved model checkpoint.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
        client_id: Client partition ID
    
    Returns:
        Model state dictionary
    """
    model_path = os.path.join(
        base_path, partition_name, 'local_models',
        str(server_round), f'{client_id}.pt'
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return torch.load(model_path, map_location='cpu')


def load_global_model(
    base_path: str,
    partition_name: str,
    server_round: int
) -> Dict:
    """
    Load a global model checkpoint after aggregation.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
    
    Returns:
        Model state dictionary
    """
    model_path = os.path.join(
        base_path, partition_name, 'global_models',
        f'{server_round}.pt'
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return torch.load(model_path, map_location='cpu')


def load_client_metadata(
    base_path: str,
    partition_name: str
) -> Dict[str, List[int]]:
    """
    Load metadata tracking client participation in each round.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
    
    Returns:
        Dictionary mapping round numbers (as strings) to list of client IDs
    """
    metadata_file = os.path.join(base_path, partition_name, 'client_metadata.json')
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def get_available_rounds(
    base_path: str,
    partition_name: str
) -> List[int]:
    """
    Get list of all available communication rounds for a partition.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
    
    Returns:
        Sorted list of round numbers
    """
    local_models_path = os.path.join(base_path, partition_name, 'local_models')
    
    if not os.path.exists(local_models_path):
        return []
    
    rounds = []
    for item in os.listdir(local_models_path):
        if os.path.isdir(os.path.join(local_models_path, item)):
            try:
                rounds.append(int(item))
            except ValueError:
                continue
    
    return sorted(rounds)


def get_clients_in_round(
    base_path: str,
    partition_name: str,
    server_round: int
) -> List[int]:
    """
    Get list of clients that participated in a specific round.
    
    Args:
        base_path: Base directory (e.g., 'res')
        partition_name: Name of the partition
        server_round: Server communication round
    
    Returns:
        List of client IDs
    """
    round_path = os.path.join(base_path, partition_name, 'local_models', str(server_round))
    
    if not os.path.exists(round_path):
        return []
    
    client_ids = []
    for filename in os.listdir(round_path):
        if filename.endswith('.pt'):
            try:
                client_id = int(filename.replace('.pt', ''))
                client_ids.append(client_id)
            except ValueError:
                continue
    
    return sorted(client_ids)
