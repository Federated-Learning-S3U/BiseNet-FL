import json
import os
from pathlib import Path


def log_client_partition(log_file_path: str, server_round: int, partition_id: int) -> None:
    """Log client partition selection to JSON file.
    
    Appends partition_id to the list of clients for the given round.
    If round doesn't exist, creates a new entry with the partition_id.
    
    Parameters
    ----------
    log_file_path : str
        Path to the JSON log file
    server_round : int
        Communication round number
    partition_id : int
        Client partition ID to log
    """
    # Ensure directory exists
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing data or create empty dict
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    else:
        data = {}
    
    # Convert round to string key
    round_key = str(server_round)
    
    # Append or create entry
    if round_key in data:
        if isinstance(data[round_key], list):
            if partition_id not in data[round_key]:
                data[round_key].append(partition_id)
        else:
            # Handle case where value might not be a list
            data[round_key] = [data[round_key], partition_id]
    else:
        data[round_key] = [partition_id]
    
    # Write back to file
    with open(log_file_path, "w") as f:
        json.dump(data, f, indent=2)