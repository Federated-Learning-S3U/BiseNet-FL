import json

import numpy as np


def aggregate_client_metrics(json_path):
    """
    Compute global mean and std from per-client mean/std and num_samples.

    Args:
        json_path (str): Path to the JSON file with client stats.
                         Each client must have:
                         - "num_samples": int
                         - "data_metrics": {"mean": [...], "std": [...]}

    Returns:
        global_mean (np.ndarray): per-channel mean
        global_std  (np.ndarray): per-channel std
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    client_stats = []
    for cid, info in data.items():
        n = info["num_samples"]
        mean = np.array(info["data_metrics"]["mean"])
        std = np.array(info["data_metrics"]["std"])
        client_stats.append({"n": n, "mean": mean, "std": std})

    total_n = sum(c["n"] for c in client_stats)
    if total_n == 0:
        raise ValueError("Total number of samples is zero.")

    # Global mean: weighted average
    global_mean = sum(c["n"] * c["mean"] for c in client_stats) / total_n

    # Global variance: weighted combination of local variance + squared mean
    global_var = (
        sum(c["n"] * (c["std"] ** 2 + c["mean"] ** 2) for c in client_stats) / total_n
        - global_mean**2
    )
    global_std = np.sqrt(global_var)

    return global_mean, global_std
