"""
SiloBN (Siloed Batch Normalization) Federated Learning Strategy.

In SiloBN, each client maintains its own local batch normalization statistics
(running_mean, running_var), while all learnable parameters including BN's
gamma (weight) and beta (bias) are aggregated across clients. This helps 
handle data heterogeneity by allowing each client to adapt its BN statistics 
to its local data distribution while still learning shared affine transformations.

Reference:
    Andreux, M., du Terrail, J. O., Beguier, C., & Tramel, E. W. (2020).
    Siloed Federated Learning for Multi-centric Histopathology Datasets.
    In Domain Adaptation and Representation Transfer, and Distributed and
    Collaborative Learning (pp. 129-139). Springer.
"""

import json
from logging import INFO
from typing import Iterable
import numpy as np

from flwr.serverapp import Grid
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    log,
)
from .CustomFedAvg import CustomFedAvg


def is_bn_statistic(param_name: str) -> bool:
    """
    Check if a parameter is a BatchNorm STATISTIC (not learnable).
    
    Only these should be kept local:
    - 'running_mean' - Running mean estimate (non-learnable)
    - 'running_var' - Running variance estimate (non-learnable)
    - 'num_batches_tracked' - Counter for batches seen (non-learnable)
    
    Note: BN weight (gamma) and bias (beta) ARE learnable and should be aggregated.
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        True if parameter is a BN statistic (should be kept local)
    """
    # Only match non-learnable BN statistics
    bn_statistic_keywords = [
        'running_mean',
        'running_var', 
        'num_batches_tracked'
    ]
    
    param_lower = param_name.lower()
    
    for keyword in bn_statistic_keywords:
        if keyword in param_lower:
            return True
    
    return False


def split_bn_statistics(arrays: ArrayRecord) -> tuple[dict, dict]:
    """
    Split parameters into aggregatable params and local BN statistics.
    
    Args:
        arrays: ArrayRecord containing all model parameters
        
    Returns:
        Tuple of (aggregatable_params_dict, bn_statistics_dict)
        - aggregatable_params: Conv, FC, AND BN gamma/beta (all learnable params)
        - bn_statistics: Only running_mean, running_var, num_batches_tracked
    """
    aggregatable_params = {}
    bn_statistics = {}
    
    for key in arrays.keys():
        if is_bn_statistic(key):
            bn_statistics[key] = arrays[key]
        else:
            aggregatable_params[key] = arrays[key]
    
    return aggregatable_params, bn_statistics


# Keep old functions for backward compatibility
def is_bn_param(param_name: str) -> bool:
    """Backward compatible: Check if param is BN statistic."""
    return is_bn_statistic(param_name)


def split_bn_params(arrays: ArrayRecord) -> tuple[dict, dict]:
    """Backward compatible: Split into aggregatable and local params."""
    return split_bn_statistics(arrays)


class CustomFedSiloBN(CustomFedAvg):
    """
    Federated Learning with Siloed Batch Normalization (SiloBN) strategy.
    
    This strategy keeps only batch normalization STATISTICS local to each client
    (running_mean, running_var), while aggregating ALL learnable parameters
    including BN's gamma (weight) and beta (bias).
    
    Key features:
    - Learnable parameters (Conv, FC, BN gamma/beta): Aggregated using weighted averaging
    - BN statistics (running_mean, running_var): Kept local on each client
    - Server maintains averaged BN statistics for evaluation purposes
    
    What gets aggregated:
    - All convolutional layer weights and biases
    - All fully connected layer weights and biases  
    - BN weight (gamma) - learnable scale parameter
    - BN bias (beta) - learnable shift parameter
    
    What stays local:
    - running_mean - running mean estimate
    - running_var - running variance estimate
    - num_batches_tracked - batch counter
    
    Attributes:
        server_bn_statistics: BN statistics maintained on server for evaluation
    """
    
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_bn_statistics: dict | None = None
        self.initial_bn_statistics: dict | None = None
    
    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedSiloBN settings:")
        log(INFO, "\t│\t└── BN statistics (running_mean/var): Local (not aggregated)")
        log(INFO, "\t│\t└── BN gamma/beta: Aggregated (learnable)")
        log(INFO, "\t│\t└── Non-BN layers: Aggregated (FedAvg)")
        super().summary()
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        
        # On first round, store initial BN statistics for server-side evaluation
        if server_round == 1 and self.initial_bn_statistics is None:
            _, bn_stats = split_bn_statistics(arrays)
            self.initial_bn_statistics = bn_stats
            self.server_bn_statistics = bn_stats
            log(INFO, f"[SiloBN] Stored {len(bn_stats)} BN statistics for server evaluation")
        
        # Call parent to handle LR decay and standard config
        return super().configure_train(server_round, arrays, config, grid)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Aggregate all learnable parameters, keep only BN statistics local.
        
        The aggregation process:
        1. Extract learnable params (including BN gamma/beta) from each client
        2. Aggregate learnable params using weighted averaging
        3. Update server's BN statistics by averaging (for evaluation only)
        4. Combine aggregated params with server's BN statistics
        """
        
        # Convert replies to a list to allow multiple iterations
        replies_list = list(replies)
        
        if not replies_list:
            return None, None
        
        # Collect arrays and weights from replies
        client_arrays_list = []
        client_weights = []
        
        for reply in replies_list:
            if reply.has_error():
                continue
            
            arrays = reply.content.get("arrays")
            metrics = reply.content.get("metrics")
            
            if arrays is None:
                continue
            
            # Get number of examples for weighting
            num_examples = 1
            if metrics is not None:
                num_examples = metrics.get("num-examples", 1)
            
            client_arrays_list.append(arrays)
            client_weights.append(num_examples)
        
        if not client_arrays_list:
            return None, None
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Split first client's arrays to get parameter keys
        first_arrays = client_arrays_list[0]
        aggregatable_keys = []  # Conv, FC, AND BN gamma/beta
        bn_statistic_keys = []  # Only running_mean, running_var, num_batches_tracked
        
        for key in first_arrays.keys():
            if is_bn_statistic(key):
                bn_statistic_keys.append(key)
            else:
                aggregatable_keys.append(key)
        
        log(INFO, f"[SiloBN] Round {server_round}: Aggregating {len(aggregatable_keys)} params "
                  f"(including BN gamma/beta), keeping {len(bn_statistic_keys)} BN statistics local")
        
        # Aggregate all learnable parameters (including BN gamma and beta)
        aggregated_params = {}
        
        for key in aggregatable_keys:
            weighted_sum = None
            
            for client_arrays, weight in zip(client_arrays_list, normalized_weights):
                param_array = client_arrays[key]
                param_np = Array.numpy(param_array)
                
                if weighted_sum is None:
                    weighted_sum = weight * param_np
                else:
                    weighted_sum += weight * param_np
            
            aggregated_params[key] = Array.from_numpy_ndarray(weighted_sum)
        
        # Update server BN statistics by averaging (for server-side evaluation only)
        # This gives a reasonable estimate for central evaluation
        if bn_statistic_keys:
            updated_bn_statistics = {}
            
            for key in bn_statistic_keys:
                weighted_sum = None
                
                for client_arrays, weight in zip(client_arrays_list, normalized_weights):
                    if key in client_arrays.keys():
                        param_array = client_arrays[key]
                        param_np = Array.numpy(param_array)
                        
                        if weighted_sum is None:
                            weighted_sum = weight * param_np
                        else:
                            weighted_sum += weight * param_np
                
                if weighted_sum is not None:
                    updated_bn_statistics[key] = Array.from_numpy_ndarray(weighted_sum)
                elif self.server_bn_statistics and key in self.server_bn_statistics:
                    updated_bn_statistics[key] = self.server_bn_statistics[key]
            
            self.server_bn_statistics = updated_bn_statistics
        
        # Combine aggregated params with server BN statistics for the global model
        combined_params = {**aggregated_params}
        
        if self.server_bn_statistics is not None:
            combined_params.update(self.server_bn_statistics)
        
        aggregated_arrays = ArrayRecord(combined_params)
        
        # Aggregate metrics (same as FedAvg)
        aggregated_metrics = self._aggregate_metrics(replies_list)
        
        return aggregated_arrays, aggregated_metrics
    
    def _aggregate_metrics(self, replies: list[Message]) -> MetricRecord | None:
        """Aggregate metrics from client replies."""
        total_examples = 0
        weighted_loss = 0.0
        
        for reply in replies:
            if reply.has_error():
                continue
            
            metrics = reply.content.get("metrics")
            if metrics is None:
                continue
            
            num_examples = metrics.get("num-examples", 0)
            train_loss = metrics.get("train_loss", 0.0)
            
            total_examples += num_examples
            weighted_loss += train_loss * num_examples
        
        if total_examples == 0:
            return None
        
        avg_loss = weighted_loss / total_examples
        
        return MetricRecord({
            "train_loss": avg_loss,
            "num-examples": total_examples,
        })
