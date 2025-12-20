"""
SiloBN (Siloed Batch Normalization) Federated Learning Strategy.

In SiloBN, each client maintains its own local batch normalization statistics
(running_mean, running_var, weight, bias), while only the non-BN parameters
are aggregated across clients. This helps handle data heterogeneity by allowing
each client to adapt its BN layers to its local data distribution.

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


def is_bn_param(param_name: str) -> bool:
    """
    Check if a parameter belongs to a BatchNorm layer.
    
    BatchNorm parameters typically have names containing:
    - 'bn' (batch norm layer name)
    - 'running_mean', 'running_var' (BN statistics)
    - 'num_batches_tracked' (BN tracking counter)
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        True if parameter belongs to BatchNorm layer
    """
    bn_keywords = [
        'bn.weight', 'bn.bias', 
        'bn.running_mean', 'bn.running_var', 
        'bn.num_batches_tracked',
        '.bn.', '_bn.',
        'running_mean', 'running_var', 'num_batches_tracked'
    ]
    
    # Check for explicit BatchNorm parameter patterns
    param_lower = param_name.lower()
    
    # Match common BN naming patterns
    for keyword in bn_keywords:
        if keyword.lower() in param_lower:
            return True
    
    return False


def split_bn_params(arrays: ArrayRecord) -> tuple[dict, dict]:
    """
    Split parameters into BatchNorm and non-BatchNorm parameters.
    
    Args:
        arrays: ArrayRecord containing all model parameters
        
    Returns:
        Tuple of (non_bn_params_dict, bn_params_dict)
    """
    non_bn_params = {}
    bn_params = {}
    
    for key in arrays.keys():
        if is_bn_param(key):
            bn_params[key] = arrays[key]
        else:
            non_bn_params[key] = arrays[key]
    
    return non_bn_params, bn_params


class CustomFedSiloBN(CustomFedAvg):
    """
    Federated Learning with Siloed Batch Normalization (SiloBN) strategy.
    
    This strategy keeps batch normalization layers local to each client,
    only aggregating the non-BN parameters. This helps handle statistical
    heterogeneity across clients by allowing each client to maintain
    BN statistics adapted to its local data distribution.
    
    Key features:
    - Non-BN parameters: Aggregated using weighted averaging (like FedAvg)
    - BN parameters: Kept local on each client (not aggregated)
    - Server maintains a reference set of BN parameters for evaluation
    
    Attributes:
        server_bn_params: BN parameters maintained on server for evaluation
    """
    
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_bn_params: dict | None = None
        self.initial_bn_params: dict | None = None
    
    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedSiloBN settings:")
        log(INFO, "\t│\t└── BN layers: Local (not aggregated)")
        log(INFO, "\t│\t└── Non-BN layers: Aggregated (FedAvg)")
        super().summary()
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        
        # On first round, store initial BN params for server-side evaluation
        if server_round == 1 and self.initial_bn_params is None:
            _, bn_params = split_bn_params(arrays)
            self.initial_bn_params = bn_params
            self.server_bn_params = bn_params
            log(INFO, f"[SiloBN] Stored {len(bn_params)} BN parameters for server evaluation")
        
        # Call parent to handle LR decay and standard config
        return super().configure_train(server_round, arrays, config, grid)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Aggregate only non-BN parameters from client updates.
        
        The aggregation process:
        1. Extract non-BN parameters from each client
        2. Aggregate non-BN parameters using weighted averaging
        3. Combine aggregated non-BN params with server's BN params
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
        non_bn_keys, bn_keys = [], []
        
        for key in first_arrays.keys():
            if is_bn_param(key):
                bn_keys.append(key)
            else:
                non_bn_keys.append(key)
        
        log(INFO, f"[SiloBN] Round {server_round}: Aggregating {len(non_bn_keys)} non-BN params, "
                  f"keeping {len(bn_keys)} BN params local")
        
        # Aggregate only non-BN parameters
        aggregated_non_bn = {}
        
        for key in non_bn_keys:
            weighted_sum = None
            
            for client_arrays, weight in zip(client_arrays_list, normalized_weights):
                param_array = client_arrays[key]
                param_np = Array.to_numpy_ndarray(param_array)
                
                if weighted_sum is None:
                    weighted_sum = weight * param_np
                else:
                    weighted_sum += weight * param_np
            
            aggregated_non_bn[key] = Array.from_numpy_ndarray(weighted_sum)
        
        # Update server BN params by averaging (for server-side evaluation only)
        # This gives a reasonable estimate for central evaluation
        if self.server_bn_params is not None and bn_keys:
            updated_bn_params = {}
            
            for key in bn_keys:
                weighted_sum = None
                
                for client_arrays, weight in zip(client_arrays_list, normalized_weights):
                    if key in client_arrays.keys():
                        param_array = client_arrays[key]
                        param_np = Array.to_numpy_ndarray(param_array)
                        
                        if weighted_sum is None:
                            weighted_sum = weight * param_np
                        else:
                            weighted_sum += weight * param_np
                
                if weighted_sum is not None:
                    updated_bn_params[key] = Array.from_numpy_ndarray(weighted_sum)
                elif key in self.server_bn_params:
                    updated_bn_params[key] = self.server_bn_params[key]
            
            self.server_bn_params = updated_bn_params
        
        # Combine aggregated non-BN params with server BN params for the global model
        combined_params = {**aggregated_non_bn}
        
        if self.server_bn_params is not None:
            combined_params.update(self.server_bn_params)
        
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
