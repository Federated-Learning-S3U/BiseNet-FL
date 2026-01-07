"""
SiloBN (Siloed Batch Normalization) Federated Learning Strategy.

In SiloBN, each client maintains its own local batch normalization statistics
(running_mean, running_var), while all learnable parameters including BN's
gamma (weight) and beta (bias) are aggregated across clients. This helps 
handle data heterogeneity by allowing each client to adapt its BN statistics 
to its local data distribution while still learning shared affine transformations.

Key Implementation Details:
- Client-side: Filters out BN statistics before sending to server
- Server-side: Only receives and aggregates learnable parameters
- Client maintains local BN statistics across rounds

Reference:
    Andreux, M., du Terrail, J. O., Beguier, C., & Tramel, E. W. (2020).
    Siloed Federated Learning for Multi-centric Histopathology Datasets.
    In Domain Adaptation and Representation Transfer, and Distributed and
    Collaborative Learning (pp. 129-139). Springer.
"""

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

# Import BN utilities from centralized module
from fl_cityscapes_bisenetv2.utils.bn_utils import (
    is_bn_statistic,
    filter_bn_statistics,
    extract_bn_statistics,
    merge_with_local_bn_stats,
)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# These functions are kept for backward compatibility but delegate to bn_utils
# =============================================================================

def filter_bn_statistics_for_server(state_dict: dict) -> dict:
    """Alias for filter_bn_statistics. Use filter_bn_statistics instead."""
    return filter_bn_statistics(state_dict)


def merge_server_params_with_local_bn(
    server_state_dict: dict,
    local_model_state_dict: dict
) -> dict:
    """
    Merge server parameters with local BN statistics.
    
    Alias that extracts BN stats from local_model_state_dict and merges.
    Consider using merge_with_local_bn_stats directly with pre-extracted stats.
    """
    local_bn_stats = extract_bn_statistics(local_model_state_dict)
    return merge_with_local_bn_stats(server_state_dict, local_bn_stats)


class CustomFedSiloBN(CustomFedAvg):
    """
    Federated Learning with Siloed Batch Normalization (SiloBN) strategy.
    
    This strategy keeps batch normalization STATISTICS (running_mean, running_var)
    completely local to each client - they are NEVER sent to or stored on the server.
    All learnable parameters including BN's gamma (weight) and beta (bias) are 
    aggregated across clients.
    
    IMPORTANT: This strategy requires client-side filtering!
    Clients MUST use `filter_bn_statistics_for_server()` before sending parameters
    to ensure BN statistics are not transmitted.
    
    Key features:
    - Learnable parameters (Conv, FC, BN gamma/beta): Aggregated using weighted averaging
    - BN statistics (running_mean, running_var): NEVER leave the client
    - Server has NO knowledge of client BN statistics
    
    What gets aggregated (sent to server):
    - All convolutional layer weights and biases
    - All fully connected layer weights and biases  
    - BN weight (gamma) - learnable scale parameter
    - BN bias (beta) - learnable shift parameter
    
    What stays local (never sent to server):
    - running_mean - running mean estimate
    - running_var - running variance estimate  
    - num_batches_tracked - batch counter
    """
    
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
    
    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedSiloBN settings:")
        log(INFO, "\t│\t└── BN statistics (running_mean/var): Siloed (never leave client)")
        log(INFO, "\t│\t└── BN gamma/beta: Aggregated (learnable)")
        log(INFO, "\t│\t└── Non-BN layers: Aggregated (FedAvg)")
        log(INFO, "\t│\t└── Note: Clients must filter BN stats before sending")
        super().summary()
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """
        Configure the next round of federated training.
        
        In SiloBN, the server sends only learnable parameters to clients.
        Clients will merge these with their local BN statistics.
        """
        if server_round == 1:
            # Count params for logging
            aggregatable_count = sum(1 for k in arrays.keys() if not is_bn_statistic(k))
            bn_stat_count = sum(1 for k in arrays.keys() if is_bn_statistic(k))
            log(INFO, f"[SiloBN] Round {server_round}: Server model has {aggregatable_count} "
                      f"aggregatable params and {bn_stat_count} BN statistics")
            log(INFO, "[SiloBN] Note: BN statistics in server model are from initialization only")
        
        # Call parent to handle LR decay and standard config
        return super().configure_train(server_round, arrays, config, grid)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Aggregate only learnable parameters from clients.
        
        In true SiloBN:
        - Clients should have already filtered out BN statistics before sending
        - Server only receives and aggregates learnable parameters
        - Server does NOT store or aggregate BN statistics
        
        The aggregation process:
        1. Receive learnable params (Conv, FC, BN gamma/beta) from each client
        2. Aggregate using weighted averaging
        3. Return aggregated learnable params (no BN statistics)
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
        
        # Get all parameter keys from first client
        # If client properly filtered, there should be no BN statistics
        first_arrays = client_arrays_list[0]
        all_keys = list(first_arrays.keys())
        
        # Check if clients properly filtered BN statistics (for logging/debugging)
        bn_stats_received = [k for k in all_keys if is_bn_statistic(k)]
        aggregatable_keys = [k for k in all_keys if not is_bn_statistic(k)]
        
        if bn_stats_received:
            log(INFO, f"[SiloBN] WARNING: Received {len(bn_stats_received)} BN statistics "
                      f"from clients. These should be filtered on client side!")
            log(INFO, f"[SiloBN] BN stats received (will be ignored): {bn_stats_received[:5]}...")
        
        log(INFO, f"[SiloBN] Round {server_round}: Aggregating {len(aggregatable_keys)} "
                  f"learnable params (Conv, FC, BN gamma/beta)")
        
        # Aggregate only learnable parameters (exclude any BN statistics that may have been sent)
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
        
        aggregated_arrays = ArrayRecord(aggregated_params)
        
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
