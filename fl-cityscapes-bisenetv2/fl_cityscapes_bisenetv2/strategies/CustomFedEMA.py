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


class CustomFedEMA(CustomFedAvg):
    """
    Federated Learning with Exponential Moving Average (FedEMA) strategy.

    This strategy updates the global model using an Exponential Moving Average (EMA)
    of the aggregated model updates from clients.

    Equation:
        theta_{t+1} = beta * theta_{t} + (1 - beta) * theta_{agg}

    Where:
        theta_{t+1}: New global model
        theta_{t}: Previous global model
        theta_{agg}: Aggregated model from clients (standard FedAvg result)
        beta: Server momentum factor (0 <= beta < 1)
    """

    def __init__(
        self,
        server_momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedEMA settings:")
        log(INFO, "\t│\t└── Server Momentum (beta): %s", self.server_momentum)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Save current global model for EMA calculation
        self.current_arrays = arrays

        # Call parent to handle LR decay and standard config
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and apply EMA."""

        # Call FedAvg aggregate_train to perform standard aggregation (averaging)
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is not None and self.current_arrays is not None:
            # Convert to numpy for calculation
            old_ndarrays = self.current_arrays.to_numpy_ndarrays()
            agg_ndarrays = aggregated_arrays.to_numpy_ndarrays()

            # Apply EMA only to non-BatchNorm buffers
            # Keep BN running statistics (running_mean, running_var, num_batches_tracked)
            # as they were in the previous global model.
            beta = self.server_momentum
            array_keys = list(aggregated_arrays.keys())
            new_ndarrays = []
            for key, old, agg in zip(
                array_keys, old_ndarrays, agg_ndarrays, strict=True
            ):
                if (
                    "running_mean" in key
                    or "running_var" in key
                    or "num_batches_tracked" in key
                ):
                    # Skip EMA for BN statistics: keep previous global value
                    print(f"FedEMA: Keeping BN buffer '{key}' unchanged.")
                    new_ndarrays.append(old)
                else:
                    # Standard EMA update for trainable parameters
                    print(f"FedEMA: Updating parameter '{key}' with EMA.")
                    new_ndarrays.append(
                        np.array(beta * old + (1 - beta) * agg))

            # Check for potential explosion (on all updated tensors)
            max_weight = max([np.max(np.abs(arr)) for arr in new_ndarrays])
            mean_weight = np.mean([np.mean(np.abs(arr))
                                  for arr in new_ndarrays])
            log(
                INFO,
                f"FedEMA (beta={beta}): Updated global model. Max weight: {max_weight:.4f}, Mean weight: {mean_weight:.4f}",
            )

            # Convert back to ArrayRecord
            aggregated_arrays = ArrayRecord(
                dict(
                    zip(
                        array_keys,
                        [Array.from_numpy_ndarray(arr)
                         for arr in new_ndarrays],
                        strict=True,
                    )
                )
            )

            # Update current arrays
            self.current_arrays = aggregated_arrays

        return aggregated_arrays, aggregated_metrics
