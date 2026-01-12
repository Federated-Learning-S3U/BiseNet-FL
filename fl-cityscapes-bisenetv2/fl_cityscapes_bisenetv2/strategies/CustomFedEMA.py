import json
from logging import INFO
from typing import Dict, Iterable
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
        neg_entropy_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.neg_entropy_weight = neg_entropy_weight
        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedEMA settings:")
        log(INFO, "\t│\t└── Server Momentum (beta): %s", self.server_momentum)
        super().summary()

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Save the current global model before sending it to clients.
        This model will be used as theta_t in the EMA update.
        """
        if self.current_arrays is None:
            # First round initialization
            self.current_arrays = arrays

        config["neg-entropy-weight"] = self.neg_entropy_weight
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Aggregate client updates using FedAvg, then apply server-side EMA.
        """

        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is None or self.current_arrays is None:
            return aggregated_arrays, aggregated_metrics

        beta = self.server_momentum

        # Convert ArrayRecords to dicts of numpy arrays (key-safe)
        old_params: Dict[str, np.ndarray] = {
            k: v
            for k, v in zip(
                self.current_arrays.keys(),
                self.current_arrays.to_numpy_ndarrays(),
                strict=True,
            )
        }

        agg_params: Dict[str, np.ndarray] = {
            k: v
            for k, v in zip(
                aggregated_arrays.keys(),
                aggregated_arrays.to_numpy_ndarrays(),
                strict=True,
            )
        }

        new_params: Dict[str, np.ndarray] = {}

        # Track stats only for trainable parameters
        max_abs_weight = 0.0
        mean_abs_weight_acc = []
        num_tracked = 0

        for key in agg_params.keys():
            old = old_params[key]
            agg = agg_params[key]

            # BatchNorm buffers -> use aggregated values directly
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                new = agg
            else:
                # EMA for trainable parameters
                new = beta * old + (1.0 - beta) * agg

                abs_new = np.abs(new)
                max_abs_weight = max(max_abs_weight, abs_new.max())
                mean_abs_weight_acc.append(abs_new.mean())
                num_tracked += 1

            new_params[key] = new

        mean_abs_weight = (
            float(np.mean(mean_abs_weight_acc)) if num_tracked > 0 else 0.0
        )

        log(
            INFO,
            (
                "FedEMA (round=%d, beta=%.3f): "
                "updated %d tensors | max|w|=%.5f | mean|w|=%.5f"
            ),
            server_round,
            beta,
            num_tracked,
            max_abs_weight,
            mean_abs_weight,
        )

        # Convert back to ArrayRecord
        new_array_record = ArrayRecord(
            {k: Array.from_numpy_ndarray(v) for k, v in new_params.items()}
        )

        # Update server state ONLY after successful aggregation
        self.current_arrays = new_array_record

        return new_array_record, aggregated_metrics
