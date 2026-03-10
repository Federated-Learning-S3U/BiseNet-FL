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


class CustomFedAvgM(CustomFedAvg):
    """
    Federated Learning with Server-Side Momentum (FedAvgM) strategy.

    This strategy applies momentum to the server-side model updates,
    accumulating velocity across rounds.

    Update rule:
        delta = w_avg_clients - w_global
        v = beta * v + delta
        w_global = w_global + server_lr * v

    Where:
        w_global: Current global model
        w_avg_clients: Weighted average of client models (standard FedAvg result)
        delta: Update/difference between aggregated and current model
        v: Momentum buffer (velocity)
        beta: Momentum coefficient (0 <= beta < 1)
        server_lr: Server learning rate
    """

    def __init__(
        self,
        server_momentum: float = 0.9,
        server_learning_rate: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.server_learning_rate = server_learning_rate
        self.current_arrays: ArrayRecord | None = None
        self.momentum_buffer: Dict[str, np.ndarray] | None = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedAvgM settings:")
        log(INFO, "\t│\t├── Server Momentum (beta): %s", self.server_momentum)
        log(INFO, "\t│\t└── Server Learning Rate: %s", self.server_learning_rate)
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
        This model will be used as w_global in the momentum update.
        """
        if self.current_arrays is None:
            # First round initialization
            self.current_arrays = arrays

        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Aggregate client updates using FedAvg, then apply server-side momentum.
        """

        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is None or self.current_arrays is None:
            return aggregated_arrays, aggregated_metrics

        beta = self.server_momentum
        server_lr = self.server_learning_rate

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

        # Initialize momentum buffer to zeros on first round
        if self.momentum_buffer is None:
            self.momentum_buffer = {k: np.zeros_like(
                v) for k, v in old_params.items()}

        new_params: Dict[str, np.ndarray] = {}

        # Track stats only for trainable parameters
        max_abs_weight = 0.0
        mean_abs_weight_acc = []
        num_tracked = 0

        for key in agg_params.keys():
            old = old_params[key]
            agg = agg_params[key]

            # BatchNorm buffers -> use aggregated values directly (no momentum)
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
            ):
                new = agg
            else:
                # Compute delta: update direction
                delta = agg - old

                # Update momentum buffer: v = beta * v + delta
                self.momentum_buffer[key] = beta * \
                    self.momentum_buffer[key] + delta

                # Apply update: w_new = w_old + server_lr * v
                new = old + server_lr * self.momentum_buffer[key]

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
                "FedAvgM (round=%d, beta=%.3f, lr=%.3f): "
                "updated %d tensors | max|w|=%.5f | mean|w|=%.5f"
            ),
            server_round,
            beta,
            server_lr,
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
