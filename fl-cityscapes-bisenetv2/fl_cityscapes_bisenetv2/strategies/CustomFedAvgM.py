import json
from logging import INFO
from typing import Iterable
import numpy as np

from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    NDArrays,
    log,
)


class CustomFedAvgM(FedAvg):
    def __init__(
        self,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
        lr_decay_factor: float = 0.9,
        lr_decay_rounds: int = 5,
        lr_schedule_file: str = "./res/lr_schedule.json",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )

        self.current_arrays: ArrayRecord | None = None
        self.momentum_vector: NDArrays | None = None

        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_schedule_file = lr_schedule_file

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        opt_status = "ON" if self.server_opt else "OFF"
        log(INFO, "\t├──> FedAvgM settings:")
        log(INFO, "\t│\t├── Server optimization: %s", opt_status)
        log(INFO, "\t│\t├── Server learning rate: %s", self.server_learning_rate)
        log(INFO, "\t│\t└── Server Momentum: %s", self.server_momentum)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        if self.current_arrays is None:
            self.current_arrays = arrays

        # Decrease learning rate by a factor of 0.9 every 5 rounds
        if server_round % self.lr_decay_rounds == 0 and server_round > 0:
            config["lr"] *= self.lr_decay_factor
            print("LR decreased to:", config["lr"])

            with open(self.lr_schedule_file, "w") as f:
                json.dump({"round": server_round, "lr": config["lr"]}, f)

        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages (with fix)."""

        # Call FedAvg aggregate_train to perform validation and aggregation
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        # The rest of the logic is copied from the original FedAvgM,
        # but with the corrected Array creation.
        if self.server_opt and aggregated_arrays is not None:
            # The initial parameters should be set in `start()` method already
            if self.current_arrays is None:
                from flwr.serverapp.exception import AggregationError

                raise AggregationError(
                    "No initial parameters set for FedAvgM. "
                    "Ensure that `configure_train` has been called before aggregation."
                )
            ndarrays = self.current_arrays.to_numpy_ndarrays()
            aggregated_ndarrays = aggregated_arrays.to_numpy_ndarrays()

            # Preserve keys for arrays in ArrayRecord
            array_keys = list(aggregated_arrays.keys())
            aggregated_arrays.clear()

            # Remember that updates are the opposite of gradients
            pseudo_gradient = [
                old - new
                for new, old in zip(aggregated_ndarrays, ndarrays, strict=True)
            ]
            if self.server_momentum > 0.0:
                if self.momentum_vector is None:
                    # Initialize momentum vector in the first round
                    self.momentum_vector = pseudo_gradient
                else:
                    self.momentum_vector = [
                        self.server_momentum * mv + pg
                        for mv, pg in zip(
                            self.momentum_vector, pseudo_gradient, strict=True
                        )
                    ]

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD and convert back to ArrayRecord
            # *** THIS IS THE FIX: Using Array.from_numpy_ndarray() ***
            updated_array_list = [
                Array.from_numpy_ndarray(np.array(old - self.server_learning_rate * pg))
                for old, pg in zip(ndarrays, pseudo_gradient, strict=True)
            ]
            aggregated_arrays = ArrayRecord(
                dict(zip(array_keys, updated_array_list, strict=True))
            )

            # Update current weights
            self.current_arrays = aggregated_arrays

        return aggregated_arrays, aggregated_metrics
