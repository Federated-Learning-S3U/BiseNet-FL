import json
from logging import INFO
from typing import Iterable
import numpy as np

from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        lr_decay_factor: float = 0.9,
        lr_decay_rounds: int = 5,
        lr_schedule_file: str = "./res/lr_schedule.json",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_schedule_file = lr_schedule_file

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
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
        """Aggregate ArrayRecords and log weight statistics."""

        # Call FedAvg aggregate_train to perform standard aggregation
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is not None:
            # Convert to numpy for statistics
            agg_ndarrays = aggregated_arrays.to_numpy_ndarrays()

            # Check for potential explosion
            max_weight = max([np.max(np.abs(arr)) for arr in agg_ndarrays])
            mean_weight = np.mean([np.mean(np.abs(arr))
                                  for arr in agg_ndarrays])
            log(INFO,
                f"FedAvg: Updated global model. Max weight: {max_weight:.4f}, Mean weight: {mean_weight:.4f}")

        return aggregated_arrays, aggregated_metrics
