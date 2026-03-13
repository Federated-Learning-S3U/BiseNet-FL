import json
from typing import Iterable

from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message

from fl_cityscapes_bisenetv2.utils.checkpoint_utils import save_global_model


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        lr_decay_factor: float = 0.9,
        lr_decay_rounds: int = 5,
        lr_schedule_file: str = "./res/lr_schedule.json",
        base_path: str = "res",
        partition_name: str = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_schedule_file = lr_schedule_file
        self.base_path = base_path
        self.partition_name = partition_name

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Save global model after aggregation (arrays contains the aggregated model)
        try:
            global_state_dict = arrays.to_torch_state_dict()
            save_global_model(
                model_state_dict=global_state_dict,
                base_path=self.base_path,
                partition_name=self.partition_name,
                server_round=server_round,
            )
            print(f"[Server] Saved global model for round {server_round}.")
        except Exception as e:
            print(f"[Server] Warning: Failed to save global model: {e}")

        # Decrease learning rate by a factor of 0.9 every 5 rounds
        if server_round % self.lr_decay_rounds == 0 and server_round > 0:
            config["lr"] *= self.lr_decay_factor
            print("LR decreased to:", config["lr"])

            with open(self.lr_schedule_file, "w") as f:
                json.dump({"round": server_round, "lr": config["lr"]}, f)

        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)
