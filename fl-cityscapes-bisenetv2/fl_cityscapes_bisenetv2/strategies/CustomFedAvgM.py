import json
from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvgM
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAvgM(FedAvgM):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.9 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.9
            print("LR decreased to:", config["lr"])

            with open("./res/lr_schedule.json", "w") as f:
                json.dump({"round": server_round, "lr": config["lr"]}, f)

        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)
