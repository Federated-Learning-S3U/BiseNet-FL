"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from lib.models import BiSeNetV2

from fl_cityscapes_bisenetv2.task import make_central_evaluate
from fl_cityscapes_bisenetv2.strategies import (
    CustomFedAvg,
    CustomFedProx,
    CustomFedAvgM,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config["fraction-train"]

    num_classes: int = context.run_config["num-classes"]

    resume = context.run_config["resume"]
    pretrained_path = context.run_config["pretrained-path"]
    rounds_trained = context.run_config["rounds-trained"]

    lr: float = context.run_config["lr"]
    lr_decay_factor: float = context.run_config["lr-decay-factor"]
    lr_decay_rounds: int = context.run_config["lr-decay-rounds"]
    lr_schedule_file: str = context.run_config["lr-schedule-file"]

    strategy_name: str = context.run_config["strategy-name"]

    # Load global model
    global_model = BiSeNetV2(num_classes)

    # Load pretrained model if resuming
    if resume:
        print(f"[Server] Resuming from pretrained model at {pretrained_path}")
        sd = torch.load(pretrained_path, map_location="cpu")
        global_model.load_state_dict(sd, strict=True)
        print(f"[Server] Pretrained model trained on {rounds_trained} rounds.")
    # Else start from random initialized model
    else:
        print(f"[Server] Starting from random initialized model.")
        print(f"[Server] Initial model trained on {rounds_trained} rounds.")

    arrays = ArrayRecord(global_model.state_dict())

    # Create central evaluation function that accepts context as an argument
    evaluate_fn = make_central_evaluate(context)

    # Initialize Custom strategy
    custom_strategy_name = "Custom" + strategy_name
    strategy = eval(custom_strategy_name)(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,
        lr_schedule_file=lr_schedule_file,
        lr_decay_factor=lr_decay_factor,
        lr_decay_rounds=lr_decay_rounds,
    )

    # Start strategy
    results = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    with open(context.run_config["server-results-file"], "w") as f:
        json.dump(results, f, indent=4)
