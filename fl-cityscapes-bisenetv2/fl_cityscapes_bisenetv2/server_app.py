"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from lib.models import BiSeNetV2

from fl_cityscapes_bisenetv2.task import make_central_evaluate, make_silobn_evaluate_aggregator
from fl_cityscapes_bisenetv2.strategies import (
    CustomFedAvg,
    CustomFedProx,
    CustomFedAvgM,
    CustomFedEMA,
    CustomFedSiloBN,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]

    num_classes: int = context.run_config["num-classes"]

    resume = context.run_config["resume"]
    pretrained_path = context.run_config["pretrained-path"]
    rounds_trained = context.run_config["rounds-trained"]

    lr: float = context.run_config["lr"]
    lr_decay_factor: float = context.run_config["lr-decay-factor"]
    lr_decay_rounds: int = context.run_config["lr-decay-rounds"]
    lr_schedule_file: str = context.run_config["lr-schedule-file"]

    strategy_name: str = context.run_config["strategy-name"]
    custom_strategy_name = "Custom" + strategy_name

    # Read Strategy Params
    strategy_params = {}
    if strategy_name == "FedProx":
        strategy_params["proximal_mu"] = context.run_config["proximity-mu"]
    elif strategy_name == "FedAvgM":
        strategy_params["server_momentum"] = context.run_config["server-momentum"]
        strategy_params["server_learning_rate"] = context.run_config[
            "server-learning-rate"
        ]
    elif strategy_name == "FedEMA":
        strategy_params["server_momentum"] = context.run_config["server-momentum"]
    elif strategy_name == "FedSiloBN":
        strategy_params["silobn_eval_aggregator"] = make_silobn_evaluate_aggregator(context)
        # Pass eval_interval and rounds_trained for evaluation scheduling & resume support
        strategy_params["eval_interval"] = context.run_config["eval-interval"]
        strategy_params["rounds_trained"] = rounds_trained
        fraction_evaluate = 1.0  # Evaluate on all participating clients
        print(f"[Server] SiloBN: Client-side evaluation enabled (fraction_evaluate={fraction_evaluate})")
        print(f"[Server] SiloBN: Evaluation interval = {context.run_config['eval-interval']} rounds")


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
    strategy = eval(custom_strategy_name)(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        lr_schedule_file=lr_schedule_file,
        lr_decay_factor=lr_decay_factor,
        lr_decay_rounds=lr_decay_rounds,
        **strategy_params,
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
        f.write(str(results))
